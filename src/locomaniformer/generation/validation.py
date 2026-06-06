from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

import mujoco
import numpy as np

from locomaniformer.generation.config import RobotGenerationConfig
from locomaniformer.generation.mjcf_robot import MJCFGeneratedRobot
from locomaniformer.generation.specs import JointSpec, RobotMorphologySpec


@dataclass(frozen=True)
class ValidationResult:
    accepted: bool
    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metrics: dict[str, float | int | str | bool] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class MorphologyValidator:
    def __init__(self, config: RobotGenerationConfig) -> None:
        self.config = config

    def validate(
        self,
        spec: RobotMorphologySpec,
        robot: MJCFGeneratedRobot | None = None,
    ) -> ValidationResult:
        reasons: list[str] = []
        warnings: list[str] = []
        metrics = summary_statistics(spec)

        self._validate_numeric_values(spec.to_dict(), reasons)
        self._validate_joints(spec, reasons)
        self._validate_names(spec, reasons)
        if not (
            self.config.total_mass_range[0]
            <= metrics["total_mass"]
            <= self.config.total_mass_range[1]
        ):
            reasons.append(f"total mass is outside configured bounds: {metrics['total_mass']:.3f}")
        if self.config.require_contact_sensors:
            for limb in spec.limbs:
                if limb.foot is not None or limb.wheel is not None:
                    expected = f"{limb.name}_touch"
                    if expected not in {sensor.name for sensor in spec.sensors}:
                        reasons.append(f"missing distal contact sensor {expected!r}")

        robot = robot or MJCFGeneratedRobot(spec)
        try:
            mj_model = mujoco.MjModel.from_xml_string(
                robot.get_mjcf_str(),
                assets=robot.get_mjcf_assets(),
            )
            mj_data = mujoco.MjData(mj_model)
            mujoco.mj_resetData(mj_model, mj_data)
            mujoco.mj_forward(mj_model, mj_data)
            metrics["mjc_ok"] = True
            metrics["actuator_count"] = int(mj_model.nu)
            metrics["sensor_count"] = int(mj_model.nsensor)
        except Exception as exc:  # noqa: BLE001 - validation should return structured failures.
            metrics["mjc_ok"] = False
            reasons.append(f"MuJoCo compile/reset failed: {exc}")

        if len(robot.actuator_names) != metrics.get("actuator_count", len(robot.actuator_names)):
            warnings.append("builder actuator count differs from compiled MuJoCo actuator count")

        if self.config.require_mjx and metrics.get("mjc_ok"):
            try:
                from mujoco import mjx

                mjx.put_model(mj_model)
                metrics["mjx_ok"] = True
            except Exception as exc:  # noqa: BLE001
                metrics["mjx_ok"] = False
                reasons.append(f"MJX compatibility check failed: {exc}")
        else:
            metrics["mjx_ok"] = False

        return ValidationResult(
            accepted=not reasons,
            reasons=reasons,
            warnings=warnings,
            metrics=metrics,
        )

    def _validate_numeric_values(self, payload: Any, reasons: list[str]) -> None:
        if isinstance(payload, dict):
            for value in payload.values():
                self._validate_numeric_values(value, reasons)
        elif isinstance(payload, list | tuple):
            for value in payload:
                self._validate_numeric_values(value, reasons)
        elif isinstance(payload, float) and not math.isfinite(payload):
            reasons.append("sampled value is NaN or infinite")

    def _validate_joints(self, spec: RobotMorphologySpec, reasons: list[str]) -> None:
        joints: list[JointSpec] = []
        for limb in spec.limbs:
            joints.extend(limb.joints)
            if limb.wheel is not None and np.linalg.norm(limb.wheel.axle_axis) <= 1e-8:
                reasons.append(f"{limb.name} wheel axle axis is zero")
        for manipulator in spec.manipulators:
            joints.extend(manipulator.joints)
        for joint in joints:
            if np.linalg.norm(joint.axis) <= 1e-8:
                reasons.append(f"{joint.name_suffix} joint axis is zero")
            if joint.range[0] >= joint.range[1]:
                reasons.append(f"{joint.name_suffix} joint range is not ordered")
            if abs(joint.range[1] - joint.range[0]) <= 1e-5:
                reasons.append(f"{joint.name_suffix} joint range is degenerate")

    def _validate_names(self, spec: RobotMorphologySpec, reasons: list[str]) -> None:
        names: list[str] = ["torso"]
        names.extend(limb.name for limb in spec.limbs)
        names.extend(
            f"{limb.name}_{joint.name_suffix}" for limb in spec.limbs for joint in limb.joints
        )
        names.extend(manipulator.name for manipulator in spec.manipulators)
        names.extend(sensor.name for sensor in spec.sensors)
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            reasons.append(f"duplicate generated names: {', '.join(duplicates)}")


def summary_statistics(spec: RobotMorphologySpec) -> dict[str, float | int | str | bool]:
    total_mass = spec.body.mass
    total_mass += sum(limb.upper.mass + limb.lower.mass for limb in spec.limbs)
    total_mass += sum(limb.foot.mass for limb in spec.limbs if limb.foot is not None)
    total_mass += sum(limb.wheel.mass for limb in spec.limbs if limb.wheel is not None)
    total_mass += sum(
        sum(link.mass for link in manipulator.links) + manipulator.end_effector_mass
        for manipulator in spec.manipulators
    )
    average_limb_length = sum(limb.upper.length + limb.lower.length for limb in spec.limbs) / max(
        1, len(spec.limbs)
    )
    return {
        "robot_id": spec.robot_id,
        "family": spec.family.value,
        "total_mass": float(total_mass),
        "body_height": float(spec.body.height + average_limb_length),
        "limb_count": len(spec.limbs),
        "actuator_count": sum(len(limb.joints) for limb in spec.limbs)
        + sum(1 + int(limb.wheel.steerable) for limb in spec.limbs if limb.wheel is not None)
        + sum(len(manipulator.joints) for manipulator in spec.manipulators),
        "sensor_count": len(spec.sensors),
        "wheel_count": len(spec.wheels),
        "manipulator_count": len(spec.manipulators),
    }
