from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from locomaniformer.generation.artifacts import GeneratedRobotArtifact
from locomaniformer.generation.config import RobotFamily
from locomaniformer.generation.metadata import ActionDescriptor, ObservationDescriptor
from locomaniformer.generation.specs import (
    ActuationSpec,
    BodySpec,
    FootSpec,
    JointSpec,
    LimbSpec,
    ManipulatorSpec,
    PhysicsSpec,
    RobotMorphologySpec,
    SegmentSpec,
    SensorSpec,
    SymmetrySpec,
    WheelSpec,
)
from locomaniformer.generation.validation import ValidationResult


@dataclass(frozen=True)
class BootstrapControllerConfig:
    seed: int = 0
    candidates: int = 32
    horizon: float = 1.5
    effort_penalty: float = 0.01
    fall_height_threshold: float = 0.18
    controller_version: str = "bootstrap_cpg.v1"
    control_timestep: float = 0.02
    objective: str = "forward"


@dataclass(frozen=True)
class BootstrapControllerArtifact:
    robot_id: str
    controller_version: str
    seed: int
    objective: str
    parameters: dict[str, Any]
    action_mapping: list[dict[str, Any]]
    score: float
    evaluation_summary: dict[str, float | int | str | bool]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def write(self, output_dir: Path | str) -> Path:
        root = Path(output_dir)
        controller_dir = root / self.robot_id
        controller_dir.mkdir(parents=True, exist_ok=True)
        path = controller_dir / "controller.json"
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
        return path


def load_generated_robot_artifact(path: Path | str) -> GeneratedRobotArtifact:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    spec = _spec_from_dict(payload["morphology_spec"])
    actions = tuple(ActionDescriptor(**item) for item in payload["action_descriptor"])
    observations = tuple(
        ObservationDescriptor(**item) for item in payload["observation_descriptor"]
    )
    validation = ValidationResult(**payload["validation_result"])
    return GeneratedRobotArtifact(
        robot_id=payload["robot_id"],
        seed=int(payload["seed"]),
        generator_version=payload["generator_version"],
        family=payload["family"],
        morphology_spec=spec,
        mjcf_xml=payload["mjcf_xml"],
        mjcf_assets={},
        action_descriptor=actions,
        observation_descriptor=observations,
        embodiment_metadata=payload["embodiment_metadata"],
        validation_result=validation,
        summary_statistics=payload["summary_statistics"],
    )


def load_manifest_generated_robot_artifacts(
    manifest_path: Path | str,
    robot_root: Path | str,
    *,
    include_rejected: bool = False,
) -> tuple[GeneratedRobotArtifact, ...]:
    manifest = Path(manifest_path)
    root = Path(robot_root)
    artifacts: list[GeneratedRobotArtifact] = []
    for line_number, line in enumerate(
        manifest.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        entry = json.loads(line)
        if not include_rejected and not entry.get("accepted", False):
            continue
        if "artifact_path" in entry:
            artifact_path = Path(entry["artifact_path"])
            if not artifact_path.is_absolute():
                artifact_path = manifest.parent / artifact_path
        else:
            try:
                artifact_path = root / entry["robot_id"] / "artifact.json"
            except KeyError as exc:
                msg = f"manifest line {line_number} has no robot_id or artifact_path"
                raise ValueError(msg) from exc
        if not artifact_path.exists():
            msg = f"manifest line {line_number} artifact does not exist: {artifact_path}"
            raise FileNotFoundError(msg)
        artifacts.append(load_generated_robot_artifact(artifact_path))
    return tuple(artifacts)


def _spec_from_dict(payload: dict[str, Any]) -> RobotMorphologySpec:
    limbs = tuple(_limb_from_dict(item) for item in payload["limbs"])
    manipulators = tuple(_manipulator_from_dict(item) for item in payload["manipulators"])
    return RobotMorphologySpec(
        robot_id=payload["robot_id"],
        seed=int(payload["seed"]),
        config_hash=payload["config_hash"],
        family=RobotFamily(payload["family"]),
        global_scale=float(payload["global_scale"]),
        body=BodySpec(**payload["body"]),
        limbs=limbs,
        wheels=tuple(_wheel_from_dict(item) for item in payload["wheels"]),
        manipulators=manipulators,
        sensors=tuple(SensorSpec(**item) for item in payload["sensors"]),
        actuation=ActuationSpec(**payload["actuation"]),
        physics=PhysicsSpec(**payload["physics"]),
        symmetry=SymmetrySpec(**payload["symmetry"]),
        source_ranges=payload.get("source_ranges", {}),
    )


def _limb_from_dict(payload: dict[str, Any]) -> LimbSpec:
    return LimbSpec(
        name=payload["name"],
        side=payload["side"],
        mount_label=payload["mount_label"],
        mount_pos=tuple(payload["mount_pos"]),
        upper=SegmentSpec(**payload["upper"]),
        lower=SegmentSpec(**payload["lower"]),
        joints=tuple(JointSpec(**item) for item in payload["joints"]),
        foot=_foot_from_dict(payload["foot"]) if payload.get("foot") is not None else None,
        wheel=_wheel_from_dict(payload["wheel"]) if payload.get("wheel") is not None else None,
        symmetry_group=payload["symmetry_group"],
    )


def _foot_from_dict(payload: dict[str, Any]) -> FootSpec:
    return FootSpec(
        shape=payload["shape"],
        size=tuple(payload["size"]),
        local_offset=tuple(payload["local_offset"]),
        mass=float(payload["mass"]),
        friction=tuple(payload["friction"]),
        contact_sensor_count=int(payload["contact_sensor_count"]),
    )


def _wheel_from_dict(payload: dict[str, Any]) -> WheelSpec:
    return WheelSpec(
        radius=float(payload["radius"]),
        width=float(payload["width"]),
        mass=float(payload["mass"]),
        local_axle_offset=tuple(payload["local_axle_offset"]),
        axle_axis=tuple(payload["axle_axis"]),
        drive_torque_range=tuple(payload["drive_torque_range"]),
        steerable=bool(payload["steerable"]),
        passive=bool(payload["passive"]),
    )


def _manipulator_from_dict(payload: dict[str, Any]) -> ManipulatorSpec:
    return ManipulatorSpec(
        name=payload["name"],
        side=payload["side"],
        mount_pos=tuple(payload["mount_pos"]),
        links=tuple(SegmentSpec(**item) for item in payload["links"]),
        joints=tuple(JointSpec(**item) for item in payload["joints"]),
        gripper_type=payload["gripper_type"],
        end_effector_mass=float(payload["end_effector_mass"]),
        end_effector_radius=float(payload["end_effector_radius"]),
    )
