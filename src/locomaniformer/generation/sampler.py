from __future__ import annotations

from dataclasses import replace

import numpy as np

from locomaniformer.generation.config import RobotFamily, RobotGenerationConfig
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
    robot_id_for,
)


class RobotFamilySampler:
    def __init__(self, config: RobotGenerationConfig) -> None:
        self.config = config

    def sample(
        self,
        seed: int,
        *,
        family: RobotFamily | str | None = None,
    ) -> RobotMorphologySpec:
        rng = np.random.default_rng(seed)
        robot_family = self._choose_family(rng, family)
        global_scale = self._uniform(rng, *self.config.global_scale_range)
        body = self._sample_body(rng, global_scale)
        physics = self._sample_physics(rng)
        actuation = self._sample_actuation(rng, global_scale)
        limbs = self._sample_limbs(rng, robot_family, body, global_scale, physics)
        manipulators = self._sample_manipulators(rng, body, global_scale, physics)
        sensors = self._sensor_specs(limbs, manipulators)
        wheels = tuple(limb.wheel for limb in limbs if limb.wheel is not None)
        symmetry = SymmetrySpec(
            mirrored=True,
            perturbation_scale=0.025
            if self.config.parameter_range_preset == "conservative"
            else 0.06,
            groups=tuple(sorted({limb.symmetry_group for limb in limbs})),
        )

        provisional = RobotMorphologySpec(
            robot_id="pending",
            seed=seed,
            config_hash=self.config.stable_hash(),
            family=robot_family,
            global_scale=global_scale,
            body=body,
            limbs=limbs,
            wheels=wheels,
            manipulators=manipulators,
            sensors=sensors,
            actuation=actuation,
            physics=physics,
            symmetry=symmetry,
            source_ranges={
                "global_scale": self.config.parameter_range_preset.value,
                "body": self.config.parameter_range_preset.value,
                "limbs": self.config.parameter_range_preset.value,
                "physics": self.config.parameter_range_preset.value,
            },
        )
        robot_id = robot_id_for(
            generator_version=self.config.generator_version,
            config_hash=self.config.stable_hash(),
            seed=seed,
            family=robot_family,
            payload=provisional.to_dict(),
        )
        return replace(provisional, robot_id=robot_id)

    def _choose_family(
        self,
        rng: np.random.Generator,
        family: RobotFamily | str | None,
    ) -> RobotFamily:
        if family is not None:
            requested = RobotFamily(family)
            if requested not in self.config.allowed_families:
                msg = f"{requested.value!r} is not allowed by the generation config"
                raise ValueError(msg)
            return requested
        return rng.choice(self.config.allowed_families).item()

    def _sample_body(self, rng: np.random.Generator, scale: float) -> BodySpec:
        length = self._uniform(rng, 0.42, 0.82) * scale
        width = self._uniform(rng, 0.18, 0.38) * scale
        height = self._uniform(rng, 0.14, 0.30) * scale
        density = self._uniform(rng, 260.0, 520.0)
        mass = length * width * height * density
        return BodySpec(
            shape="box",
            length=length,
            width=width,
            height=height,
            mass=max(mass, 3.0 * scale),
            density=density,
            com_offset=(
                self._uniform(rng, -0.02, 0.02) * scale,
                self._uniform(rng, -0.015, 0.015) * scale,
                self._uniform(rng, -0.01, 0.015) * scale,
            ),
            shoulder_mount_offset=(length * 0.22, width * 0.58, height * 0.1),
        )

    def _sample_physics(self, rng: np.random.Generator) -> PhysicsSpec:
        return PhysicsSpec(
            friction=(self._uniform(rng, 0.8, 1.4), 0.04, 0.002),
            damping=self._log_uniform(rng, 0.08, 0.7),
            armature=self._log_uniform(rng, 0.005, 0.05),
            density=self._uniform(rng, 450.0, 900.0),
            restitution=self._uniform(rng, 0.0, 0.02),
        )

    def _sample_actuation(self, rng: np.random.Generator, scale: float) -> ActuationSpec:
        strength = self._uniform(rng, 0.85, 1.35) * scale
        return ActuationSpec(
            actuator_type="motor",
            strength_multiplier=strength,
            ctrl_range=(-1.0, 1.0),
            force_range=(-120.0 * strength, 120.0 * strength),
        )

    def _sample_limbs(
        self,
        rng: np.random.Generator,
        family: RobotFamily,
        body: BodySpec,
        scale: float,
        physics: PhysicsSpec,
    ) -> tuple[LimbSpec, ...]:
        is_quad = family in (RobotFamily.QUADRUPED, RobotFamily.WHEELED_QUADRUPED)
        is_wheeled = family in (RobotFamily.WHEELED_BIPED, RobotFamily.WHEELED_QUADRUPED)
        templates = ("hip_roll", "hip_pitch", "knee_pitch")
        if rng.random() > 0.45:
            templates = ("hip_yaw",) + templates
        if not is_wheeled and rng.random() > 0.7:
            templates = templates + ("ankle_pitch",)

        x_positions = (-body.length * 0.28, body.length * 0.28) if is_quad else (0.0,)
        y_positions = (-body.width * 0.62, body.width * 0.62)
        limbs: list[LimbSpec] = []
        for x_pos in x_positions:
            for y_pos in y_positions:
                side = "left" if y_pos > 0 else "right"
                mount_label = ("front" if x_pos > 0 else "rear") if is_quad else "mid"
                name = f"{mount_label}_{side}_leg" if is_quad else f"{side}_leg"
                perturb = self._uniform(rng, -0.025, 0.025) * scale
                upper_length = self._uniform(rng, 0.22, 0.42) * scale
                lower_length = self._uniform(rng, 0.20, 0.40) * scale
                radius = self._uniform(rng, 0.025, 0.055) * scale
                upper = SegmentSpec(
                    length=upper_length,
                    radius=radius,
                    mass=max(0.35, upper_length * radius * physics.density * 0.22),
                )
                lower = SegmentSpec(
                    length=lower_length,
                    radius=radius * self._uniform(rng, 0.82, 1.0),
                    mass=max(0.25, lower_length * radius * physics.density * 0.18),
                )
                joints = tuple(self._joint_for(template, physics) for template in templates)
                foot = None if is_wheeled else self._sample_foot(rng, scale, physics)
                wheel = self._sample_wheel(rng, scale) if is_wheeled else None
                limbs.append(
                    LimbSpec(
                        name=name,
                        side=side,
                        mount_label=mount_label,
                        mount_pos=(x_pos + perturb, y_pos, -body.height * 0.18),
                        upper=upper,
                        lower=lower,
                        joints=joints,
                        foot=foot,
                        wheel=wheel,
                        symmetry_group=f"{mount_label}_pair",
                    )
                )
        return tuple(limbs)

    def _joint_for(self, template: str, physics: PhysicsSpec) -> JointSpec:
        axes = {
            "hip_yaw": (0.0, 0.0, 1.0),
            "hip_roll": (1.0, 0.0, 0.0),
            "hip_pitch": (0.0, 1.0, 0.0),
            "knee_pitch": (0.0, 1.0, 0.0),
            "ankle_pitch": (0.0, 1.0, 0.0),
        }
        ranges = {
            "hip_yaw": (-0.55, 0.55),
            "hip_roll": (-0.65, 0.65),
            "hip_pitch": (-1.1, 0.9),
            "knee_pitch": (-1.8, -0.05),
            "ankle_pitch": (-0.8, 0.8),
        }
        return JointSpec(
            name_suffix=template,
            axis=axes[template],
            range=ranges[template],
            damping=physics.damping,
            armature=physics.armature,
        )

    def _sample_foot(
        self,
        rng: np.random.Generator,
        scale: float,
        physics: PhysicsSpec,
    ) -> FootSpec:
        return FootSpec(
            shape="box",
            size=(
                self._uniform(rng, 0.055, 0.13) * scale,
                self._uniform(rng, 0.035, 0.075) * scale,
                self._uniform(rng, 0.018, 0.035) * scale,
            ),
            local_offset=(self._uniform(rng, -0.02, 0.035) * scale, 0.0, -0.02 * scale),
            mass=self._uniform(rng, 0.08, 0.35) * scale,
            friction=physics.friction,
            contact_sensor_count=1,
        )

    def _sample_wheel(self, rng: np.random.Generator, scale: float) -> WheelSpec:
        radius = self._uniform(rng, 0.08, 0.17) * scale
        width = self._uniform(rng, 0.035, 0.075) * scale
        return WheelSpec(
            radius=radius,
            width=width,
            mass=max(0.12, radius * width * 8.0),
            local_axle_offset=(0.0, 0.0, -radius * 0.2),
            axle_axis=(0.0, 1.0, 0.0),
            drive_torque_range=(-70.0 * scale, 70.0 * scale),
            steerable=bool(rng.random() > 0.72),
            passive=False,
        )

    def _sample_manipulators(
        self,
        rng: np.random.Generator,
        body: BodySpec,
        scale: float,
        physics: PhysicsSpec,
    ) -> tuple[ManipulatorSpec, ...]:
        if rng.random() >= self.config.manipulator_probability:
            return ()
        allowed = tuple(count for count in self.config.allowed_manipulator_counts if count > 0)
        if not allowed:
            return ()
        count = int(rng.choice(allowed))
        sides = ("left", "right") if count == 2 else (rng.choice(("left", "right")).item(),)
        manipulators: list[ManipulatorSpec] = []
        for side in sides:
            y_sign = 1.0 if side == "left" else -1.0
            links = (
                SegmentSpec(
                    length=self._uniform(rng, 0.16, 0.28) * scale,
                    radius=self._uniform(rng, 0.018, 0.035) * scale,
                    mass=self._uniform(rng, 0.12, 0.35) * scale,
                ),
                SegmentSpec(
                    length=self._uniform(rng, 0.14, 0.25) * scale,
                    radius=self._uniform(rng, 0.014, 0.03) * scale,
                    mass=self._uniform(rng, 0.08, 0.28) * scale,
                ),
            )
            joints = (
                JointSpec(
                    "shoulder_pitch",
                    (0.0, 1.0, 0.0),
                    (-1.4, 1.4),
                    physics.damping,
                    physics.armature,
                ),
                JointSpec(
                    "elbow_pitch", (0.0, 1.0, 0.0), (-1.7, 0.05), physics.damping, physics.armature
                ),
            )
            manipulators.append(
                ManipulatorSpec(
                    name=f"{side}_arm",
                    side=side,
                    mount_pos=(
                        body.shoulder_mount_offset[0],
                        body.shoulder_mount_offset[1] * y_sign,
                        body.shoulder_mount_offset[2],
                    ),
                    links=links,
                    joints=joints,
                    gripper_type="spherical_end_effector",
                    end_effector_mass=self._uniform(rng, 0.05, 0.18) * scale,
                    end_effector_radius=self._uniform(rng, 0.025, 0.055) * scale,
                )
            )
        return tuple(manipulators)

    def _sensor_specs(
        self,
        limbs: tuple[LimbSpec, ...],
        manipulators: tuple[ManipulatorSpec, ...],
    ) -> tuple[SensorSpec, ...]:
        sensors: list[SensorSpec] = [
            SensorSpec("torso_orientation", "framequat", "torso_imu", "torso", (4,)),
            SensorSpec("torso_angular_velocity", "gyro", "torso_imu", "torso", (3,)),
        ]
        for limb in limbs:
            for joint in limb.joints:
                joint_name = f"{limb.name}_{joint.name_suffix}"
                sensors.append(SensorSpec(f"{joint_name}_pos", "jointpos", joint_name, limb.name))
                sensors.append(SensorSpec(f"{joint_name}_vel", "jointvel", joint_name, limb.name))
            if limb.foot is not None:
                sensors.append(
                    SensorSpec(f"{limb.name}_touch", "touch", f"{limb.name}_contact", limb.name)
                )
            if limb.wheel is not None:
                sensors.append(
                    SensorSpec(
                        f"{limb.name}_wheel_vel", "jointvel", f"{limb.name}_wheel_drive", limb.name
                    )
                )
                sensors.append(
                    SensorSpec(f"{limb.name}_touch", "touch", f"{limb.name}_contact", limb.name)
                )
        for manipulator in manipulators:
            sensors.append(
                SensorSpec(
                    f"{manipulator.name}_end_effector_pos",
                    "framepos",
                    f"{manipulator.name}_end_effector",
                    manipulator.name,
                    (3,),
                )
            )
        return tuple(sensors)

    @staticmethod
    def _uniform(rng: np.random.Generator, low: float, high: float) -> float:
        return float(rng.uniform(low, high))

    @staticmethod
    def _log_uniform(rng: np.random.Generator, low: float, high: float) -> float:
        return float(np.exp(rng.uniform(np.log(low), np.log(high))))
