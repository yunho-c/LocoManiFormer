from __future__ import annotations

import math

from dm_control import mjcf
from moojoco.mjcf.morphology import MJCFMorphology

from locomaniformer.generation.specs import (
    GeneratedRobotSpecification,
    JointSpec,
    LimbSpec,
    ManipulatorSpec,
    RobotMorphologySpec,
)


class MJCFGeneratedRobot(MJCFMorphology):
    def __init__(self, spec: RobotMorphologySpec, name: str | None = None) -> None:
        self.spec = spec
        self._joint_names: list[str] = []
        self._actuator_names: list[str] = []
        self._sensor_names: list[str] = []
        super().__init__(
            specification=GeneratedRobotSpecification(spec),
            name=name or spec.robot_id,
        )

    @property
    def joint_names(self) -> tuple[str, ...]:
        return tuple(self._joint_names)

    @property
    def actuator_names(self) -> tuple[str, ...]:
        return tuple(self._actuator_names)

    @property
    def sensor_names(self) -> tuple[str, ...]:
        return tuple(self._sensor_names)

    def _build(self, *args, **kwargs) -> None:
        del args, kwargs
        self.mjcf_model.compiler.angle = "radian"
        self.mjcf_model.option.gravity = [0.0, 0.0, -9.81]
        self.mjcf_model.option.timestep = 0.002

        self.mjcf_model.default.geom.friction = self.spec.physics.friction
        self.mjcf_model.default.geom.condim = 3
        self.mjcf_model.default.joint.damping = self.spec.physics.damping
        self.mjcf_model.default.joint.armature = self.spec.physics.armature

        self._add_materials()
        torso = self._build_torso()
        for limb in self.spec.limbs:
            self._build_limb(torso, limb)
        for manipulator in self.spec.manipulators:
            self._build_manipulator(torso, manipulator)

    def _add_materials(self) -> None:
        self.mjcf_model.asset.add("material", name="torso_mat", rgba=[0.22, 0.32, 0.40, 1.0])
        self.mjcf_model.asset.add("material", name="limb_mat", rgba=[0.70, 0.58, 0.38, 1.0])
        self.mjcf_model.asset.add("material", name="distal_mat", rgba=[0.18, 0.18, 0.18, 1.0])
        self.mjcf_model.asset.add("material", name="wheel_mat", rgba=[0.08, 0.09, 0.10, 1.0])
        self.mjcf_model.asset.add("material", name="arm_mat", rgba=[0.40, 0.45, 0.62, 1.0])

    def _build_torso(self) -> mjcf.Element:
        body = self.spec.body
        torso = self.mjcf_body.add(
            "body",
            name="torso",
            pos=[0.0, 0.0, self.nominal_body_height],
        )
        torso.add(
            "geom",
            name="torso_geom",
            type="box",
            size=[body.length / 2.0, body.width / 2.0, body.height / 2.0],
            mass=body.mass,
            pos=body.com_offset,
            material="torso_mat",
        )
        torso.add("site", name="torso_imu", pos=[0.0, 0.0, body.height / 2.0])
        self._add_sensor("framequat", "torso_orientation", objtype="site", objname="torso_imu")
        self._add_sensor("gyro", "torso_angular_velocity", site="torso_imu")
        return torso

    @property
    def nominal_body_height(self) -> float:
        leg_lengths = [limb.upper.length + limb.lower.length for limb in self.spec.limbs]
        distal = max(
            [
                (limb.wheel.radius if limb.wheel else (limb.foot.size[2] if limb.foot else 0.02))
                for limb in self.spec.limbs
            ],
            default=0.02,
        )
        return max(
            0.2, (sum(leg_lengths) / max(1, len(leg_lengths))) + distal + self.spec.body.height
        )

    def _build_limb(self, torso: mjcf.Element, limb: LimbSpec) -> None:
        hip = torso.add("body", name=f"{limb.name}_hip_body", pos=limb.mount_pos)
        hip_joint_specs = [joint for joint in limb.joints if joint.name_suffix.startswith("hip")]
        distal_joint_specs = [
            joint for joint in limb.joints if not joint.name_suffix.startswith("hip")
        ]
        for joint in hip_joint_specs:
            self._add_joint_and_actuator(hip, limb.name, joint)
        hip.add(
            "geom",
            name=f"{limb.name}_upper_geom",
            type="capsule",
            fromto=[0.0, 0.0, 0.0, 0.0, 0.0, -limb.upper.length],
            size=[limb.upper.radius],
            mass=limb.upper.mass,
            material="limb_mat",
        )
        lower = hip.add("body", name=f"{limb.name}_lower_body", pos=[0.0, 0.0, -limb.upper.length])
        for joint in distal_joint_specs:
            self._add_joint_and_actuator(lower, limb.name, joint)
        lower.add(
            "geom",
            name=f"{limb.name}_lower_geom",
            type="capsule",
            fromto=[0.0, 0.0, 0.0, 0.0, 0.0, -limb.lower.length],
            size=[limb.lower.radius],
            mass=limb.lower.mass,
            material="limb_mat",
        )
        terminal = lower.add(
            "body",
            name=f"{limb.name}_terminal_body",
            pos=[0.0, 0.0, -limb.lower.length],
        )
        if limb.wheel is not None:
            self._build_wheel(terminal, limb)
        elif limb.foot is not None:
            self._build_foot(terminal, limb)

    def _build_foot(self, terminal: mjcf.Element, limb: LimbSpec) -> None:
        assert limb.foot is not None
        foot = limb.foot
        terminal.add(
            "geom",
            name=f"{limb.name}_foot_geom",
            type=foot.shape,
            size=list(foot.size),
            pos=list(foot.local_offset),
            mass=foot.mass,
            friction=list(foot.friction),
            material="distal_mat",
        )
        terminal.add("site", name=f"{limb.name}_contact", pos=list(foot.local_offset), size=[0.012])
        self._add_sensor("touch", f"{limb.name}_touch", site=f"{limb.name}_contact")

    def _build_wheel(self, terminal: mjcf.Element, limb: LimbSpec) -> None:
        assert limb.wheel is not None
        wheel = limb.wheel
        parent = terminal
        if wheel.steerable:
            steer_joint_name = f"{limb.name}_wheel_steer"
            steer_body = terminal.add(
                "body", name=f"{limb.name}_steer_body", pos=list(wheel.local_axle_offset)
            )
            steer_joint = steer_body.add(
                "joint",
                name=steer_joint_name,
                type="hinge",
                axis=[0.0, 0.0, 1.0],
                range=[-0.75, 0.75],
                limited=True,
                damping=self.spec.physics.damping,
                armature=self.spec.physics.armature,
            )
            self._joint_names.append(steer_joint_name)
            self._add_motor(
                steer_joint, steer_joint_name, gear=20.0 * self.spec.actuation.strength_multiplier
            )
            steer_body.add(
                "geom",
                name=f"{limb.name}_steer_hub_geom",
                type="sphere",
                size=[max(0.012, wheel.width * 0.3)],
                mass=max(0.02, wheel.mass * 0.1),
                material="distal_mat",
            )
            parent = steer_body
            wheel_pos = [0.0, 0.0, 0.0]
        else:
            wheel_pos = list(wheel.local_axle_offset)

        drive_name = f"{limb.name}_wheel_drive"
        drive_body = parent.add("body", name=f"{limb.name}_wheel_body", pos=wheel_pos)
        drive_joint = drive_body.add(
            "joint",
            name=drive_name,
            type="hinge",
            axis=list(wheel.axle_axis),
            limited=False,
            damping=self.spec.physics.damping * 0.25,
            armature=self.spec.physics.armature,
        )
        self._joint_names.append(drive_name)
        if not wheel.passive:
            self._add_motor(drive_joint, drive_name, gear=abs(wheel.drive_torque_range[1]))
        drive_body.add(
            "geom",
            name=f"{limb.name}_wheel_geom",
            type="cylinder",
            size=[wheel.radius, wheel.width / 2.0],
            euler=[math.pi / 2.0, 0.0, 0.0],
            mass=wheel.mass,
            material="wheel_mat",
            friction=list(self.spec.physics.friction),
        )
        drive_body.add(
            "site", name=f"{limb.name}_contact", pos=[0.0, 0.0, -wheel.radius], size=[0.012]
        )
        self._add_sensor("jointvel", f"{limb.name}_wheel_vel", joint=drive_name)
        self._add_sensor("touch", f"{limb.name}_touch", site=f"{limb.name}_contact")

    def _build_manipulator(self, torso: mjcf.Element, manipulator: ManipulatorSpec) -> None:
        current = torso.add(
            "body", name=f"{manipulator.name}_base_body", pos=list(manipulator.mount_pos)
        )
        for index, link in enumerate(manipulator.links):
            joint = manipulator.joints[min(index, len(manipulator.joints) - 1)]
            self._add_joint_and_actuator(current, manipulator.name, joint)
            current.add(
                "geom",
                name=f"{manipulator.name}_link_{index}_geom",
                type="capsule",
                fromto=[0.0, 0.0, 0.0, link.length, 0.0, 0.0],
                size=[link.radius],
                mass=link.mass,
                material="arm_mat",
            )
            current = current.add(
                "body",
                name=f"{manipulator.name}_link_{index}_body",
                pos=[link.length, 0.0, 0.0],
            )
        current.add(
            "geom",
            name=f"{manipulator.name}_end_effector_geom",
            type="sphere",
            size=[manipulator.end_effector_radius],
            mass=manipulator.end_effector_mass,
            material="distal_mat",
        )
        current.add("site", name=f"{manipulator.name}_end_effector", size=[0.018])
        self._add_sensor(
            "framepos",
            f"{manipulator.name}_end_effector_pos",
            objtype="site",
            objname=f"{manipulator.name}_end_effector",
        )

    def _add_joint_and_actuator(
        self,
        body: mjcf.Element,
        prefix: str,
        joint: JointSpec,
    ) -> mjcf.Element:
        joint_name = f"{prefix}_{joint.name_suffix}"
        element = body.add(
            "joint",
            name=joint_name,
            type="hinge",
            axis=list(joint.axis),
            range=list(joint.range),
            limited=True,
            damping=joint.damping,
            armature=joint.armature,
            stiffness=joint.stiffness,
        )
        self._joint_names.append(joint_name)
        self._add_motor(element, joint_name, gear=45.0 * self.spec.actuation.strength_multiplier)
        self._add_sensor("jointpos", f"{joint_name}_pos", joint=joint_name)
        self._add_sensor("jointvel", f"{joint_name}_vel", joint=joint_name)
        return element

    def _add_motor(self, joint: mjcf.Element, joint_name: str, gear: float) -> None:
        actuator_name = f"{joint_name}_motor"
        self.mjcf_model.actuator.add(
            "motor",
            name=actuator_name,
            joint=joint,
            gear=[gear],
            ctrllimited=True,
            ctrlrange=list(self.spec.actuation.ctrl_range),
        )
        self._actuator_names.append(actuator_name)

    def _add_sensor(self, sensor_type: str, name: str, **kwargs: object) -> None:
        self.mjcf_model.sensor.add(sensor_type, name=name, **kwargs)
        self._sensor_names.append(name)
