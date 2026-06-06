from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from locomaniformer.generation.mjcf_robot import MJCFGeneratedRobot
from locomaniformer.generation.specs import RobotMorphologySpec


@dataclass(frozen=True)
class ActionDescriptor:
    actuator_name: str
    actuator_type: str
    target: str
    body_part_path: str
    control_range: tuple[float, float]
    force_range: tuple[float, float]
    normalized_action_index: int
    symmetry_group: str


@dataclass(frozen=True)
class ObservationDescriptor:
    observation_name: str
    source: str
    shape: tuple[int, ...]
    bounds: tuple[float, float]
    body_part_path: str
    frame: str
    normalization_policy: str


def action_descriptors(
    spec: RobotMorphologySpec,
    robot: MJCFGeneratedRobot,
) -> tuple[ActionDescriptor, ...]:
    limb_groups = {limb.name: limb.symmetry_group for limb in spec.limbs}
    descriptors: list[ActionDescriptor] = []
    for index, actuator_name in enumerate(robot.actuator_names):
        target = actuator_name.removesuffix("_motor")
        body_part = target.rsplit("_", 1)[0]
        descriptors.append(
            ActionDescriptor(
                actuator_name=actuator_name,
                actuator_type=spec.actuation.actuator_type,
                target=target,
                body_part_path=body_part,
                control_range=spec.actuation.ctrl_range,
                force_range=spec.actuation.force_range,
                normalized_action_index=index,
                symmetry_group=limb_groups.get(body_part, body_part),
            )
        )
    return tuple(descriptors)


def observation_descriptors(spec: RobotMorphologySpec) -> tuple[ObservationDescriptor, ...]:
    return tuple(
        ObservationDescriptor(
            observation_name=sensor.name,
            source=f"{sensor.kind}:{sensor.target}",
            shape=sensor.shape,
            bounds=(-float("inf"), float("inf")),
            body_part_path=sensor.body_part_path,
            frame="local" if sensor.kind in {"jointpos", "jointvel", "touch"} else "root",
            normalization_policy="standardize",
        )
        for sensor in spec.sensors
    )


def embodiment_metadata(
    spec: RobotMorphologySpec,
    actions: tuple[ActionDescriptor, ...],
    observations: tuple[ObservationDescriptor, ...],
) -> dict[str, Any]:
    nodes: list[dict[str, Any]] = [
        {
            "id": "torso",
            "type": "body",
            "features": {
                "length": spec.body.length,
                "width": spec.body.width,
                "height": spec.body.height,
                "mass": spec.body.mass,
            },
        }
    ]
    edges: list[dict[str, str]] = []

    for limb in spec.limbs:
        nodes.append(
            {
                "id": limb.name,
                "type": "limb",
                "symmetry_group": limb.symmetry_group,
                "features": {
                    "upper_length": limb.upper.length,
                    "lower_length": limb.lower.length,
                    "upper_mass": limb.upper.mass,
                    "lower_mass": limb.lower.mass,
                    "is_wheeled": limb.wheel is not None,
                },
            }
        )
        edges.append({"source": "torso", "target": limb.name, "type": "kinematic"})
        for joint in limb.joints:
            joint_id = f"{limb.name}_{joint.name_suffix}"
            nodes.append(
                {
                    "id": joint_id,
                    "type": "joint",
                    "features": {
                        "axis": joint.axis,
                        "range": joint.range,
                        "damping": joint.damping,
                        "armature": joint.armature,
                    },
                }
            )
            edges.append({"source": limb.name, "target": joint_id, "type": "joint"})
        terminal_type = "wheel" if limb.wheel is not None else "foot"
        terminal_id = f"{limb.name}_{terminal_type}"
        nodes.append({"id": terminal_id, "type": terminal_type, "features": {}})
        edges.append({"source": limb.name, "target": terminal_id, "type": "terminal"})

    for manipulator in spec.manipulators:
        nodes.append(
            {
                "id": manipulator.name,
                "type": "manipulator",
                "features": {
                    "link_count": len(manipulator.links),
                    "gripper_type": manipulator.gripper_type,
                },
            }
        )
        edges.append({"source": "torso", "target": manipulator.name, "type": "kinematic"})
        nodes.append(
            {
                "id": f"{manipulator.name}_end_effector",
                "type": "end_effector",
                "features": {"radius": manipulator.end_effector_radius},
            }
        )
        edges.append(
            {
                "source": manipulator.name,
                "target": f"{manipulator.name}_end_effector",
                "type": "terminal",
            }
        )

    return {
        "schema_version": "embodiment_metadata.v1",
        "robot_id": spec.robot_id,
        "family": spec.family.value,
        "nodes": nodes,
        "edges": edges,
        "summary_features": {
            "global_scale": spec.global_scale,
            "limb_count": len(spec.limbs),
            "actuator_count": len(actions),
            "observation_count": len(observations),
            "wheel_count": len(spec.wheels),
            "manipulator_count": len(spec.manipulators),
        },
        "actions": [asdict(action) for action in actions],
        "observations": [asdict(observation) for observation in observations],
    }
