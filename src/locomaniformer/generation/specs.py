from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, is_dataclass
from enum import StrEnum
from typing import Any

from fprs.specification import MorphologySpecification

from locomaniformer.generation.config import RobotFamily

Vector3 = tuple[float, float, float]


@dataclass(frozen=True)
class RangeSample:
    value: float
    source_range: str


@dataclass(frozen=True)
class BodySpec:
    shape: str
    length: float
    width: float
    height: float
    mass: float
    density: float
    com_offset: Vector3
    shoulder_mount_offset: Vector3


@dataclass(frozen=True)
class SegmentSpec:
    length: float
    radius: float
    mass: float
    com_offset: Vector3 = (0.0, 0.0, 0.0)


@dataclass(frozen=True)
class JointSpec:
    name_suffix: str
    axis: Vector3
    range: tuple[float, float]
    damping: float
    armature: float
    stiffness: float = 0.0


@dataclass(frozen=True)
class FootSpec:
    shape: str
    size: Vector3
    local_offset: Vector3
    mass: float
    friction: tuple[float, float, float]
    contact_sensor_count: int = 1


@dataclass(frozen=True)
class WheelSpec:
    radius: float
    width: float
    mass: float
    local_axle_offset: Vector3
    axle_axis: Vector3
    drive_torque_range: tuple[float, float]
    steerable: bool = False
    passive: bool = False


@dataclass(frozen=True)
class LimbSpec:
    name: str
    side: str
    mount_label: str
    mount_pos: Vector3
    upper: SegmentSpec
    lower: SegmentSpec
    joints: tuple[JointSpec, ...]
    foot: FootSpec | None = None
    wheel: WheelSpec | None = None
    symmetry_group: str = "none"


@dataclass(frozen=True)
class ManipulatorSpec:
    name: str
    side: str
    mount_pos: Vector3
    links: tuple[SegmentSpec, ...]
    joints: tuple[JointSpec, ...]
    gripper_type: str
    end_effector_mass: float
    end_effector_radius: float


@dataclass(frozen=True)
class SensorSpec:
    name: str
    kind: str
    target: str
    body_part_path: str
    shape: tuple[int, ...] = (1,)


@dataclass(frozen=True)
class ActuationSpec:
    actuator_type: str
    strength_multiplier: float
    ctrl_range: tuple[float, float]
    force_range: tuple[float, float]


@dataclass(frozen=True)
class PhysicsSpec:
    friction: tuple[float, float, float]
    damping: float
    armature: float
    density: float
    restitution: float = 0.0


@dataclass(frozen=True)
class SymmetrySpec:
    mirrored: bool
    perturbation_scale: float
    groups: tuple[str, ...]


@dataclass(frozen=True)
class RobotMorphologySpec:
    robot_id: str
    seed: int
    config_hash: str
    family: RobotFamily
    global_scale: float
    body: BodySpec
    limbs: tuple[LimbSpec, ...]
    wheels: tuple[WheelSpec, ...]
    manipulators: tuple[ManipulatorSpec, ...]
    sensors: tuple[SensorSpec, ...]
    actuation: ActuationSpec
    physics: PhysicsSpec
    symmetry: SymmetrySpec
    source_ranges: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))

    def stable_hash(self) -> str:
        payload = self.to_dict()
        payload.pop("robot_id", None)
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]


class GeneratedRobotSpecification(MorphologySpecification):
    """FPRS-compatible wrapper around the generator dataclass spec."""

    def __init__(self, morphology_spec: RobotMorphologySpec) -> None:
        super().__init__()
        self.morphology_spec = morphology_spec


def robot_id_for(
    *,
    generator_version: str,
    config_hash: str,
    seed: int,
    family: RobotFamily,
    payload: dict[str, Any],
) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:12]
    return f"{generator_version}-{family.value}-{seed}-{config_hash[:8]}-{digest}"


def _jsonable(value: Any) -> Any:
    if isinstance(value, StrEnum):
        return value.value
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    return value
