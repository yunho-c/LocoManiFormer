from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any


class RobotFamily(StrEnum):
    BIPED = "biped"
    QUADRUPED = "quadruped"
    WHEELED_BIPED = "wheeled_biped"
    WHEELED_QUADRUPED = "wheeled_quadruped"


class ParameterRangePreset(StrEnum):
    COMMERCIAL_SURROGATE = "commercial_surrogate"
    CONSERVATIVE = "conservative"
    BROAD = "broad"
    EXTREME = "extreme"
    HELDOUT = "heldout"


class ValidationStrictness(StrEnum):
    RELAXED = "relaxed"
    STANDARD = "standard"
    STRICT = "strict"


@dataclass(frozen=True)
class RobotGenerationConfig:
    allowed_families: tuple[RobotFamily, ...] = (
        RobotFamily.BIPED,
        RobotFamily.QUADRUPED,
        RobotFamily.WHEELED_BIPED,
        RobotFamily.WHEELED_QUADRUPED,
    )
    global_scale_range: tuple[float, float] = (0.8, 1.25)
    parameter_range_preset: ParameterRangePreset = ParameterRangePreset.COMMERCIAL_SURROGATE
    validation_strictness: ValidationStrictness = ValidationStrictness.STANDARD
    require_mjx: bool = False
    dataset_split: str = "train"
    generator_version: str = "v1"
    seed_policy: str = "explicit"
    manipulator_probability: float = 0.0
    allowed_manipulator_counts: tuple[int, ...] = (0, 1, 2)
    max_rejection_attempts: int = 16
    total_mass_range: tuple[float, float] = (3.0, 180.0)
    require_contact_sensors: bool = True

    @classmethod
    def conservative(
        cls,
        *,
        allowed_families: tuple[RobotFamily, ...] | None = None,
        require_mjx: bool = False,
        manipulator_probability: float = 0.0,
    ) -> RobotGenerationConfig:
        return cls.from_preset(
            ParameterRangePreset.COMMERCIAL_SURROGATE,
            allowed_families=allowed_families,
            require_mjx=require_mjx,
            manipulator_probability=manipulator_probability,
        )

    @classmethod
    def from_preset(
        cls,
        preset: ParameterRangePreset | str,
        *,
        allowed_families: tuple[RobotFamily, ...] | None = None,
        require_mjx: bool = False,
        manipulator_probability: float = 0.0,
    ) -> RobotGenerationConfig:
        range_preset = ParameterRangePreset(preset)
        default = cls()
        return cls(
            allowed_families=allowed_families or default.allowed_families,
            parameter_range_preset=range_preset,
            require_mjx=require_mjx,
            manipulator_probability=manipulator_probability,
        )

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))

    def stable_hash(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _jsonable(value: Any) -> Any:
    if isinstance(value, StrEnum):
        return value.value
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    return value
