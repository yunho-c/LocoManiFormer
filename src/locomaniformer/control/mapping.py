from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from locomaniformer.control.cpg import CPGState
from locomaniformer.generation.artifacts import GeneratedRobotArtifact
from locomaniformer.generation.metadata import ActionDescriptor

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class ActionMapEntry:
    action_index: int
    actuator_name: str
    target: str
    limb_name: str | None
    joint_role: str
    family_role: str
    control_range: tuple[float, float]
    oscillator_index: int | None

    def to_dict(self) -> dict[str, float | int | str | None | list[float]]:
        return {
            "action_index": self.action_index,
            "actuator_name": self.actuator_name,
            "target": self.target,
            "limb_name": self.limb_name,
            "joint_role": self.joint_role,
            "family_role": self.family_role,
            "control_range": list(self.control_range),
            "oscillator_index": self.oscillator_index,
        }


class CPGActionMapper:
    def __init__(self, entries: tuple[ActionMapEntry, ...]) -> None:
        self.entries = entries
        self.action_size = len(entries)
        self.oscillator_count = max(
            (entry.oscillator_index for entry in entries if entry.oscillator_index is not None),
            default=-1,
        ) + 1

    @classmethod
    def from_artifact(cls, artifact: GeneratedRobotArtifact) -> CPGActionMapper:
        limb_names = tuple(limb.name for limb in artifact.morphology_spec.limbs)
        manipulator_names = tuple(
            manipulator.name for manipulator in artifact.morphology_spec.manipulators
        )
        entries: list[ActionMapEntry] = []
        oscillator_index = 0
        sorted_actions = sorted(
            artifact.action_descriptor,
            key=lambda item: item.normalized_action_index,
        )
        for action in sorted_actions:
            limb_name = _matching_prefix(action.target, limb_names)
            manipulator_name = _matching_prefix(action.target, manipulator_names)
            joint_role = _joint_role(action, limb_name, manipulator_name)
            is_manipulator = manipulator_name is not None and limb_name is None
            mapped_index = None if is_manipulator else oscillator_index
            if mapped_index is not None:
                oscillator_index += 1
            entries.append(
                ActionMapEntry(
                    action_index=action.normalized_action_index,
                    actuator_name=action.actuator_name,
                    target=action.target,
                    limb_name=limb_name,
                    joint_role=joint_role,
                    family_role=_family_role(limb_name),
                    control_range=tuple(action.control_range),
                    oscillator_index=mapped_index,
                )
            )
        return cls(tuple(entries))

    def action(self, cpg_state: CPGState) -> FloatArray:
        action = np.zeros(self.action_size, dtype=np.float64)
        for entry in self.entries:
            if entry.oscillator_index is None:
                value = 0.0
            else:
                value = float(cpg_state.outputs[entry.oscillator_index])
            low, high = entry.control_range
            action[entry.action_index] = np.clip(value, low, high)
        return action

    def to_dict(self) -> list[dict[str, float | int | str | None | list[float]]]:
        return [entry.to_dict() for entry in self.entries]

    def locomotion_mask(self) -> FloatArray:
        return np.array([entry.oscillator_index is not None for entry in self.entries], dtype=bool)


def _matching_prefix(target: str, prefixes: tuple[str, ...]) -> str | None:
    matches = [prefix for prefix in prefixes if target == prefix or target.startswith(f"{prefix}_")]
    if not matches:
        return None
    return max(matches, key=len)


def _joint_role(
    action: ActionDescriptor,
    limb_name: str | None,
    manipulator_name: str | None,
) -> str:
    if limb_name is not None:
        return action.target.removeprefix(f"{limb_name}_")
    if manipulator_name is not None:
        return action.target.removeprefix(f"{manipulator_name}_")
    return action.target


def _family_role(limb_name: str | None) -> str:
    if limb_name is None:
        return "manipulator"
    if "front" in limb_name:
        longitudinal = "front"
    elif "rear" in limb_name:
        longitudinal = "rear"
    else:
        longitudinal = "mid"
    if "left" in limb_name:
        lateral = "left"
    elif "right" in limb_name:
        lateral = "right"
    else:
        lateral = "center"
    return f"{longitudinal}_{lateral}"
