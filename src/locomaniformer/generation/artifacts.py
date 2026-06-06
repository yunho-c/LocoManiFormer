from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from locomaniformer.generation.config import RobotFamily, RobotGenerationConfig
from locomaniformer.generation.metadata import (
    ActionDescriptor,
    ObservationDescriptor,
    action_descriptors,
    embodiment_metadata,
    observation_descriptors,
)
from locomaniformer.generation.mjcf_robot import MJCFGeneratedRobot
from locomaniformer.generation.sampler import RobotFamilySampler
from locomaniformer.generation.specs import RobotMorphologySpec
from locomaniformer.generation.validation import (
    MorphologyValidator,
    ValidationResult,
    summary_statistics,
)


@dataclass(frozen=True)
class GeneratedRobotArtifact:
    robot_id: str
    seed: int
    generator_version: str
    family: str
    morphology_spec: RobotMorphologySpec
    mjcf_xml: str
    mjcf_assets: dict[str, Any]
    action_descriptor: tuple[ActionDescriptor, ...]
    observation_descriptor: tuple[ObservationDescriptor, ...]
    embodiment_metadata: dict[str, Any]
    validation_result: ValidationResult
    summary_statistics: dict[str, float | int | str | bool]

    def to_dict(self) -> dict[str, Any]:
        return {
            "robot_id": self.robot_id,
            "seed": self.seed,
            "generator_version": self.generator_version,
            "family": self.family,
            "morphology_spec": self.morphology_spec.to_dict(),
            "mjcf_xml": self.mjcf_xml,
            "mjcf_assets": sorted(self.mjcf_assets),
            "action_descriptor": [asdict(item) for item in self.action_descriptor],
            "observation_descriptor": [asdict(item) for item in self.observation_descriptor],
            "embodiment_metadata": self.embodiment_metadata,
            "validation_result": self.validation_result.to_dict(),
            "summary_statistics": self.summary_statistics,
        }

    def write(self, output_dir: Path | str) -> dict[str, Path]:
        root = Path(output_dir)
        robot_dir = root / self.robot_id
        robot_dir.mkdir(parents=True, exist_ok=True)
        paths = {
            "artifact": robot_dir / "artifact.json",
            "mjcf": robot_dir / "robot.xml",
            "metadata": robot_dir / "metadata.json",
            "validation": robot_dir / "validation.json",
        }
        paths["artifact"].write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8"
        )
        paths["mjcf"].write_text(self.mjcf_xml, encoding="utf-8")
        paths["metadata"].write_text(
            json.dumps(self.embodiment_metadata, indent=2, sort_keys=True), encoding="utf-8"
        )
        paths["validation"].write_text(
            json.dumps(self.validation_result.to_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return paths


def generate_robot_artifact(
    config: RobotGenerationConfig,
    seed: int,
    *,
    family: RobotFamily | str | None = None,
) -> GeneratedRobotArtifact:
    sampler = RobotFamilySampler(config)
    spec = sampler.sample(seed=seed, family=family)
    robot = MJCFGeneratedRobot(spec)
    validator = MorphologyValidator(config)
    validation = validator.validate(spec, robot)
    actions = action_descriptors(spec, robot)
    observations = observation_descriptors(spec)
    metadata = embodiment_metadata(spec, actions, observations)
    return GeneratedRobotArtifact(
        robot_id=spec.robot_id,
        seed=seed,
        generator_version=config.generator_version,
        family=spec.family.value,
        morphology_spec=spec,
        mjcf_xml=robot.get_mjcf_str(),
        mjcf_assets=robot.get_mjcf_assets(),
        action_descriptor=actions,
        observation_descriptor=observations,
        embodiment_metadata=metadata,
        validation_result=validation,
        summary_statistics=summary_statistics(spec),
    )


def write_manifest(artifacts: list[GeneratedRobotArtifact], path: Path | str) -> None:
    manifest_path = Path(path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for artifact in artifacts:
        lines.append(
            json.dumps(
                {
                    "robot_id": artifact.robot_id,
                    "seed": artifact.seed,
                    "family": artifact.family,
                    "accepted": artifact.validation_result.accepted,
                    "summary_statistics": artifact.summary_statistics,
                },
                sort_keys=True,
            )
        )
    manifest_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
