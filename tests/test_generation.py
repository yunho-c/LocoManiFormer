from __future__ import annotations

from dataclasses import replace

from locomaniformer.generation import (
    MorphologyValidator,
    RobotFamily,
    RobotFamilySampler,
    RobotGenerationConfig,
    create_preview_collage,
    generate_robot_artifact,
    write_preview_collage,
)


def test_sampler_reproducibility_and_stable_ids() -> None:
    config = RobotGenerationConfig.conservative()
    sampler = RobotFamilySampler(config)

    first = sampler.sample(seed=42, family=RobotFamily.QUADRUPED)
    second = sampler.sample(seed=42, family=RobotFamily.QUADRUPED)

    assert first.robot_id == second.robot_id
    assert first.to_dict() == second.to_dict()


def test_sampler_random_family_keeps_enum_type() -> None:
    spec = RobotFamilySampler(RobotGenerationConfig.conservative()).sample(seed=0)

    assert isinstance(spec.family, RobotFamily)


def test_every_family_generates_valid_native_mujoco_artifact() -> None:
    config = RobotGenerationConfig.conservative()

    for index, family in enumerate(RobotFamily):
        artifact = generate_robot_artifact(config, seed=100 + index, family=family)

        assert artifact.validation_result.accepted, artifact.validation_result.reasons
        assert len(artifact.action_descriptor) == artifact.summary_statistics["actuator_count"]
        assert len(artifact.observation_descriptor) == len(artifact.morphology_spec.sensors)
        assert artifact.summary_statistics["limb_count"] in {2, 4}


def test_wheel_robot_exposes_wheel_velocity_observations() -> None:
    artifact = generate_robot_artifact(
        RobotGenerationConfig.conservative(),
        seed=7,
        family=RobotFamily.WHEELED_QUADRUPED,
    )

    observation_names = {
        descriptor.observation_name for descriptor in artifact.observation_descriptor
    }

    assert artifact.validation_result.accepted, artifact.validation_result.reasons
    assert any(name.endswith("_wheel_vel") for name in observation_names)


def test_manipulator_robot_exposes_end_effector_metadata() -> None:
    config = RobotGenerationConfig.conservative(
        allowed_families=(RobotFamily.BIPED,),
        manipulator_probability=1.0,
    )
    artifact = generate_robot_artifact(config, seed=9, family=RobotFamily.BIPED)
    node_ids = {node["id"] for node in artifact.embodiment_metadata["nodes"]}

    assert artifact.validation_result.accepted, artifact.validation_result.reasons
    assert any(node_id.endswith("_end_effector") for node_id in node_ids)


def test_validator_rejects_malformed_joint_axis_and_range() -> None:
    config = RobotGenerationConfig.conservative()
    spec = RobotFamilySampler(config).sample(seed=5, family=RobotFamily.BIPED)
    first_limb = spec.limbs[0]
    bad_joint = replace(first_limb.joints[0], axis=(0.0, 0.0, 0.0), range=(1.0, 1.0))
    bad_limb = replace(first_limb, joints=(bad_joint, *first_limb.joints[1:]))
    bad_spec = replace(spec, limbs=(bad_limb, *spec.limbs[1:]))

    result = MorphologyValidator(config).validate(bad_spec)

    assert not result.accepted
    assert any("axis is zero" in reason for reason in result.reasons)
    assert any("range is not ordered" in reason for reason in result.reasons)


def test_artifact_write_outputs_json_and_xml(tmp_path) -> None:
    artifact = generate_robot_artifact(
        RobotGenerationConfig.conservative(),
        seed=11,
        family=RobotFamily.BIPED,
    )

    paths = artifact.write(tmp_path)

    assert paths["artifact"].exists()
    assert paths["metadata"].exists()
    assert paths["validation"].exists()
    assert paths["mjcf"].read_text(encoding="utf-8").startswith("<mujoco")


def test_preview_collage_renders_regular_grid(tmp_path) -> None:
    collage = create_preview_collage(
        RobotGenerationConfig.conservative(allowed_families=(RobotFamily.BIPED,)),
        count=2,
        start_seed=0,
        family=RobotFamily.BIPED,
        columns=2,
        cell_size=96,
        padding=4,
    )
    output = write_preview_collage(collage, tmp_path / "preview.png")

    assert output.exists()
    assert collage.image.shape == (104, 204, 3)
    assert collage.image.max() > collage.image.min()
    assert collage.accepted_count == 2
