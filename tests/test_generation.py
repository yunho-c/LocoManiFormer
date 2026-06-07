from __future__ import annotations

from dataclasses import replace

from locomaniformer.generation import (
    MorphologyValidator,
    ParameterRangePreset,
    RobotFamily,
    RobotFamilySampler,
    RobotGenerationConfig,
    create_preview_collage,
    generate_robot_artifact,
    write_preview_collage,
)
from locomaniformer.generation.mjcf_robot import MJCFGeneratedRobot


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


def test_default_quadruped_uses_commercial_twelve_actuator_layout() -> None:
    artifact = generate_robot_artifact(
        RobotGenerationConfig.conservative(),
        seed=12,
        family=RobotFamily.QUADRUPED,
    )

    assert artifact.validation_result.accepted, artifact.validation_result.reasons
    assert (
        artifact.morphology_spec.config_hash == RobotGenerationConfig.conservative().stable_hash()
    )
    assert artifact.morphology_spec.actuation.actuator_type == "motor"
    assert artifact.summary_statistics["actuator_count"] == 12
    assert {len(limb.joints) for limb in artifact.morphology_spec.limbs} == {3}


def test_default_biped_uses_commercial_ten_actuator_layout() -> None:
    artifact = generate_robot_artifact(
        RobotGenerationConfig.conservative(),
        seed=13,
        family=RobotFamily.BIPED,
    )

    assert artifact.validation_result.accepted, artifact.validation_result.reasons
    assert artifact.summary_statistics["actuator_count"] == 10
    assert {len(limb.joints) for limb in artifact.morphology_spec.limbs} == {5}


def test_default_quadruped_left_right_pairs_are_mirrored() -> None:
    spec = RobotFamilySampler(RobotGenerationConfig.conservative()).sample(
        seed=16,
        family=RobotFamily.QUADRUPED,
    )

    for mount_label in ("rear", "front"):
        right = next(limb for limb in spec.limbs if limb.name == f"{mount_label}_right_leg")
        left = next(limb for limb in spec.limbs if limb.name == f"{mount_label}_left_leg")

        assert right.upper == left.upper
        assert right.lower == left.lower
        assert right.joints == left.joints
        assert right.foot == left.foot
        assert right.mount_pos[0] == left.mount_pos[0]
        assert right.mount_pos[1] == -left.mount_pos[1]
        assert right.mount_pos[2] == left.mount_pos[2]


def test_default_biped_left_right_pair_is_mirrored() -> None:
    spec = RobotFamilySampler(RobotGenerationConfig.conservative()).sample(
        seed=17,
        family=RobotFamily.BIPED,
    )
    right = next(limb for limb in spec.limbs if limb.name == "right_leg")
    left = next(limb for limb in spec.limbs if limb.name == "left_leg")

    assert right.upper == left.upper
    assert right.lower == left.lower
    assert right.joints == left.joints
    assert right.foot == left.foot
    assert right.mount_pos[0] == left.mount_pos[0]
    assert right.mount_pos[1] == -left.mount_pos[1]
    assert right.mount_pos[2] == left.mount_pos[2]


def test_default_leg_joints_are_hinges_not_ball_joints() -> None:
    spec = RobotFamilySampler(RobotGenerationConfig.conservative()).sample(
        seed=14,
        family=RobotFamily.QUADRUPED,
    )
    xml = MJCFGeneratedRobot(spec).get_mjcf_str()

    assert 'type="ball"' not in xml
    assert 'type="hinge"' in xml


def test_explicit_conservative_preset_remains_available() -> None:
    config = RobotGenerationConfig.from_preset(ParameterRangePreset.CONSERVATIVE)
    spec = RobotFamilySampler(config).sample(seed=15, family=RobotFamily.QUADRUPED)

    assert spec.config_hash == config.stable_hash()
    assert spec.source_ranges["limbs"] == ParameterRangePreset.CONSERVATIVE.value


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
