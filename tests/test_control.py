from __future__ import annotations

import shutil

import numpy as np
import pytest

from locomaniformer.control import (
    CPG,
    BootstrapControllerConfig,
    CPGActionMapper,
    CPGParameters,
    create_heuristic_controller,
    load_manifest_generated_robot_artifacts,
    optimize_bootstrap_controller,
    optimize_bootstrap_controllers,
    render_bootstrap_preview,
)
from locomaniformer.generation import (
    RobotFamily,
    RobotGenerationConfig,
    generate_robot_artifact,
)
from locomaniformer.generation.artifacts import write_manifest


def test_cpg_step_is_deterministic_for_fixed_parameters() -> None:
    parameters = CPGParameters(
        amplitudes=np.array([0.5, 0.25]),
        offsets=np.array([0.0, 0.1]),
        frequencies=np.array([1.0, 2.0]),
        phase_biases=np.zeros((2, 2)),
        coupling_weights=np.zeros((2, 2)),
    )
    cpg = CPG(parameters, dt=0.1)

    first = cpg.step(cpg.reset())
    second = cpg.step(cpg.reset())

    np.testing.assert_allclose(first.phases, second.phases)
    np.testing.assert_allclose(first.outputs, second.outputs)


def test_cpg_parameters_round_trip_from_dict() -> None:
    parameters = CPGParameters(
        amplitudes=np.array([0.5, 0.25]),
        offsets=np.array([0.0, 0.1]),
        frequencies=np.array([1.0, 2.0]),
        phase_biases=np.zeros((2, 2)),
        coupling_weights=np.ones((2, 2)),
    )

    restored = CPGParameters.from_dict(parameters.to_dict())

    np.testing.assert_allclose(restored.amplitudes, parameters.amplitudes)
    np.testing.assert_allclose(restored.offsets, parameters.offsets)
    np.testing.assert_allclose(restored.frequencies, parameters.frequencies)
    np.testing.assert_allclose(restored.phase_biases, parameters.phase_biases)
    np.testing.assert_allclose(restored.coupling_weights, parameters.coupling_weights)


def test_action_mapper_clips_outputs_and_orders_by_action_descriptor() -> None:
    artifact = generate_robot_artifact(
        RobotGenerationConfig.conservative(),
        seed=21,
        family=RobotFamily.BIPED,
    )
    mapper = CPGActionMapper.from_artifact(artifact)
    outputs = np.full(mapper.oscillator_count, 4.0)
    state = cpg_state(outputs)

    action = mapper.action(state)

    assert action.shape == (len(artifact.action_descriptor),)
    assert np.all(action <= 1.0)
    assert np.all(action >= -1.0)


def test_manipulator_actuators_map_to_zero_by_default() -> None:
    config = RobotGenerationConfig.conservative(
        allowed_families=(RobotFamily.BIPED,),
        manipulator_probability=1.0,
    )
    artifact = generate_robot_artifact(config, seed=22, family=RobotFamily.BIPED)
    mapper = CPGActionMapper.from_artifact(artifact)
    parameters = create_heuristic_controller(artifact)
    state = CPG(parameters, dt=0.02).step(CPG(parameters, dt=0.02).reset())
    action = mapper.action(state)

    manipulator_indices = [
        entry.action_index for entry in mapper.entries if entry.family_role == "manipulator"
    ]

    assert manipulator_indices
    assert np.all(action[manipulator_indices] == 0.0)


def test_bootstrap_controller_serializes_and_links_robot_id(tmp_path) -> None:
    artifact = generate_robot_artifact(
        RobotGenerationConfig.conservative(),
        seed=23,
        family=RobotFamily.WHEELED_BIPED,
    )
    controller = optimize_bootstrap_controller(
        artifact,
        BootstrapControllerConfig(seed=1, candidates=2, horizon=0.04),
    )

    path = controller.write(tmp_path)

    assert path.exists()
    assert controller.robot_id == artifact.robot_id
    assert controller.evaluation_summary["steps"] >= 1
    assert "amplitudes" in controller.parameters
    assert controller.action_mapping


def test_manifest_loader_resolves_robot_artifacts(tmp_path) -> None:
    artifacts = [
        generate_robot_artifact(
            RobotGenerationConfig.conservative(),
            seed=24,
            family=RobotFamily.BIPED,
        ),
        generate_robot_artifact(
            RobotGenerationConfig.conservative(),
            seed=25,
            family=RobotFamily.QUADRUPED,
        ),
    ]
    robot_root = tmp_path / "robots"
    for artifact in artifacts:
        artifact.write(robot_root)
    manifest = tmp_path / "manifest.jsonl"
    write_manifest(artifacts, manifest)

    loaded = load_manifest_generated_robot_artifacts(manifest, robot_root)

    assert [artifact.robot_id for artifact in loaded] == [
        artifact.robot_id for artifact in artifacts
    ]


def test_batch_bootstrap_controllers_use_manifest_ordered_seed_offsets(tmp_path) -> None:
    artifacts = [
        generate_robot_artifact(
            RobotGenerationConfig.conservative(),
            seed=26,
            family=RobotFamily.WHEELED_BIPED,
        ),
        generate_robot_artifact(
            RobotGenerationConfig.conservative(),
            seed=27,
            family=RobotFamily.WHEELED_QUADRUPED,
        ),
    ]
    controllers = optimize_bootstrap_controllers(
        artifacts,
        BootstrapControllerConfig(seed=10, candidates=1, horizon=0.02),
    )
    paths = [controller.write(tmp_path) for controller in controllers]

    assert [controller.robot_id for controller in controllers] == [
        artifact.robot_id for artifact in artifacts
    ]
    assert [controller.seed for controller in controllers] == [10, 11]
    assert all(path.exists() for path in paths)


def test_bootstrap_preview_writes_mp4_when_ffmpeg_available(tmp_path) -> None:
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg is not installed")
    artifact = generate_robot_artifact(
        RobotGenerationConfig.conservative(),
        seed=28,
        family=RobotFamily.WHEELED_BIPED,
    )
    controller = optimize_bootstrap_controller(
        artifact,
        BootstrapControllerConfig(seed=2, candidates=1, horizon=0.02),
    )

    path = render_bootstrap_preview(
        artifact,
        controller,
        tmp_path / "preview.mp4",
        duration=0.2,
        fps=5,
        width=160,
        height=120,
    )

    assert path.exists()
    assert path.stat().st_size > 0


def cpg_state(outputs: np.ndarray):
    from locomaniformer.control.cpg import CPGState

    return CPGState(time=0.0, phases=np.zeros_like(outputs), outputs=outputs)
