from __future__ import annotations

import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from locomaniformer.control.artifacts import (
    BootstrapControllerArtifact,
    BootstrapControllerConfig,
)
from locomaniformer.control.cpg import CPG, CPGParameters
from locomaniformer.control.mapping import CPGActionMapper
from locomaniformer.generation.artifacts import GeneratedRobotArtifact


def create_heuristic_controller(
    artifact: GeneratedRobotArtifact,
    config: BootstrapControllerConfig | None = None,
) -> CPGParameters:
    config = config or BootstrapControllerConfig()
    del config
    mapper = CPGActionMapper.from_artifact(artifact)
    count = mapper.oscillator_count
    amplitudes = np.zeros(count, dtype=np.float64)
    offsets = np.zeros(count, dtype=np.float64)
    frequencies = np.full(count, 2.0 * np.pi * 1.1, dtype=np.float64)
    phase_biases = np.zeros((count, count), dtype=np.float64)
    coupling_weights = np.zeros((count, count), dtype=np.float64)

    base_phases = {entry.limb_name: _limb_phase(entry.family_role) for entry in mapper.entries}
    oscillator_phases = np.zeros(count, dtype=np.float64)

    for entry in mapper.entries:
        if entry.oscillator_index is None:
            continue
        phase = base_phases.get(entry.limb_name, 0.0)
        role = entry.joint_role
        if role == "hip_yaw":
            amplitudes[entry.oscillator_index] = 0.08
        elif role == "hip_roll":
            amplitudes[entry.oscillator_index] = 0.16
            phase += np.pi / 2.0
        elif role == "hip_pitch":
            amplitudes[entry.oscillator_index] = 0.32
        elif role == "knee_pitch":
            amplitudes[entry.oscillator_index] = 0.28
            offsets[entry.oscillator_index] = -0.08
            phase += np.pi
        elif role == "ankle_pitch":
            amplitudes[entry.oscillator_index] = 0.14
            phase += np.pi
        elif role == "wheel_drive":
            amplitudes[entry.oscillator_index] = 0.03
            offsets[entry.oscillator_index] = 0.35
        elif role == "wheel_steer":
            amplitudes[entry.oscillator_index] = 0.05
        else:
            amplitudes[entry.oscillator_index] = 0.05
        oscillator_phases[entry.oscillator_index] = phase

    for i in range(count):
        for j in range(count):
            if i == j:
                continue
            phase_biases[i, j] = oscillator_phases[j] - oscillator_phases[i]
            coupling_weights[i, j] = 0.35

    return CPGParameters(
        amplitudes=amplitudes,
        offsets=offsets,
        frequencies=frequencies,
        phase_biases=phase_biases,
        coupling_weights=coupling_weights,
    )


def optimize_bootstrap_controller(
    artifact: GeneratedRobotArtifact,
    config: BootstrapControllerConfig,
) -> BootstrapControllerArtifact:
    mapper = CPGActionMapper.from_artifact(artifact)
    rng = np.random.default_rng(config.seed)
    heuristic = create_heuristic_controller(artifact, config)

    best_parameters = heuristic
    best_summary = _evaluate(artifact, mapper, heuristic, config)
    best_score = float(best_summary["score"])

    for _ in range(max(0, config.candidates - 1)):
        candidate = _mutate_parameters(heuristic, rng)
        summary = _evaluate(artifact, mapper, candidate, config)
        score = float(summary["score"])
        if score > best_score:
            best_parameters = candidate
            best_summary = summary
            best_score = score

    return BootstrapControllerArtifact(
        robot_id=artifact.robot_id,
        controller_version=config.controller_version,
        seed=config.seed,
        objective=config.objective,
        parameters=best_parameters.to_dict(),
        action_mapping=mapper.to_dict(),
        score=best_score,
        evaluation_summary=best_summary,
    )


def _mutate_parameters(parameters: CPGParameters, rng: np.random.Generator) -> CPGParameters:
    mutated = parameters.copy()
    amplitudes = np.clip(
        mutated.amplitudes + rng.normal(0.0, 0.08, size=mutated.amplitudes.shape),
        0.0,
        0.8,
    )
    offsets = np.clip(
        mutated.offsets + rng.normal(0.0, 0.08, size=mutated.offsets.shape),
        -0.75,
        0.75,
    )
    frequencies = np.clip(
        mutated.frequencies * rng.uniform(0.65, 1.35, size=mutated.frequencies.shape),
        2.0,
        12.0,
    )
    phase_noise = rng.normal(0.0, 0.35, size=mutated.phase_biases.shape)
    phase_biases = mutated.phase_biases + phase_noise - phase_noise.T
    return CPGParameters(
        amplitudes=amplitudes,
        offsets=offsets,
        frequencies=frequencies,
        phase_biases=phase_biases,
        coupling_weights=mutated.coupling_weights,
    )


def _evaluate(
    artifact: GeneratedRobotArtifact,
    mapper: CPGActionMapper,
    parameters: CPGParameters,
    config: BootstrapControllerConfig,
) -> dict[str, float | int | str | bool]:
    model = _compile_bootstrap_model(artifact)
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    torso_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso")
    initial_x = float(data.xpos[torso_id, 0])
    initial_z = float(data.xpos[torso_id, 2])

    cpg = CPG(parameters, dt=config.control_timestep)
    cpg_state = cpg.reset()
    n_substeps = max(1, int(round(config.control_timestep / model.opt.timestep)))
    steps = max(1, int(round(config.horizon / config.control_timestep)))
    effort = 0.0
    fallen = False

    for _ in range(steps):
        cpg_state = cpg.step(cpg_state)
        action = mapper.action(cpg_state)
        data.ctrl[:] = action
        effort += float(np.mean(np.square(action)))
        mujoco.mj_step(model, data, nstep=n_substeps)
        torso_z = float(data.xpos[torso_id, 2])
        if torso_z < config.fall_height_threshold or torso_z < 0.35 * initial_z:
            fallen = True
            break

    final_x = float(data.xpos[torso_id, 0])
    final_z = float(data.xpos[torso_id, 2])
    simulated_steps = max(1, int(round(data.time / config.control_timestep)))
    mean_effort = effort / simulated_steps
    displacement = final_x - initial_x
    fall_penalty = 1.0 if fallen else 0.0
    score = displacement - config.effort_penalty * mean_effort - fall_penalty
    return {
        "score": float(score),
        "forward_displacement": float(displacement),
        "mean_control_effort": float(mean_effort),
        "fell": fallen,
        "initial_torso_height": initial_z,
        "final_torso_height": final_z,
        "simulated_time": float(data.time),
        "steps": simulated_steps,
    }


def _compile_bootstrap_model(artifact: GeneratedRobotArtifact) -> mujoco.MjModel:
    xml = _bootstrap_scene_xml(artifact.mjcf_xml)
    return mujoco.MjModel.from_xml_string(xml)


def _bootstrap_scene_xml(xml: str) -> str:
    root = ET.fromstring(xml)
    worldbody = root.find("worldbody")
    if worldbody is None:
        msg = "generated robot MJCF has no worldbody"
        raise ValueError(msg)
    torso = worldbody.find("./body[@name='torso']")
    if torso is None:
        msg = "generated robot MJCF has no torso body"
        raise ValueError(msg)
    if torso.find("./freejoint[@name='root_freejoint']") is None:
        freejoint = ET.Element("freejoint", {"name": "root_freejoint"})
        torso.insert(0, freejoint)
    if worldbody.find("./geom[@name='bootstrap_ground']") is None:
        worldbody.insert(
            0,
            ET.Element(
                "geom",
                {
                    "name": "bootstrap_ground",
                    "type": "plane",
                    "size": "20 20 0.1",
                    "friction": "1.2 0.05 0.002",
                },
            ),
        )
    return ET.tostring(root, encoding="unicode")


def _limb_phase(family_role: str) -> float:
    phases = {
        "front_left": 0.0,
        "rear_right": 0.0,
        "front_right": np.pi,
        "rear_left": np.pi,
        "mid_left": 0.0,
        "mid_right": np.pi,
    }
    return phases.get(family_role, 0.0)
