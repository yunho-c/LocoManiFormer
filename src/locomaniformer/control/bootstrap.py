from __future__ import annotations

import shutil
import subprocess
import xml.etree.ElementTree as ET
from collections.abc import Iterable
from pathlib import Path

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


def optimize_bootstrap_controllers(
    artifacts: Iterable[GeneratedRobotArtifact],
    config: BootstrapControllerConfig,
) -> tuple[BootstrapControllerArtifact, ...]:
    controllers: list[BootstrapControllerArtifact] = []
    for index, artifact in enumerate(artifacts):
        per_robot_config = BootstrapControllerConfig(
            seed=config.seed + index,
            candidates=config.candidates,
            horizon=config.horizon,
            effort_penalty=config.effort_penalty,
            fall_height_threshold=config.fall_height_threshold,
            controller_version=config.controller_version,
            control_timestep=config.control_timestep,
            objective=config.objective,
        )
        controllers.append(optimize_bootstrap_controller(artifact, per_robot_config))
    return tuple(controllers)


def render_bootstrap_preview(
    artifact: GeneratedRobotArtifact,
    controller: BootstrapControllerArtifact,
    output_path: Path | str,
    *,
    duration: float = 10.0,
    fps: int = 30,
    width: int = 640,
    height: int = 480,
) -> Path:
    if duration <= 0.0:
        raise ValueError("duration must be positive")
    if fps <= 0:
        raise ValueError("fps must be positive")
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")
    if width % 2 != 0 or height % 2 != 0:
        raise ValueError("width and height must be even for MP4 encoding")

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        msg = "ffmpeg is required for --render-preview but was not found on PATH"
        raise RuntimeError(msg)

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model = _compile_bootstrap_model(artifact)
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    mapper = CPGActionMapper.from_artifact(artifact)
    parameters = CPGParameters.from_dict(controller.parameters)
    control_timestep = float(
        controller.evaluation_summary.get(
            "control_timestep",
            BootstrapControllerConfig().control_timestep,
        )
    )
    cpg = CPG(parameters, dt=control_timestep)
    cpg_state = cpg.reset()
    n_substeps = max(1, int(round(control_timestep / model.opt.timestep)))
    frame_count = max(1, int(round(duration * fps)))
    renderer = mujoco.Renderer(model, height=height, width=width)
    camera = _rollout_camera(model, data)

    command = [
        ffmpeg,
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-vcodec",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(path),
    ]
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    assert process.stdin is not None
    try:
        for frame_index in range(frame_count):
            target_time = (frame_index + 1) / fps
            while data.time < target_time:
                cpg_state = cpg.step(cpg_state)
                data.ctrl[:] = mapper.action(cpg_state)
                mujoco.mj_step(model, data, nstep=n_substeps)
            _update_rollout_camera(camera, model, data)
            renderer.update_scene(data, camera=camera)
            process.stdin.write(renderer.render().tobytes())
    except BrokenPipeError as exc:
        stderr = _finish_ffmpeg(process)
        msg = f"ffmpeg stopped while writing {path}: {stderr}"
        raise RuntimeError(msg) from exc
    finally:
        renderer.close()

    stderr = _finish_ffmpeg(process)
    if process.returncode != 0:
        msg = f"ffmpeg failed while writing {path}: {stderr}"
        raise RuntimeError(msg)
    return path


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
        "control_timestep": config.control_timestep,
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


def _rollout_camera(model: mujoco.MjModel, data: mujoco.MjData) -> mujoco.MjvCamera:
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.distance = 3.0
    camera.azimuth = 135.0
    camera.elevation = -18.0
    _update_rollout_camera(camera, model, data)
    return camera


def _update_rollout_camera(
    camera: mujoco.MjvCamera,
    model: mujoco.MjModel,
    data: mujoco.MjData,
) -> None:
    torso_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso")
    torso_pos = data.xpos[torso_id]
    camera.lookat = [
        float(torso_pos[0]),
        float(torso_pos[1]),
        max(0.2, float(torso_pos[2]) * 0.65),
    ]
    camera.distance = max(2.5, float(torso_pos[2]) * 3.0)


def _finish_ffmpeg(process: subprocess.Popen[bytes]) -> str:
    if process.stdin is not None:
        process.stdin.close()
        process.stdin = None
    _, stderr = process.communicate()
    return stderr.decode("utf-8", errors="replace").strip()
