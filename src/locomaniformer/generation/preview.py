from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import imageio.v3 as iio
import mujoco
import numpy as np

from locomaniformer.generation.config import RobotFamily, RobotGenerationConfig
from locomaniformer.generation.mjcf_robot import MJCFGeneratedRobot
from locomaniformer.generation.sampler import RobotFamilySampler
from locomaniformer.generation.specs import RobotMorphologySpec
from locomaniformer.generation.validation import MorphologyValidator


@dataclass(frozen=True)
class PreviewCollage:
    image: np.ndarray
    robot_ids: tuple[str, ...]
    accepted_count: int


def create_preview_collage(
    config: RobotGenerationConfig,
    *,
    count: int,
    start_seed: int = 0,
    family: RobotFamily | str | None = None,
    columns: int | None = None,
    cell_size: int = 256,
    padding: int = 12,
) -> PreviewCollage:
    if count < 1:
        raise ValueError("count must be at least 1")
    if cell_size < 64:
        raise ValueError("cell_size must be at least 64")

    sampler = RobotFamilySampler(config)
    validator = MorphologyValidator(config)
    cells: list[np.ndarray] = []
    robot_ids: list[str] = []
    accepted_count = 0

    for seed in range(start_seed, start_seed + count):
        spec = sampler.sample(seed=seed, family=family)
        robot = MJCFGeneratedRobot(spec)
        validation = validator.validate(spec, robot)
        accepted_count += int(validation.accepted)
        cells.append(render_robot_preview(spec, width=cell_size, height=cell_size))
        robot_ids.append(spec.robot_id)

    grid_columns = columns or math.ceil(math.sqrt(count))
    return PreviewCollage(
        image=tile_images(cells, columns=grid_columns, padding=padding),
        robot_ids=tuple(robot_ids),
        accepted_count=accepted_count,
    )


def write_preview_collage(collage: PreviewCollage, output_path: Path | str) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(path, collage.image)
    return path


def render_robot_preview(
    spec: RobotMorphologySpec,
    *,
    width: int = 256,
    height: int = 256,
) -> np.ndarray:
    robot = MJCFGeneratedRobot(spec)
    _add_preview_scene(robot)
    model = mujoco.MjModel.from_xml_string(robot.get_mjcf_str(), robot.get_mjcf_assets())
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    renderer = mujoco.Renderer(model, height=height, width=width)
    camera = _preview_camera(robot.nominal_body_height)
    renderer.update_scene(data, camera=camera)
    image = renderer.render()
    renderer.close()
    return image


def tile_images(
    images: list[np.ndarray],
    *,
    columns: int,
    padding: int = 12,
    background: tuple[int, int, int] = (245, 245, 242),
) -> np.ndarray:
    if not images:
        raise ValueError("at least one image is required")
    if columns < 1:
        raise ValueError("columns must be at least 1")
    rows = math.ceil(len(images) / columns)
    cell_height, cell_width, channels = images[0].shape
    canvas_height = rows * cell_height + (rows + 1) * padding
    canvas_width = columns * cell_width + (columns + 1) * padding
    canvas = np.full((canvas_height, canvas_width, channels), background, dtype=np.uint8)

    for index, image in enumerate(images):
        row = index // columns
        column = index % columns
        top = padding + row * (cell_height + padding)
        left = padding + column * (cell_width + padding)
        canvas[top : top + cell_height, left : left + cell_width] = image
    return canvas


def _add_preview_scene(robot: MJCFGeneratedRobot) -> None:
    robot.mjcf_model.visual.headlight.active = True
    robot.mjcf_model.visual.headlight.diffuse = [0.55, 0.55, 0.55]
    robot.mjcf_model.visual.headlight.ambient = [0.35, 0.35, 0.35]
    robot.mjcf_model.worldbody.add(
        "light",
        name="preview_key_light",
        pos=[0.0, -4.0, 6.0],
        dir=[0.0, 1.0, -1.0],
        directional=True,
    )
    robot.mjcf_model.worldbody.add(
        "geom",
        name="preview_floor",
        type="plane",
        size=[3.0, 3.0, 0.05],
        rgba=[0.90, 0.90, 0.87, 1.0],
        friction=[1.0, 0.04, 0.002],
    )


def _preview_camera(nominal_body_height: float) -> mujoco.MjvCamera:
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat = [0.0, 0.0, nominal_body_height * 0.42]
    camera.distance = max(2.5, nominal_body_height * 2.2)
    camera.azimuth = 135.0
    camera.elevation = -18.0
    return camera
