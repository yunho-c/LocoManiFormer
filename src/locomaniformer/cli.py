from __future__ import annotations

import importlib
from importlib import metadata
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from locomaniformer import __version__
from locomaniformer.control import (
    BootstrapControllerConfig,
    load_generated_robot_artifact,
    load_manifest_generated_robot_artifacts,
    optimize_bootstrap_controller,
    optimize_bootstrap_controllers,
    render_bootstrap_preview,
)
from locomaniformer.generation import (
    ParameterRangePreset,
    RobotFamily,
    RobotGenerationConfig,
    create_preview_collage,
    generate_robot_artifact,
    write_preview_collage,
)
from locomaniformer.generation.artifacts import write_manifest

app = typer.Typer(
    help="Command-line tools for LocoManiFormer experiments.",
    no_args_is_help=True,
)
generate_app = typer.Typer(help="Generate procedural robot artifacts.", no_args_is_help=True)
bootstrap_app = typer.Typer(help="Create bootstrap controller artifacts.", no_args_is_help=True)
app.add_typer(generate_app, name="generate")
app.add_typer(bootstrap_app, name="bootstrap")
console = Console()


def _package_version(package_name: str) -> str:
    try:
        return metadata.version(package_name)
    except metadata.PackageNotFoundError:
        return "unknown"


@app.command()
def doctor() -> None:
    """Check that the local project and simulation dependencies are importable."""
    console.print(f"LocoManiFormer: {__version__}")

    for package_name in ("typer", "moojoco"):
        try:
            importlib.import_module(package_name)
        except ImportError as exc:
            console.print(f"[red]{package_name}: not importable ({exc})[/red]")
            raise typer.Exit(code=1) from exc

        console.print(f"{package_name}: {_package_version(package_name)}")


@app.command()
def hello(name: str = typer.Argument("world", help="Name to greet.")) -> None:
    """Print a small smoke-test greeting."""
    console.print(f"Hello, {name}.")


@generate_app.command("robot")
def generate_robot(
    seed: Annotated[int, typer.Option("--seed", help="Deterministic generation seed.")],
    family: Annotated[
        RobotFamily | None,
        typer.Option("--family", help="Robot family to generate."),
    ] = None,
    output: Annotated[
        Path,
        typer.Option(
            "--output",
            "-o",
            help="Directory where the robot artifact folder should be written.",
        ),
    ] = Path("artifacts/robots"),
    manipulators: Annotated[
        bool,
        typer.Option(
            "--manipulators",
            help="Allow simple torso-mounted manipulators for this generated robot.",
        ),
    ] = False,
    preset: Annotated[
        ParameterRangePreset,
        typer.Option("--preset", help="Robot generation distribution preset."),
    ] = ParameterRangePreset.COMMERCIAL_SURROGATE,
    require_mjx: Annotated[
        bool,
        typer.Option(
            "--require-mjx",
            help="Reject robots that fail the MJX compatibility smoke check.",
        ),
    ] = False,
) -> None:
    """Generate one robot artifact and write JSON/XML outputs."""
    config = _generation_config(
        preset=preset,
        allowed_families=(family,) if family is not None else None,
        require_mjx=require_mjx,
        manipulator_probability=1.0 if manipulators else 0.0,
    )
    artifact = generate_robot_artifact(config, seed=seed, family=family)
    paths = artifact.write(output)

    status = "accepted" if artifact.validation_result.accepted else "rejected"
    console.print(f"{artifact.robot_id}: {status}")
    console.print(
        "family={family} actuators={actuators} sensors={sensors} mjcf={mjcf}".format(
            family=artifact.family,
            actuators=artifact.summary_statistics["actuator_count"],
            sensors=artifact.summary_statistics["sensor_count"],
            mjcf=paths["mjcf"],
        )
    )
    if artifact.validation_result.reasons:
        for reason in artifact.validation_result.reasons:
            console.print(f"[red]- {reason}[/red]")
        raise typer.Exit(code=1)


@generate_app.command("manifest")
def generate_manifest(
    count: Annotated[
        int,
        typer.Option("--count", min=1, help="Number of robots to generate."),
    ] = 8,
    start_seed: Annotated[
        int,
        typer.Option("--start-seed", help="First deterministic seed."),
    ] = 0,
    output: Annotated[
        Path,
        typer.Option(
            "--output",
            "-o",
            help="Directory where robot artifact folders should be written.",
        ),
    ] = Path("artifacts/robots"),
    manifest_path: Annotated[
        Path,
        typer.Option("--manifest", help="JSONL manifest path."),
    ] = Path("artifacts/manifest.jsonl"),
    preset: Annotated[
        ParameterRangePreset,
        typer.Option("--preset", help="Robot generation distribution preset."),
    ] = ParameterRangePreset.COMMERCIAL_SURROGATE,
) -> None:
    """Generate a small deterministic robot manifest."""
    config = _generation_config(preset=preset)
    artifacts = []
    for seed in range(start_seed, start_seed + count):
        artifact = generate_robot_artifact(config, seed=seed)
        artifact.write(output)
        artifacts.append(artifact)
    write_manifest(artifacts, manifest_path)
    accepted = sum(artifact.validation_result.accepted for artifact in artifacts)
    console.print(f"wrote {len(artifacts)} robots ({accepted} accepted) to {output}")
    console.print(f"manifest={manifest_path}")


@generate_app.command("preview")
def generate_preview(
    count: Annotated[
        int,
        typer.Option("--count", min=1, help="Number of robots to render."),
    ] = 8,
    start_seed: Annotated[
        int,
        typer.Option("--start-seed", help="First deterministic seed."),
    ] = 0,
    family: Annotated[
        RobotFamily | None,
        typer.Option("--family", help="Robot family to render."),
    ] = None,
    output: Annotated[
        Path,
        typer.Option(
            "--output",
            "-o",
            help="PNG path for the preview collage.",
        ),
    ] = Path("artifacts/preview.png"),
    columns: Annotated[
        int | None,
        typer.Option("--columns", min=1, help="Grid columns. Defaults to a square-ish grid."),
    ] = None,
    cell_size: Annotated[
        int,
        typer.Option("--cell-size", min=64, help="Rendered pixel size for each robot cell."),
    ] = 256,
    manipulators: Annotated[
        bool,
        typer.Option(
            "--manipulators",
            help="Allow simple torso-mounted manipulators in generated robots.",
        ),
    ] = False,
    preset: Annotated[
        ParameterRangePreset,
        typer.Option("--preset", help="Robot generation distribution preset."),
    ] = ParameterRangePreset.COMMERCIAL_SURROGATE,
) -> None:
    """Render generated robots into a regular-grid PNG collage."""
    config = _generation_config(
        preset=preset,
        allowed_families=(family,) if family is not None else None,
        manipulator_probability=1.0 if manipulators else 0.0,
    )
    collage = create_preview_collage(
        config,
        count=count,
        start_seed=start_seed,
        family=family,
        columns=columns,
        cell_size=cell_size,
    )
    path = write_preview_collage(collage, output)
    console.print(
        f"wrote preview={path} robots={len(collage.robot_ids)} accepted={collage.accepted_count}"
    )


@bootstrap_app.command("controller")
def bootstrap_controller(
    robot_artifact: Annotated[
        Path,
        typer.Option(
            "--robot-artifact",
            help="Path to a generated robot artifact.json file.",
            exists=True,
            dir_okay=False,
        ),
    ],
    output: Annotated[
        Path,
        typer.Option(
            "--output",
            "-o",
            help="Directory where the controller artifact folder should be written.",
        ),
    ] = Path("artifacts/controllers"),
    seed: Annotated[int, typer.Option("--seed", help="Deterministic search seed.")] = 0,
    candidates: Annotated[
        int,
        typer.Option("--candidates", min=1, help="Number of random-search candidates."),
    ] = 32,
    horizon: Annotated[
        float,
        typer.Option("--horizon", min=0.02, help="Evaluation horizon in seconds."),
    ] = 1.5,
    effort_penalty: Annotated[
        float,
        typer.Option("--effort-penalty", min=0.0, help="Mean squared action penalty weight."),
    ] = 0.01,
    objective: Annotated[
        str,
        typer.Option("--objective", help="Bootstrap objective label."),
    ] = "forward",
    render_preview: Annotated[
        bool,
        typer.Option(
            "--render-preview",
            help="Render a 10-second MP4 rollout with the best optimized gait.",
        ),
    ] = False,
) -> None:
    """Create a CPG bootstrap controller for a generated robot artifact."""
    artifact = load_generated_robot_artifact(robot_artifact)
    config = BootstrapControllerConfig(
        seed=seed,
        candidates=candidates,
        horizon=horizon,
        effort_penalty=effort_penalty,
        objective=objective,
    )
    controller = optimize_bootstrap_controller(artifact, config)
    path = controller.write(output)
    summary = controller.evaluation_summary
    preview_path = None
    if render_preview:
        preview_path = render_bootstrap_preview(
            artifact,
            controller,
            path.parent / "preview.mp4",
        )
    console.print(f"{controller.robot_id}: controller score={controller.score:.4f}")
    console.print(
        "displacement={disp:.4f} effort={effort:.4f} fell={fell} artifact={path}".format(
            disp=summary["forward_displacement"],
            effort=summary["mean_control_effort"],
            fell=summary["fell"],
            path=path,
        )
    )
    if preview_path is not None:
        console.print(f"preview={preview_path}")


@bootstrap_app.command("manifest")
def bootstrap_manifest(
    manifest: Annotated[
        Path,
        typer.Option(
            "--manifest",
            help="JSONL robot manifest produced by `generate manifest`.",
            exists=True,
            dir_okay=False,
        ),
    ] = Path("artifacts/manifest.jsonl"),
    robot_root: Annotated[
        Path,
        typer.Option(
            "--robot-root",
            help="Directory containing generated robot artifact folders.",
        ),
    ] = Path("artifacts/robots"),
    output: Annotated[
        Path,
        typer.Option(
            "--output",
            "-o",
            help="Directory where controller artifact folders should be written.",
        ),
    ] = Path("artifacts/controllers"),
    seed: Annotated[int, typer.Option("--seed", help="Base deterministic search seed.")] = 0,
    candidates: Annotated[
        int,
        typer.Option("--candidates", min=1, help="Number of random-search candidates per robot."),
    ] = 32,
    horizon: Annotated[
        float,
        typer.Option("--horizon", min=0.02, help="Evaluation horizon in seconds."),
    ] = 1.5,
    effort_penalty: Annotated[
        float,
        typer.Option("--effort-penalty", min=0.0, help="Mean squared action penalty weight."),
    ] = 0.01,
    objective: Annotated[
        str,
        typer.Option("--objective", help="Bootstrap objective label."),
    ] = "forward",
    include_rejected: Annotated[
        bool,
        typer.Option("--include-rejected", help="Also process rejected robots from the manifest."),
    ] = False,
    render_preview: Annotated[
        bool,
        typer.Option(
            "--render-preview",
            help="Render a 10-second MP4 rollout for every generated controller.",
        ),
    ] = False,
) -> None:
    """Create CPG bootstrap controllers for every robot in a manifest."""
    artifacts = load_manifest_generated_robot_artifacts(
        manifest,
        robot_root,
        include_rejected=include_rejected,
    )
    if not artifacts:
        console.print("[yellow]manifest contained no robots to process[/yellow]")
        return
    config = BootstrapControllerConfig(
        seed=seed,
        candidates=candidates,
        horizon=horizon,
        effort_penalty=effort_penalty,
        objective=objective,
    )
    controllers = optimize_bootstrap_controllers(artifacts, config)
    for artifact, controller in zip(artifacts, controllers, strict=True):
        path = controller.write(output)
        summary = controller.evaluation_summary
        preview_path = None
        if render_preview:
            preview_path = render_bootstrap_preview(
                artifact,
                controller,
                path.parent / "preview.mp4",
            )
        console.print(
            (
                "{robot_id}: score={score:.4f} displacement={disp:.4f} "
                "fell={fell} artifact={path}{preview}"
            ).format(
                robot_id=controller.robot_id,
                score=controller.score,
                disp=summary["forward_displacement"],
                fell=summary["fell"],
                path=path,
                preview=f" preview={preview_path}" if preview_path is not None else "",
            )
        )
    console.print(f"wrote {len(controllers)} controller artifacts to {output}")


def _generation_config(
    *,
    preset: ParameterRangePreset,
    allowed_families: tuple[RobotFamily, ...] | None = None,
    require_mjx: bool = False,
    manipulator_probability: float = 0.0,
) -> RobotGenerationConfig:
    return RobotGenerationConfig.from_preset(
        preset,
        allowed_families=allowed_families,
        require_mjx=require_mjx,
        manipulator_probability=manipulator_probability,
    )
