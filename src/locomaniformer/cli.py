from __future__ import annotations

import importlib
from importlib import metadata
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from locomaniformer import __version__
from locomaniformer.generation import (
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
app.add_typer(generate_app, name="generate")
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
    require_mjx: Annotated[
        bool,
        typer.Option(
            "--require-mjx",
            help="Reject robots that fail the MJX compatibility smoke check.",
        ),
    ] = False,
) -> None:
    """Generate one robot artifact and write JSON/XML outputs."""
    config = RobotGenerationConfig.conservative(
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
) -> None:
    """Generate a small deterministic robot manifest."""
    config = RobotGenerationConfig.conservative()
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
) -> None:
    """Render generated robots into a regular-grid PNG collage."""
    config = RobotGenerationConfig.conservative(
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
        f"wrote preview={path} robots={len(collage.robot_ids)} "
        f"accepted={collage.accepted_count}"
    )
