from pathlib import Path

from typer.testing import CliRunner

import locomaniformer.cli as cli
from locomaniformer.generation import (
    RobotFamily,
    RobotGenerationConfig,
    generate_robot_artifact,
)
from locomaniformer.generation.artifacts import write_manifest


def test_hello_command() -> None:
    result = CliRunner().invoke(cli.app, ["hello", "tester"])

    assert result.exit_code == 0
    assert "Hello, tester." in result.stdout


def test_bootstrap_controller_render_preview_flag_writes_path(tmp_path, monkeypatch) -> None:
    artifact = generate_robot_artifact(
        RobotGenerationConfig.conservative(),
        seed=31,
        family=RobotFamily.WHEELED_BIPED,
    )
    paths = artifact.write(tmp_path / "robots")

    def fake_render_preview(*args, **kwargs) -> Path:
        del args, kwargs
        path = tmp_path / "controllers" / artifact.robot_id / "preview.mp4"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fake mp4")
        return path

    monkeypatch.setattr(cli, "render_bootstrap_preview", fake_render_preview)
    result = CliRunner().invoke(
        cli.app,
        [
            "bootstrap",
            "controller",
            "--robot-artifact",
            str(paths["artifact"]),
            "--output",
            str(tmp_path / "controllers"),
            "--candidates",
            "1",
            "--horizon",
            "0.02",
            "--render-preview",
        ],
    )

    assert result.exit_code == 0, result.stdout
    preview = tmp_path / "controllers" / artifact.robot_id / "preview.mp4"
    assert preview.exists()
    assert "preview=" in result.stdout


def test_bootstrap_manifest_render_preview_flag_writes_one_preview_per_robot(
    tmp_path,
    monkeypatch,
) -> None:
    artifacts = [
        generate_robot_artifact(
            RobotGenerationConfig.conservative(),
            seed=32,
            family=RobotFamily.WHEELED_BIPED,
        ),
        generate_robot_artifact(
            RobotGenerationConfig.conservative(),
            seed=33,
            family=RobotFamily.QUADRUPED,
        ),
    ]
    robot_root = tmp_path / "robots"
    for artifact in artifacts:
        artifact.write(robot_root)
    manifest = tmp_path / "manifest.jsonl"
    write_manifest(artifacts, manifest)

    def fake_render_preview(*args, **kwargs) -> Path:
        del kwargs
        path = Path(args[2])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fake mp4")
        return path

    monkeypatch.setattr(cli, "render_bootstrap_preview", fake_render_preview)
    result = CliRunner().invoke(
        cli.app,
        [
            "bootstrap",
            "manifest",
            "--manifest",
            str(manifest),
            "--robot-root",
            str(robot_root),
            "--output",
            str(tmp_path / "controllers"),
            "--candidates",
            "1",
            "--horizon",
            "0.02",
            "--render-preview",
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert result.stdout.count("preview=") == len(artifacts)
    for artifact in artifacts:
        preview = tmp_path / "controllers" / artifact.robot_id / "preview.mp4"
        assert preview.exists()
