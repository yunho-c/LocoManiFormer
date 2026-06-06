from __future__ import annotations

import importlib
from importlib import metadata

import typer
from rich.console import Console

from locomaniformer import __version__

app = typer.Typer(
    help="Command-line tools for LocoManiFormer experiments.",
    no_args_is_help=True,
)
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
