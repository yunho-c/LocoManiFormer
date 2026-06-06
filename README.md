# LocoManiFormer

LocoManiFormer is a Python project scaffold for locomotion and manipulation experiments.
It uses pixi for the development environment, Typer for the command-line interface,
and a local editable checkout of `moojoco` from `../moojoco`.

## Setup

```bash
pixi install
pixi run locomaniformer doctor
```

## Common Tasks

```bash
pixi run cli
pixi run test
pixi run lint
pixi run format
```

The `moojoco` dependency is configured in `pyproject.toml` as a pixi PyPI path
dependency. From this repository, `../moojoco` resolves to `~/GitHub/moojoco`.
