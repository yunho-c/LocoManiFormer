# LocoManiFormer

LocoManiFormer is a Python project for procedural locomotion and manipulation
experiments. It uses Pixi for the development environment, Typer for the command
line interface, and a local editable checkout of `moojoco` from `../moojoco`.

The current tooling can generate MuJoCo robot artifacts, render preview collages,
write deterministic manifests, and create CPG-based bootstrap controller artifacts
for generated robots.

## Setup

Install the Pixi environment and verify the local package plus MooJoCo dependency:

```bash
pixi install
pixi run locomaniformer doctor
```

The `moojoco` dependency is configured in `pyproject.toml` as a Pixi PyPI path
dependency. From this repository, `../moojoco` resolves to `~/GitHub/moojoco`.

## Common Tasks

```bash
pixi run locomaniformer --help
pixi run locomaniformer generate --help
pixi run locomaniformer bootstrap --help
pixi run test
pixi run lint
pixi run format
```

## Generate Robot Artifacts

Generate one deterministic robot artifact:

```bash
pixi run locomaniformer generate robot \
  --seed 0 \
  --family quadruped \
  --output artifacts/robots
```

This writes a folder like:

```text
artifacts/robots/<robot_id>/
  artifact.json
  metadata.json
  robot.xml
  validation.json
```

Useful options:

- `--family`: one of `biped`, `quadruped`, `wheeled_biped`, or `wheeled_quadruped`.
- `--preset`: generation distribution, defaulting to `commercial_surrogate`.
- `--manipulators`: include simple torso-mounted manipulators.
- `--require-mjx`: reject robots that fail the MJX compatibility smoke check.

Generate a deterministic batch and JSONL manifest:

```bash
pixi run locomaniformer generate manifest \
  --count 8 \
  --start-seed 0 \
  --output artifacts/robots \
  --manifest artifacts/manifest.jsonl
```

## Render Preview Collages

Render a grid of generated robots:

```bash
pixi run locomaniformer generate preview \
  --count 8 \
  --start-seed 0 \
  --output artifacts/preview.png \
  --columns 4
```

You can constrain previews to one family:

```bash
pixi run locomaniformer generate preview \
  --family wheeled_quadruped \
  --count 4 \
  --output artifacts/wheeled-preview.png
```

## Bootstrap Controllers

After generating a robot, create a CPG-based bootstrap controller:

```bash
pixi run locomaniformer bootstrap controller \
  --robot-artifact artifacts/robots/<robot_id>/artifact.json \
  --output artifacts/controllers \
  --seed 0 \
  --candidates 32 \
  --horizon 1.5 \
  --render-preview
```

The command writes:

```text
artifacts/controllers/<robot_id>/controller.json
artifacts/controllers/<robot_id>/preview.mp4
```

To create controllers for every accepted robot in a generated manifest, run:

```bash
pixi run locomaniformer bootstrap manifest \
  --manifest artifacts/manifest.jsonl \
  --robot-root artifacts/robots \
  --output artifacts/controllers \
  --seed 0 \
  --candidates 32 \
  --horizon 1.5
```

The manifest command preserves manifest order, uses `--seed` as the base search
seed, increments the seed once per processed robot, and writes one
`controller.json` under `artifacts/controllers/<robot_id>/`. Rejected robots are
skipped by default; pass `--include-rejected` to process every manifest row.
Pass `--render-preview` to also write a 10-second `preview.mp4` for each
controller.

The bootstrap layer:

- builds a heuristic CPG from the robot's action descriptors,
- maps CPG outputs into the generated actuator order,
- holds manipulator actuators at zero by default,
- runs a short native MuJoCo random search around the heuristic gait,
- scores candidates by forward displacement minus effort and fall penalties,
- serializes the best controller parameters and evaluation summary,
- and optionally renders the best gait to an MP4 video through system `ffmpeg`.

For a fast smoke test, use a tiny search:

```bash
pixi run locomaniformer bootstrap controller \
  --robot-artifact artifacts/robots/<robot_id>/artifact.json \
  --candidates 2 \
  --horizon 0.04
```

Preview rendering requires `ffmpeg` on `PATH`. The default preview is a
10-second, 30 FPS, 640x480 MP4 rollout of the best optimized gait.

## Development

Run all tests and lint checks:

```bash
pixi run pytest
pixi run ruff check src tests
```

The configured task aliases are:

```bash
pixi run test
pixi run lint
pixi run format
```
