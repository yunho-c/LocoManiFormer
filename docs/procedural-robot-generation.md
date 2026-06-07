# Procedural Robot Generation

## Purpose

LocoManiFormer needs a large and diverse population of robots for training
generalist locomotion and manipulation policies. The generator should produce
MuJoCo-compatible morphologies that vary across body plans, proportions,
actuation, sensors, contact geometry, mass properties, and manipulation
attachments while remaining physically plausible enough for large-scale RL.

The target is not a single benchmark robot. The target is an embodiment factory:
given a seed and a generation configuration, it should produce a reproducible
robot asset, metadata describing the embodiment, and validation results that say
whether the robot is suitable for training. The default distribution is a
commercial-surrogate distribution: it should cover the design space of common
commercial quadruped and biped platforms rather than arbitrary abstract robots.

This document specifies a MooJoCo-based architecture for generating:

- bipeds,
- quadrupeds,
- wheeled bipeds,
- wheeled quadrupeds,
- and any of the above with optional loco-manipulation appendages.

## Design Goals

- Generate many valid robot bodies without hand-authoring MJCF files.
- Keep morphology randomization structured rather than arbitrary so generated
  robots are diverse but usually trainable.
- Support both native MuJoCo and MJX wherever the generated assets use supported
  physics features.
- Export model-facing embodiment metadata so transformer policies can condition
  on robot structure instead of inferring it only from observations.
- Make generation reproducible from a robot family, seed, version, and config.
- Separate morphology generation from task and terrain generation so the same
  robot can be reused across locomotion and manipulation tasks.

## Architecture Overview

The generator should follow the same broad pattern used by MooJoCo and BRT:
configuration objects define parameters, MJCF builder classes turn them into
MuJoCo XML, and environment classes consume the resulting morphology.

The core pipeline is:

1. Sample a structured `RobotMorphologySpec`.
2. Build an `MJCFGeneratedRobot` from the spec.
3. Compile the MJCF with MuJoCo.
4. Validate geometry, dynamics, sensors, and action spaces.
5. Export a `GeneratedRobotArtifact` containing MJCF, assets, metadata, and a
   validation report.
6. Use the artifact in MooJoCo MJC/MJX environments for rollout generation.

Recommended future module layout:

```text
src/locomaniformer/
  generation/
    config.py          # RobotGenerationConfig and parameter ranges
    specs.py           # RobotMorphologySpec and component specs
    sampler.py         # RobotFamilySampler and seeded sampling
    mjcf_robot.py      # MJCFGeneratedRobot and component builders
    validation.py      # MorphologyValidator and ValidationResult
    metadata.py        # EmbodimentMetadata export
    artifacts.py       # GeneratedRobotArtifact serialization
```

## Core Interfaces

### RobotGenerationConfig

`RobotGenerationConfig` controls what can be generated in a run. It should be
treated as the top-level reproducibility contract for morphology datasets.

It should include:

- allowed robot families,
- allowed manipulator configurations,
- global scale range,
- parameter range preset: `commercial_surrogate`, `conservative`, `broad`,
  `extreme`, or `heldout`,
- validation strictness,
- MJX compatibility requirement,
- dataset split label,
- generator version,
- and random seed policy.

The config should avoid storing a sampled robot directly. It defines the
distribution; `RobotMorphologySpec` is one draw from that distribution.

### RobotMorphologySpec

`RobotMorphologySpec` is an immutable structured description of one robot before
MJCF construction. It should contain only data, not builder logic.

Top-level fields:

- `robot_id`: stable ID derived from generator version, config hash, and seed.
- `family`: `biped`, `quadruped`, `wheeled_biped`, or `wheeled_quadruped`.
- `global_scale`: scalar applied to body and limb ranges.
- `body`: torso and root body parameters.
- `limbs`: ordered leg modules.
- `wheels`: optional wheel modules associated with distal limbs.
- `manipulators`: optional arm or gripper modules.
- `sensors`: proprioceptive, contact, wheel, IMU-like, and end-effector sensors.
- `actuation`: actuator type and strength policy.
- `physics`: friction, damping, armature, density, and solver-relevant values.
- `symmetry`: mirroring and perturbation settings.

Each sub-spec should include sampled values and the source range name. Keeping the
source range makes dataset audits easier.

### MJCFGeneratedRobot

`MJCFGeneratedRobot` should subclass MooJoCo's `MJCFMorphology`. It should build
the robot from a `RobotMorphologySpec` and expose the same surface that MooJoCo
environments already expect.

Builder responsibilities:

- configure compiler defaults and contact defaults,
- build torso/root body,
- attach legs in a deterministic order,
- attach feet or wheel modules at distal leg segments,
- attach manipulators to configured mounts,
- create actuators and sensors,
- assign stable names for all bodies, geoms, joints, actuators, and sensors,
- expose MJCF XML and assets through MooJoCo's existing helpers.

The builder should not sample random values. It should only consume the spec.

### GeneratedRobotArtifact

`GeneratedRobotArtifact` is the final product of generation.

It should include:

- `robot_id`,
- `seed`,
- `generator_version`,
- `family`,
- `morphology_spec`,
- `mjcf_xml`,
- `mjcf_assets`,
- `action_descriptor`,
- `observation_descriptor`,
- `embodiment_metadata`,
- `validation_result`,
- and summary statistics such as total mass, body height, limb count, actuator
  count, sensor count, wheel count, and manipulator count.

Artifacts should be serializable so training jobs can either regenerate robots
from seeds or load fixed robot manifests.

## Morphology Model

### Body

The body module defines the root body and major inertial frame. It should support:

- box, capsule, ellipsoid-like, or composite torso shapes,
- torso length, width, height, and orientation,
- root mass and density,
- center of mass offset,
- inertial tensor perturbation,
- pelvis or shoulder mount offsets,
- base collision geometry,
- and optional payload masses for domain randomization.

The root free joint should be added at the morphology attachment point, following
the BRT pattern of attaching a morphology to an arena with `free_joint=True`.

### Legs

Legs should be represented as repeated modules with mirrored defaults and
per-leg perturbations.

Each leg should support:

- hip mount offset from torso frame,
- optional yaw, roll, and pitch hip axes modeled as stacked hinge/revolute
  joints,
- upper segment length, radius, mass, and COM offset,
- knee joint axis, range, stiffness, damping, and armature,
- lower segment length, radius, mass, and COM offset,
- ankle joint or distal wheel joint,
- contact geometry,
- and actuator strength scaled by segment size and robot mass.

Initial v1 commercial-surrogate leg templates:

- Quadruped leg: hip roll/abduction, hip pitch, and knee pitch.
- Biped leg: hip yaw, hip roll, hip pitch, knee pitch, and ankle pitch.
- Wheeled biped leg: hip yaw, hip roll, hip pitch, knee pitch, and distal wheel
  drive.

These templates intentionally avoid actuated ball joints. Commercial robots
usually approximate multi-axis hip motion with multiple single-axis actuators,
transmissions, and bearings; the generated MJCF should model those axes as
separate hinge joints.

Broader non-default presets can include:

- 2-DoF leg: hip pitch and knee pitch.
- 3-DoF leg: hip roll, hip pitch, knee pitch.
- 4-DoF leg: hip yaw, hip roll, hip pitch, knee pitch.
- 5-DoF leg: 4-DoF leg plus ankle pitch.

The default sampler should choose commercial-surrogate templates per family,
then apply small left/right/front/back perturbations. Large asymmetry and
unusual extra axes should be allowed only in `broad`, `heldout`, or `extreme`
presets.

### Feet

Foot modules should be replaceable terminal components.

Foot parameters:

- shape: sphere, capsule, box, or composite sole,
- size and aspect ratio,
- toe or heel extension,
- local offset from ankle or lower segment,
- mass and COM offset,
- friction coefficients,
- contact sensor layout,
- and visual material.

Feet should use conservative collision geometry by default to avoid unstable
initial contacts.

### Wheels

Wheeled counterparts replace feet with wheel modules. The wheel should be a
first-class terminal module rather than a special case inside the leg.

Wheel parameters:

- radius,
- width,
- mass and inertia,
- local axle offset,
- axle joint axis,
- drive torque range,
- optional steering joint,
- passive suspension compliance,
- rolling friction,
- lateral friction,
- contact material,
- wheel velocity sensor,
- and optional encoder-like position sensor.

Initial wheel templates:

- fixed drive wheel: one hinge actuator at the distal limb.
- steerable drive wheel: steering hinge plus drive hinge.
- passive caster: unactuated wheel for stability experiments.

For MJX compatibility, the first implementation should prefer simple hinge
joints and primitive geoms over complex tire contact models.

### Manipulators

Manipulators should be optional generated appendages for loco-manipulation.
They should attach to torso shoulder mounts, not to leg modules.

Manipulator parameters:

- number of arms: zero, one, or two,
- mount side and offset,
- link count,
- link lengths and radii,
- joint axes and limits,
- actuator type and strength,
- gripper type: fixed pad, parallel jaw, hook, or simple spherical end-effector,
- end-effector mass and contact geometry,
- wrist or end-effector sensors,
- and task frame site.

The conservative preset should start with simple 2-DoF or 3-DoF arms. Broader
presets can add wrist joints and grippers.

## Sampling Strategy

Sampling should be hierarchical:

1. Choose family.
2. Choose global scale.
3. Sample torso dimensions and mass.
4. Sample limb template and mount geometry.
5. Sample segment proportions relative to body scale.
6. Sample distal module: foot or wheel.
7. Sample optional manipulators.
8. Sample actuator and sensor layout.
9. Sample physics perturbations.
10. Validate and either accept or reject.

This hierarchy prevents impossible combinations, such as tiny torsos with
overlarge wheels or arms with unreachable mounts.

Recommended distribution policy:

- Use bounded continuous distributions for dimensions and masses.
- Use log-uniform distributions for quantities spanning orders of magnitude,
  such as damping, armature, and actuator strength multipliers.
- Use categorical distributions for templates and module presence.
- Use correlated sampling for physically linked values, such as limb length and
  limb mass.
- Keep a rejection budget per robot so dataset generation cannot stall forever.

## Randomizable Parameters

The generator should support at least these parameter groups:

- Geometry: link lengths, radii, torso dimensions, wheel radius/width, foot size,
  manipulator reach, joint offsets, mount offsets, and contact shape sizes.
- Dynamics: link mass, density, COM offset, inertia scaling, payload mass,
  damping, stiffness, armature, friction, and restitution.
- Kinematics: joint axes, joint ranges, limb sweep angles, hip spacing, ankle
  placement, wheel axle placement, and manipulator mount frame.
- Actuation: actuator type, torque range, position gain, gear ratio, control
  range, force range, and motor-to-link strength scaling.
- Sensors: sensor presence, contact sensor count, site placement, proprioceptive
  noise metadata, wheel encoders, IMU-like body sensors, and end-effector sites.
- Appearance: color/material assignment for debugging and dataset inspection.

## Validation

Validation is mandatory. A diverse invalid robot population is not useful for
training.

`MorphologyValidator` should run deterministic checks before rollout:

- MJCF compiles with MuJoCo.
- All names are unique and stable.
- No NaN or infinite sampled values.
- Total mass is within configured bounds.
- Inertias are positive and plausible.
- Joint axes are non-zero.
- Joint ranges are ordered and non-degenerate.
- Actuator ranges match actuator count and expected control mode.
- Initial qpos is inside joint limits.
- Initial geometry is not deeply self-intersecting.
- Feet or wheels start close to the support surface.
- Contact geoms exist for all distal modules.
- Sensors referenced by observables exist.
- MJC reset succeeds.
- MJX reset succeeds when `require_mjx=True`.

Validation should return a structured `ValidationResult`:

- `accepted: bool`,
- `reasons: list[str]`,
- `warnings: list[str]`,
- `metrics: dict[str, float | int | str]`.

Rejected robots should be saved in audit logs, not silently discarded, so the
sampling distribution can be improved.

## Action And Observation Design

Generated robots will not have a fixed action dimension. The environment layer
should derive action and observation descriptors from the MJCF model, as MooJoCo
already does for action spaces and observables.

Action descriptor fields:

- actuator name,
- actuator type,
- joint or tendon target,
- body part path,
- control range,
- force range,
- normalized action index,
- and symmetry group.

Observation descriptor fields:

- observation name,
- source sensor or derived state,
- shape,
- bounds,
- body part path,
- frame,
- and normalization policy.

The policy model should receive both numeric observations and embodiment
metadata. This avoids forcing a transformer to infer morphology solely from
state vectors whose ordering changes across generated robots.

## Embodiment Metadata For Transformers

Each robot should export a graph-like metadata representation:

- nodes for torso, links, wheels, feet, joints, actuators, sensors, and end
  effectors,
- edges for parent-child kinematic relationships,
- node features for dimensions, mass, COM, joint axis, joint range, actuator
  strength, and sensor type,
- symmetry labels for mirrored limbs,
- family label,
- and normalized scalar summary features.

This metadata can be used in several ways:

- prepend embodiment tokens to a transformer context,
- condition action heads on actuator descriptors,
- mask invalid action slots for smaller robots,
- group trajectories by morphology family,
- and create held-out morphology evaluation splits.

The metadata schema should be versioned independently from the generator so old
datasets remain interpretable.

## MooJoCo Integration

The generated robot should reuse MooJoCo conventions:

- `RobotMorphologySpec` should play the same role as BRT's FPRS morphology
  specification objects.
- `MJCFGeneratedRobot` should subclass `MJCFMorphology`.
- Tasks should attach the generated morphology to an arena through
  `from_morphology_and_arena(...)`.
- MJC environments should use NumPy observables and native MuJoCo data.
- MJX environments should use JAX observables and avoid unsupported MuJoCo
  features where possible.

The first arena should be simple:

- flat ground,
- optional low obstacles,
- optional target object,
- configurable friction,
- cameras for top and side views,
- and no robot-family-specific assumptions.

Manipulation tasks can add object arenas later, but robot generation should stay
task-independent.

## Dataset Generation Strategy

Large-scale generation should produce manifests rather than only ad hoc XML
files.

Recommended manifest fields:

- robot ID,
- seed,
- generator config hash,
- dataset split,
- family,
- accepted validation status,
- artifact path,
- metadata path,
- summary statistics,
- and known compatibility flags such as `mjc_ok` and `mjx_ok`.

Split policy:

- training: broad in-distribution families and parameters,
- validation: same families with different seeds,
- held-out morphology: shifted proportions, rare templates, unusual wheel/leg
  combinations, or unseen manipulator placements,
- stress: extreme but valid dynamics and geometry.

The generator should make it easy to regenerate exact robots from the manifest.

## Implementation Milestones

### Milestone 1: Legged Generator

- Implement biped and quadruped specs.
- Build torso, legs, feet, actuators, and sensors.
- Compile MJCF and validate native MuJoCo reset.
- Export metadata and descriptors.

### Milestone 2: Wheeled Counterparts

- Add wheel terminal modules.
- Support fixed drive wheels and steerable drive wheels.
- Add wheel sensors and wheel-specific validation.
- Verify MJC and MJX compatibility for simple wheel bodies.

### Milestone 3: Manipulators

- Add optional torso-mounted arms.
- Add simple grippers or end-effector sites.
- Export end-effector metadata and observations.
- Validate locomotion-only robots and loco-manipulation robots through the same
  artifact path.

### Milestone 4: Dataset Manifests

- Add deterministic generation manifests.
- Add split generation.
- Add diversity audits.
- Add rejection reports and summary plots.

## Test Plan

Minimum tests for the first implementation:

- sampler reproducibility for fixed seeds,
- generated values stay inside configured ranges,
- robot IDs are stable for equivalent specs,
- every family compiles to MuJoCo MJCF,
- action count equals actuator count,
- observation descriptors match generated sensors,
- validation rejects malformed joint axes and invalid ranges,
- MJC reset smoke test for every family,
- MJX reset smoke test for MJX-compatible families,
- wheel robots expose wheel velocity observations,
- manipulator robots expose end-effector sites,
- manifest regeneration produces the same robot ID and summary statistics.

Longer-running tests should sample hundreds or thousands of robots and report:

- acceptance rate,
- rejection reasons,
- family balance,
- total mass distribution,
- actuator count distribution,
- body height distribution,
- wheel radius distribution,
- limb length distribution,
- manipulator reach distribution,
- and MJX compatibility rate.

## Practical Defaults

For the first implementation, choose conservative defaults:

- primitive geoms only,
- hinge joints only,
- no actuated ball joints,
- direct motor or position actuators,
- flat-ground validation,
- simple collision shapes,
- no tendon routing,
- no mesh assets,
- and optional manipulators disabled unless requested by config.

These defaults make the generator easier to validate and keep MJX compatibility
high. Broader morphology features can be added once the acceptance rate and
training pipeline are stable.
