# ramms-fleet

Federated learning experiments on fleets of simulated rovers for
[RAMMS](https://github.com/rammp-org/ramms-sim).

Assistive robots work in people's homes, and their sensor data is private. The
question here is how well a fleet can learn together when each robot keeps its
data local and only shares model updates, especially when the robots and their
environments differ. Small differential-drive rovers stand in for the robots,
and the shared task is predicting their own collisions.

## Status

| Step | State | Where |
|---|---|---|
| Rover model, MuJoCo fleet runner, viewer | done | #1 |
| Federated training with a process per rover, baselines, seed sweeps | done | #2 |
| Rovers with different speeds and sensor noise, fine-tuning, videos | done | #3 |
| Distance-based collision labels | done | #4 |
| RAMMS backend over the URLab bridge, federated training on RAMMS data | done | #5 |
| Front camera as a model input | done | #6 |
| Pedestrians in both backends | running | #7 |

Headline results (details and every table in [docs/results.md](docs/results.md)):

- FedAvg beats each rover training alone in every seed of every experiment
  (48 of 48), by about +0.03 to +0.07 AUPRC.
- It closes most of the gap to pooling all rovers' data: 0.890 against 0.907
  for centralized training, with local-only at 0.821 (5 seeds).
- Five local epochs over 10 rounds match one epoch over 50 rounds, so
  communication can drop 5x without losing accuracy.
- Slow rovers, which collide rarely and so have few examples, gain the most
  from federation.
- FedProx and per-rover fine-tuning never beat plain FedAvg beyond seed noise.
- Results hold inside RAMMS, models move between MuJoCo and RAMMS with almost
  no loss, and adding the front camera raises FedAvg by +0.03 (3 of 3 seeds).

## Install

Python 3.10 or newer and [uv](https://docs.astral.sh/uv/).

```bash
uv venv
uv pip install torch --index-url https://download.pytorch.org/whl/cpu   # optional: skip the CUDA wheels
uv pip install -e ".[dev,video,ramms]"                                   # video: rendering; ramms: RAMMS backend
.venv/bin/pytest
```

The `video` extra (imageio, imageio-ffmpeg, Pillow) is only needed for
`ramms-fleet-render`. Rendering uses MuJoCo's EGL backend by default; set
`MUJOCO_GL=glfw` if EGL is unavailable.

## Workflow

One experiment, end to end:

```bash
# 1. Collect: one dataset file per rover
.venv/bin/ramms-fleet-collect --rovers 8 --seconds 600 --out data/run0

# 2. Watch the fleet (optional)
.venv/bin/ramms-fleet-view --rovers 8

# 3. Train the reference points: local-only and centralized
.venv/bin/ramms-fleet-baseline --data data/run0 --out results/run0 --epochs 50

# 4. Federated training: a server process plus one process per rover
.venv/bin/ramms-fleet-federate --data data/run0 --out results/run0 --rounds 10 --local-epochs 5

# 5. Score every model on every rover's held-out data
.venv/bin/ramms-fleet-eval --data data/run0 --results results/run0 --finetune-epochs 5
```

Keep training budgets equal when comparing: baselines train `--epochs`, and a
federated run trains `--rounds` x `--local-epochs` epochs on each rover.

To repeat this over seeds and summarize, use `ramms-fleet-sweep`, which runs
every step per seed, skips finished steps on re-run, and writes `summary.json`
and `per_rover.csv`:

```bash
.venv/bin/ramms-fleet-sweep --seeds 0 1 2 3 4 --rounds 10 --local-epochs 5 --proximal-mus \
    --finetune-epochs 5 --evaluate-every 2 --data-root data/sweep --out results/sweep
```

## How it works

### Rover and arenas

`src/ramms_fleet/assets/rover.xml` is plain MJCF (no compiler overrides, no
euler angles, no plugins) so the same file can load in MuJoCo and in RAMMS
through URLab:

- 0.20 x 0.15 m chassis, two velocity-controlled wheels (35 mm radius,
  190 mm track) and a frictionless rear caster
- five rangefinders fanned from -60 to +60 degrees, 2 m range
- IMU (accelerometer and gyro) and wheel velocity sensors
- a touch sensor around the chassis, used as the bumper
- a forward camera (not used by the MuJoCo backend yet)

Commands are body velocities: linear m/s and angular rad/s, at 20 Hz on a 5 ms
physics step.

Each rover drives alone in a walled 3 x 3 m arena with seeded random obstacles.
Rover *i* gets clutter spread linearly from `--clutter-min` to `--clutter-max`
obstacles per square metre. The exploration policy (`policy.py`) wanders with a
random turn rate and avoids obstacles only weakly (`--avoid-gain`), so rovers
collide often enough to produce positive labels, then backs off and spins after
each bump.

For collection, each rover gets its own small MuJoCo model, which steps faster
than one combined world and mirrors one client per environment. 8 rovers
collect at about 15x real time on one core. `MujocoFleet(..., shared_world=True)`
puts all cells in one model instead, which the viewer and renderer use.

### Rovers that differ

`--speed MIN MAX`, `--range-noise MIN MAX`, `--accel-noise MIN MAX`, and
`--gyro-noise MIN MAX` give each rover its own cruise speed (m/s) and Gaussian
sensor noise (standard deviations in m, m/s², rad/s). Values are spread evenly
over the fleet in seeded shuffles, independent of clutter and of each other; one
noise order covers all sensors, so a noisy rover is noisy on every sensor.
Noise corrupts what the rover observes (and so how it drives and what it
records); the bumper and pose stay exact. With the defaults, collection
reproduces earlier datasets bit for bit.

### Pedestrians

`--pedestrians MIN MAX` puts scripted pedestrians in each arena, with counts
spread over the fleet in a seeded shuffle (independent of clutter). A
pedestrian is an upright capsule (0.07 m radius, 0.45 m tall) on two slide
joints with a velocity actuator per axis. `crowd.py` walks each one between
random goals at its own speed (0.15 to 0.35 m/s), pauses at each goal, and
steers around walls, obstacles, and other pedestrians. Half of them also steer
around the rover; the rest walk as if it were not there.

The walking model only chooses velocities, and both backends apply them as
actuator commands, so pedestrians push and get pushed like any other body and
every rangefinder and camera sees them. Positions come back as
`FleetObs.pedestrians` and are saved per rover; a collision counts as involving
a pedestrian when one is within 0.25 m of the rover's centre
(`pedestrian_collisions_per_min` in `meta.json`).

These are not RAMMS's own crowd agents. RammsCrowd pedestrians are Unreal Mass
entities with no interface for reading their positions from outside the
editor, and they would only exist in RAMMS; scripted pedestrians behave the
same in both simulators and repeat exactly with the seed.

### Datasets and labels

Each run directory has `meta.json` (settings, per-rover profiles, and
statistics) and one `rover_NN.npz` per rover, as a federated client would keep
it:

| Array | Shape | Meaning |
|-------|-------|---------|
| `features` | (T, 17) | 5 ranges, accel, gyro, 2 wheel velocities, commanded v and w (names in `meta.json`) |
| `label` | (T,) | time label: a new bumper contact within the next `--horizon` seconds (default 0.5) |
| `valid` | (T,) | cruising and not in contact; train and evaluate on these samples |
| `bump` | (T,) | bumper in contact |
| `episode` | (T,) | increments when a tipped-over rover is reset |
| `pose` | (T, 3) | x, y in the rover's cell and yaw |

Labels are self-supervised from the bumper. Two definitions are available
wherever data is loaded (`--label`):

- `time` (default): the label stored at collection, a collision within 0.5 s.
- `distance`: relabeled on load from `pose`, a collision within `--horizon-m`
  metres of travel (default 0.15, which equals 0.5 s at 0.3 m/s). A time
  horizon covers more ground for a fast rover, so use `distance` when speeds
  differ. Scores are not comparable across label types.

### Model and training

`learning.py` holds everything shared by baselines, clients, and evaluation.
`--inputs` picks what a model sees: `features` (default), `camera`, or `both`.
Camera models add a small CNN over the latest frame, average-pooled to 32 x 24,
fused with the feature MLP for `both`. When a dataset has frames, samples with a
stale frame are dropped for every input choice, so comparisons share samples.
A sample stacks the last 4 control steps of features (68 inputs) into an MLP
(68 -> 64 -> 64 -> 1). Features are scaled by fixed physical limits rather than
dataset statistics, so no client needs another client's data to normalize.
Training uses Adam with the positive class re-weighted by each dataset's own
negative-to-positive ratio; `--proximal-mu` adds the FedProx penalty. Each
rover's final 20% of time is its test split, so test samples never sit next to
training samples. Every method starts from the same initial weights for a seed.

### Federated training

`ramms-fleet-federate` runs the Flower 1.30 Deployment Engine on one machine:

- a SuperLink, whose ServerApp runs FedAvg (`fl/server_app.py`) and never
  opens rover data
- one SuperNode per rover, started with `--node-config "data-path='.../rover_NN.npz'"`
  so it only knows its own file; every ClientApp task (`fl/client_app.py`)
  runs in a fresh child process
- `flwr run` submits this repository as the app (`[tool.flwr.app]` in
  `pyproject.toml`)

Clients send back model weights and scalar metrics only. Each run writes
`<out>/<name>/model.pt` (the global model), `rounds.json` (client metrics per
round), `processes.json` (process IDs and each node's data path), and `logs/`.
Flower state stays under `<out>/<name>/.flwr` (`FLWR_HOME`), not `~/.flwr`.

Processes are separated by what each is told, not by operating-system
permissions: they run as the same user. Every task starts a new Python process
(about 4 s of imports), so a round takes roughly 30 s for 8 rovers regardless
of model size. `--evaluate-every` skips client evaluation on most rounds to
save that cost. Each run uses ports `--port-base` to `--port-base + 2 + rovers`.

### Evaluation

`ramms-fleet-eval` scores every model under the results directory on each
rover's test split: `local/rover_NN.pt` on its own rover (and, for reference,
on the others), and every other `<name>/model.pt` on all rovers. The metric is
AUPRC (area under the precision-recall curve), since collisions are 5 to 20% of
samples; AUROC and log loss are also recorded. `--finetune-epochs` adds a
personalized variant of each federated model, fine-tuned on one rover's own
training data (`<name>+ft`).

Evaluation reads every rover's test data, which a real deployment could not;
it is the experimenter's view, outside the federated protocol.

### RAMMS backend

`--backend ramms` on `ramms-fleet-collect` simulates the fleet inside a running
RAMMS editor instead of standalone MuJoCo, with the same policy, noise, and
dataset format:

```bash
.venv/bin/ramms-fleet-collect --backend ramms --rovers 8 --seconds 600 --out data/ramms/run0
```

`RammsFleet` (`ramms/fleet.py`) talks to URLab's RPC bridge (`ramms/bridge.py`,
ZMQ and msgpack on port 5559, `--ramms-address`). For each run it:

1. exports each rover's arena as standalone MJCF (`build_cell_xml`: rover.xml
   plus walls, obstacles, and a finite floor box) to `results/ramms-scene/`
2. loads or creates the `/Game/Levels/FleetArena` level, adds a MuJoCo manager,
   replaces any earlier fleet actors, imports each arena, spawns it 0.5 m apart,
   and saves the level
3. starts Play In Editor, switches URLab to direct (client-clocked) stepping at a
   5 ms timestep, and claims control of every rover
4. resets with `reset` plus `set_qpos` per rover, and steps all rovers with one
   `step` request per control step

What to know:

- The RAMMS editor must be running with the URLab bridge enabled. Imported
  arenas land in the RAMMS checkout under `Content/MuJoCoImports/` and the level
  in `Content/Levels/FleetArena.umap`; neither is committed there.
- Rangefinders match standalone MuJoCo exactly for the same pose; driving
  diverges by about 1 cm and 0.04 rad over 2 s, since RAMMS embeds MuJoCo 3.11.
- 8 rovers collect at about 3x real time, limited by the editor.
- URLab's `reset` cannot set free-joint poses, so poses go through `set_qpos`,
  which does not zero velocities: a single tipped rover is reset with its last
  velocity.
- URLab binds its bridge and camera ports on all network interfaces.
- `--camera` also records each rover's front camera, rendered in sync with the
  step (64 x 48 grayscale, `images` and `image_age` arrays). This needs lights
  in the level (the backend adds two), the editor's "Use Less CPU when in
  Background" preference off, and a CPU not saturated by training; collection
  then runs at about 0.4x real time.
- `RAMMS_BRIDGE=tcp://127.0.0.1:5559 pytest tests/test_ramms_backend.py` runs the
  integration tests against a live editor; they are skipped otherwise.

### Videos

```bash
.venv/bin/ramms-fleet-render overview --data data/run0 --model FedAvg=results/run0/federated/model.pt \
    --seconds 30 --out videos/fleet.mp4
.venv/bin/ramms-fleet-render compare --data data/run0 --rover 3 \
    --models "Local only=results/run0/local/rover_{rover}.pt" "FedAvg=results/run0/federated/model.pt" \
    --out videos/rover3.mp4
```

Both replay a collected run's rovers (same clutter, obstacles, speeds, and
noise) and color each rover from blue to red by a model's predicted collision
probability, using the rover's own noisy observations. Rangefinder rays are
drawn from the true geometry.

## Commands

| Command | Purpose | Key options |
|---|---|---|
| `ramms-fleet-collect` | simulate a fleet and write per-rover datasets | `--rovers`, `--seconds`, `--clutter-min/max`, `--speed`, `--range/accel/gyro-noise`, `--pedestrians`, `--avoid-gain`, `--seed`, `--backend mujoco\|ramms`, `--camera` |
| `ramms-fleet-view` | watch the fleet in the MuJoCo viewer | `--rovers`, `--clutter-min/max`, `--pedestrians`, `--speed` (playback rate, not rover speed) |
| `ramms-fleet-baseline` | train local-only and centralized models | `--epochs`, `--label`, `--horizon-m`, `--inputs`, `--seed` |
| `ramms-fleet-federate` | run Flower with a process per rover | `--rounds`, `--local-epochs`, `--proximal-mu`, `--evaluate-every`, `--name`, `--label`, `--inputs`, `--port-base` |
| `ramms-fleet-eval` | score every model on every rover | `--finetune-epochs`, `--label`, `--horizon-m` |
| `ramms-fleet-sweep` | repeat collect, baselines, federated runs, and eval over seeds | `--seeds`, `--jobs`, `--proximal-mus` (none for FedAvg only), profile, pedestrian, and label options, `--summarize-only` |
| `ramms-fleet-compare` | markdown tables across finished sweeps | `NAME=RESULTS_DIR ...`, `--out` |
| `ramms-fleet-render` | overview and side-by-side videos | `overview --model`, `compare --rover --models` |

Every command has `--help`. Data and results default to `data/` and `results/`,
which git ignores.

## Experiments

Scripts that reproduce the recorded experiments:

| Script | What it runs | Time on an 8-thread CPU |
|---|---|---|
| `scripts/overnight_heterogeneity.sh` | clutter, speed, noise, combined, and 16-rover conditions; videos | about 4 h |
| `scripts/distance_labels.sh` | the same datasets relabeled by distance | about 100 min |
| `scripts/ramms_federated.sh` | collection in RAMMS, federated training, cross-simulator scoring | about 1 h |
| `scripts/ramms_camera.sh` | camera collection in RAMMS, features vs camera vs both | about 4 h |
| `scripts/crowds.sh` | pedestrians: RAMMS camera runs (features, camera, both) and a MuJoCo sweep | about 6 h |

The earlier sweeps are single `ramms-fleet-sweep` commands, listed with their
results in [docs/results.md](docs/results.md).

## Layout

```
src/ramms_fleet/
  assets/rover.xml   rover MJCF
  spec.py            rover, policy, profile, and feature constants (no MuJoCo import)
  world.py           arena cells, obstacles, pedestrian bodies, and rover placement
  crowd.py           pedestrian walking model (no MuJoCo import)
  fleet.py           MujocoFleet: stepping, sensor noise, observations
  policy.py          exploration policy
  collect.py         ramms-fleet-collect
  labels.py          time and distance labels
  learning.py        datasets, model, training, metrics
  experiment.py      ramms-fleet-baseline and ramms-fleet-eval
  fl/                Flower ServerApp and ClientApp
  federate.py        ramms-fleet-federate
  sweep.py           ramms-fleet-sweep and ramms-fleet-compare
  view.py            ramms-fleet-view
  render.py          ramms-fleet-render
  ramms/             RAMMS backend: URLab bridge client and RammsFleet
scripts/             experiment scripts
docs/results.md      every experiment and its results
tests/
```

## License

MIT
