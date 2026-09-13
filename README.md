# ramms-fleet

Federated learning experiments on fleets of simulated rovers for
[RAMMS](https://github.com/rammp-org/ramms-sim).

Assistive robots work in people's homes, and their sensor data is private. The
question here is how well a fleet can learn together when each robot keeps its
data local and only shares model updates, especially when every robot sees a
different environment. Small differential-drive rovers stand in for the
robots.

## Status and roadmap

1. **Rover model and MuJoCo fleet runner** (done): an MJCF rover, one
   walled arena per rover with its own clutter level, an exploration policy, and
   per-rover collision-prediction datasets.
2. **Federated training** (done): one [Flower](https://flower.ai) client
   process per rover training on its local dataset, FedAvg on the server,
   compared against local-only and centralized training.
3. **RAMMS backend**: load the same rover into RAMMS through URLab
   (`scene.spawn_grid`) and drive it over the URLab RPC bridge, adding cameras
   and crowds.
4. **Heterogeneity experiments**: how environment differences between rovers
   affect the shared model, and personalized federated variants.

## The rover

`src/ramms_fleet/assets/rover.xml` is plain MJCF so it can load in both MuJoCo
and RAMMS:

- two velocity-controlled wheels and a frictionless rear caster
- five rangefinders fanned from -60 to +60 degrees
- IMU (accelerometer and gyro) and wheel velocity sensors
- a touch sensor around the chassis used as the bumper
- a forward camera (not used by the MuJoCo backend yet)

Commands are body velocities: linear m/s and angular rad/s.

## Quickstart

```bash
uv venv
uv pip install torch --index-url https://download.pytorch.org/whl/cpu   # optional: skip the CUDA wheels
uv pip install -e ".[dev]"

.venv/bin/ramms-fleet-collect --rovers 8 --seconds 600 --out data/run0
.venv/bin/ramms-fleet-view --rovers 8                                   # watch them in the MuJoCo viewer

.venv/bin/ramms-fleet-baseline --data data/run0 --out results/run0 --epochs 20
.venv/bin/ramms-fleet-federate --data data/run0 --out results/run0 --rounds 20
.venv/bin/ramms-fleet-eval --data data/run0 --results results/run0
.venv/bin/pytest
```

`ramms-fleet-collect` prints per-rover statistics. Rover *i* gets clutter
spread linearly from `--clutter-min` to `--clutter-max` obstacles per square
metre, so each rover's data comes from a different distribution. 8 rovers run
at roughly 15x real time on one CPU core.

## Dataset format

Each run directory has `meta.json` and one `rover_NN.npz` per rover, as a
federated client would keep it:

| Array | Shape | Meaning |
|-------|-------|---------|
| `features` | (T, 17) | ranges, accel, gyro, wheel velocities, commanded v and w (names in `meta.json`) |
| `label` | (T,) | a new bumper contact happens within the next `horizon` seconds |
| `valid` | (T,) | cruising and not in contact; train and evaluate on these samples |
| `bump` | (T,) | bumper in contact |
| `episode` | (T,) | increments when a tipped-over rover is reset |
| `pose` | (T, 3) | x, y in the rover's cell and yaw, for analysis only |

Labels are self-supervised from the bumper, so no manual labeling is needed.
The exploration policy avoids obstacles only weakly (`--avoid-gain`) so rovers
collide often enough to produce positives.

## License

MIT

## Federated training

`ramms-fleet-federate` runs the Flower Deployment Engine on one machine:

- a SuperLink, whose ServerApp runs FedAvg and never opens rover data
- one SuperNode per rover, started with `--node-config "data-path='.../rover_NN.npz'"`
  so it only knows its own file; every ClientApp task runs in a child process
- `flwr run` submits the app from this repo (`[tool.flwr.app]` in `pyproject.toml`)

Clients send back model weights and scalar metrics only. Process IDs and each
node's data path are recorded in `results/<run>/federated/processes.json`, and
all logs are in `federated/logs/`. Flower state stays under
`federated/.flwr` (`FLWR_HOME`), not `~/.flwr`. Every task starts a fresh
Python process, so a round takes about 30 s of mostly start-up time.

The model is a small MLP over the last 4 control steps of features. Features
are scaled by fixed physical limits rather than dataset statistics, so no
client needs another client's data to normalize. Every method starts from the
same initial weights and processes the same number of samples: 20 rounds of 1
local epoch for federated, 20 epochs for each baseline.

`ramms-fleet-eval` scores every model on each rover's held-out final 20% of
time. It reads all rovers' test data, which a real deployment could not do; it
is the experimenter's view.

### First result

8 rovers, 600 simulated seconds each, clutter 0.1 to 1.2, seed 0, one run.
Test AUPRC (positive rates 5 to 23%):

| Rover | Clutter | Local only | Federated | Centralized |
|------:|--------:|-----------:|----------:|------------:|
| 0 | 0.10 | 0.904 | 0.889 | 0.927 |
| 1 | 0.26 | 0.959 | 0.955 | 0.948 |
| 2 | 0.41 | 0.753 | 0.883 | 0.925 |
| 3 | 0.57 | 0.920 | 0.921 | 0.971 |
| 4 | 0.73 | 0.874 | 0.894 | 0.935 |
| 5 | 0.89 | 0.800 | 0.856 | 0.969 |
| 6 | 1.04 | 0.775 | 0.798 | 0.922 |
| 7 | 1.20 | 0.814 | 0.839 | 0.929 |
| **Mean** | | **0.850** | **0.880** | **0.941** |

Federated beats local-only on average, most in the cluttered arenas, but is
still below centralized. Its evaluation AUPRC was still rising at round 20
(0.858 at round 15, 0.884 at round 20), so more rounds should narrow the gap.
Single seed; treat the differences as indicative.
