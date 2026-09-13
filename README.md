# ramms-fleet

Federated learning experiments on fleets of simulated rovers for
[RAMMS](https://github.com/rammp-org/ramms-sim).

Assistive robots work in people's homes, and their sensor data is private. The
question here is how well a fleet can learn together when each robot keeps its
data local and only shares model updates, especially when every robot sees a
different environment. Small differential-drive rovers stand in for the
robots.

## Status and roadmap

1. **Rover model and MuJoCo fleet runner** (this step): an MJCF rover, one
   walled arena per rover with its own clutter level, an exploration policy, and
   per-rover collision-prediction datasets.
2. **Federated training**: one [Flower](https://flower.ai) client per rover
   training on its local dataset, FedAvg on the server, compared against
   local-only and centralized training.
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
uv venv && uv pip install -e ".[dev]"
.venv/bin/ramms-fleet-collect --rovers 8 --seconds 600 --out data/run0
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
