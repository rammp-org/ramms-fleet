"""Collects per-rover collision-prediction datasets from the MuJoCo fleet.

Every rover writes its own file, the way a federated client would keep its data
local. Labels are self-supervised: a sample is positive when the rover's bumper
registers a new contact within the next `horizon` seconds.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from ramms_fleet.fleet import MujocoFleet
from ramms_fleet.policy import CRUISE, WanderPolicy
from ramms_fleet.spec import FEATURES, RANGE_SENSORS, RoverParams, WanderParams
from ramms_fleet.world import EnvConfig

__all__ = ["FEATURES", "collect", "collision_labels", "spread_configs"]


def collision_labels(bump: np.ndarray, episode: np.ndarray, horizon_steps: int) -> np.ndarray:
    """Marks step t positive when a bump onset occurs in (t, t + horizon_steps].

    Onsets are rising edges of `bump` inside one episode; labels never look
    across an episode boundary.
    """
    steps = len(bump)
    previous = np.concatenate([[False], bump[:-1]])
    same_episode = np.concatenate([[False], episode[1:] == episode[:-1]])
    onset = bump & ~(previous & same_episode)
    labels = np.zeros(steps, dtype=bool)
    for t in np.flatnonzero(onset):
        start = max(0, t - horizon_steps)
        window = slice(start, t)
        labels[window] |= episode[window] == episode[t]
    return labels


def collect(
    configs: list[EnvConfig],
    seconds: float,
    out_dir: Path,
    control_hz: float = 20.0,
    horizon: float = 0.5,
    seed: int = 0,
    wander: WanderParams = WanderParams(),
) -> dict:
    fleet = MujocoFleet(configs, control_hz=control_hz, seed=seed)
    policy = WanderPolicy(fleet.num_rovers, fleet.dt, wander, seed=seed + 1)
    steps = round(seconds / fleet.dt)
    n = fleet.num_rovers

    features = np.zeros((n, steps, len(FEATURES)), dtype=np.float32)
    bump = np.zeros((n, steps), dtype=bool)
    valid = np.zeros((n, steps), dtype=bool)
    episode = np.zeros((n, steps), dtype=np.int32)
    pose = np.zeros((n, steps, 3), dtype=np.float32)
    current_episode = np.zeros(n, dtype=np.int32)

    obs = fleet.reset()
    policy.reset(np.arange(n), obs)
    started = time.perf_counter()
    for t in range(steps):
        commands = policy.act(obs)
        features[:, t] = np.concatenate([obs.ranges, obs.accel, obs.gyro, obs.wheel_vel, commands], axis=1)
        bump[:, t] = obs.bump
        # Only cruising samples without contact are useful for prediction; the
        # recovery manoeuvre is scripted and always starts in contact.
        valid[:, t] = (policy.mode == CRUISE) & ~obs.bump
        episode[:, t] = current_episode
        pose[:, t] = obs.pose

        obs = fleet.step(commands)
        tipped = np.flatnonzero(~obs.upright)
        if len(tipped):
            current_episode[tipped] += 1
            obs = fleet.reset(list(tipped))
            policy.reset(tipped, obs)
    elapsed = time.perf_counter() - started

    horizon_steps = max(1, round(horizon / fleet.dt))
    out_dir.mkdir(parents=True, exist_ok=True)
    rovers = []
    for i in range(n):
        labels = collision_labels(bump[i], episode[i], horizon_steps)
        np.savez_compressed(
            out_dir / f"rover_{i:02d}.npz",
            features=features[i],
            label=labels,
            valid=valid[i],
            bump=bump[i],
            episode=episode[i],
            pose=pose[i],
        )
        onsets = int((bump[i][1:] & ~bump[i][:-1]).sum() + bump[i][0])
        rovers.append(
            {
                "rover": i,
                "clutter": configs[i].clutter,
                "env_seed": configs[i].seed,
                "collisions_per_min": onsets / (seconds / 60),
                "positive_rate": float(labels[valid[i]].mean()) if valid[i].any() else 0.0,
                "valid_samples": int(valid[i].sum()),
                "mean_front_range": float(features[i, :, RANGE_SENSORS.index("range_0")].mean()),
                "episodes": int(episode[i].max()) + 1,
            }
        )

    meta = {
        "backend": "mujoco",
        "seconds": seconds,
        "control_hz": control_hz,
        "dt": fleet.dt,
        "horizon": horizon,
        "seed": seed,
        "features": list(FEATURES),
        "rover_params": asdict(RoverParams()),
        "wander_params": asdict(wander),
        "sim_seconds_per_wall_second": seconds / elapsed,
        "rovers": rovers,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def spread_configs(rovers: int, clutter_min: float, clutter_max: float, seed: int) -> list[EnvConfig]:
    clutter = np.linspace(clutter_min, clutter_max, rovers) if rovers > 1 else [clutter_min]
    return [EnvConfig(clutter=float(c), seed=seed * 1000 + i) for i, c in enumerate(clutter)]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rovers", type=int, default=8)
    parser.add_argument("--seconds", type=float, default=600.0, help="simulated seconds per rover")
    parser.add_argument("--clutter-min", type=float, default=0.1, help="obstacles per m^2 for rover 0")
    parser.add_argument("--clutter-max", type=float, default=1.2, help="obstacles per m^2 for the last rover")
    parser.add_argument("--avoid-gain", type=float, default=WanderParams.avoid_gain)
    parser.add_argument("--horizon", type=float, default=0.5, help="collision look-ahead in seconds")
    parser.add_argument("--control-hz", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    configs = spread_configs(args.rovers, args.clutter_min, args.clutter_max, args.seed)
    meta = collect(
        configs,
        seconds=args.seconds,
        out_dir=args.out,
        control_hz=args.control_hz,
        horizon=args.horizon,
        seed=args.seed,
        wander=WanderParams(avoid_gain=args.avoid_gain),
    )

    print(f"wrote {args.out} ({meta['sim_seconds_per_wall_second']:.0f}x real time)")
    print(f"{'rover':>5} {'clutter':>7} {'coll/min':>8} {'pos rate':>8} {'samples':>8} {'front m':>7} {'episodes':>8}")
    for r in meta["rovers"]:
        print(
            f"{r['rover']:>5} {r['clutter']:>7.2f} {r['collisions_per_min']:>8.1f} {r['positive_rate']:>8.3f}"
            f" {r['valid_samples']:>8} {r['mean_front_range']:>7.2f} {r['episodes']:>8}"
        )


if __name__ == "__main__":
    main()
