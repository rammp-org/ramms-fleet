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
from ramms_fleet.labels import collision_labels
from ramms_fleet.policy import CRUISE, WanderPolicy
from ramms_fleet.spec import FEATURES, RANGE_SENSORS, RoverParams, RoverProfile, WanderParams
from ramms_fleet.world import EnvConfig

__all__ = ["FEATURES", "collect", "collision_labels", "spread_configs", "spread_profiles"]


def collect(
    configs: list[EnvConfig],
    seconds: float,
    out_dir: Path,
    control_hz: float = 20.0,
    horizon: float = 0.5,
    seed: int = 0,
    wander: WanderParams = WanderParams(),
    profiles: list[RoverProfile] | None = None,
    backend: str = "mujoco",
    ramms_address: str = "tcp://127.0.0.1:5559",
    camera: bool = False,
) -> dict:
    if backend == "ramms":
        # Imported lazily: the RAMMS backend needs pyzmq and msgpack (the [ramms] extra).
        from ramms_fleet.ramms.fleet import RammsFleet

        fleet = RammsFleet(
            configs, control_hz=control_hz, seed=seed, profiles=profiles, address=ramms_address, camera=camera
        )
    elif backend == "mujoco":
        if camera:
            raise ValueError("--camera needs --backend ramms")
        fleet = MujocoFleet(configs, control_hz=control_hz, seed=seed, profiles=profiles)
    else:
        raise ValueError(f"unknown backend {backend!r}")
    speeds = np.array([p.cruise_speed for p in fleet.profiles])
    policy = WanderPolicy(fleet.num_rovers, fleet.dt, wander, seed=seed + 1, cruise_speeds=speeds)
    steps = round(seconds / fleet.dt)
    n = fleet.num_rovers

    features = np.zeros((n, steps, len(FEATURES)), dtype=np.float32)
    bump = np.zeros((n, steps), dtype=bool)
    valid = np.zeros((n, steps), dtype=bool)
    episode = np.zeros((n, steps), dtype=np.int32)
    pose = np.zeros((n, steps, 3), dtype=np.float32)
    current_episode = np.zeros(n, dtype=np.int32)

    images = image_age = None
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
        if obs.images is not None:
            if images is None:
                images = np.zeros((n, steps, *obs.images.shape[1:]), dtype=np.uint8)
                image_age = np.full((n, steps), np.inf, dtype=np.float32)
            images[:, t] = obs.images
            image_age[:, t] = obs.image_age

        obs = fleet.step(commands)
        tipped = np.flatnonzero(~obs.upright)
        if len(tipped):
            current_episode[tipped] += 1
            obs = fleet.reset(list(tipped))
            policy.reset(tipped, obs)
    elapsed = time.perf_counter() - started
    if hasattr(fleet, "close"):
        fleet.close()

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
            **({} if images is None else {"images": images[i], "image_age": image_age[i]}),
        )
        onsets = int((bump[i][1:] & ~bump[i][:-1]).sum() + bump[i][0])
        rovers.append(
            {
                "rover": i,
                "clutter": configs[i].clutter,
                "env_seed": configs[i].seed,
                **asdict(fleet.profiles[i]),
                "collisions_per_min": onsets / (seconds / 60),
                "positive_rate": float(labels[valid[i]].mean()) if valid[i].any() else 0.0,
                "valid_samples": int(valid[i].sum()),
                "mean_front_range": float(features[i, :, RANGE_SENSORS.index("range_0")].mean()),
                "episodes": int(episode[i].max()) + 1,
            }
        )

    meta = {
        "backend": backend,
        "camera": None if images is None else {"name": "front", "height": images.shape[2], "width": images.shape[3]},
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


def spread_profiles(
    rovers: int,
    speed: tuple[float, float] = (RoverProfile.cruise_speed, RoverProfile.cruise_speed),
    range_noise: tuple[float, float] = (0.0, 0.0),
    accel_noise: tuple[float, float] = (0.0, 0.0),
    gyro_noise: tuple[float, float] = (0.0, 0.0),
    seed: int = 0,
) -> list[RoverProfile]:
    """Spreads speed and noise evenly over the fleet in seeded random orders.

    Speed and noise get independent shuffles so neither lines up with clutter
    (which rises with rover index) or with each other. One noise order covers
    all sensors: a noisy rover is noisy on every sensor.
    """
    rng = np.random.default_rng([seed, 2])
    speed_order, noise_order = rng.permutation(rovers), rng.permutation(rovers)

    def spread(bounds: tuple[float, float], order: np.ndarray) -> np.ndarray:
        values = np.linspace(bounds[0], bounds[1], rovers) if rovers > 1 else np.array([bounds[0]])
        return values[order]

    speeds = spread(speed, speed_order)
    ranges, accels, gyros = (spread(b, noise_order) for b in (range_noise, accel_noise, gyro_noise))
    return [
        RoverProfile(cruise_speed=float(s), range_noise=float(r), accel_noise=float(a), gyro_noise=float(g))
        for s, r, a, g in zip(speeds, ranges, accels, gyros, strict=True)
    ]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rovers", type=int, default=8)
    parser.add_argument("--seconds", type=float, default=600.0, help="simulated seconds per rover")
    parser.add_argument("--clutter-min", type=float, default=0.1, help="obstacles per m^2 for rover 0")
    parser.add_argument("--clutter-max", type=float, default=1.2, help="obstacles per m^2 for the last rover")
    parser.add_argument("--avoid-gain", type=float, default=WanderParams.avoid_gain)
    parser.add_argument("--speed", type=float, nargs=2, default=(0.3, 0.3), metavar=("MIN", "MAX"), help="m/s")
    parser.add_argument("--range-noise", type=float, nargs=2, default=(0.0, 0.0), metavar=("MIN", "MAX"), help="m")
    parser.add_argument("--accel-noise", type=float, nargs=2, default=(0.0, 0.0), metavar=("MIN", "MAX"), help="m/s^2")
    parser.add_argument("--gyro-noise", type=float, nargs=2, default=(0.0, 0.0), metavar=("MIN", "MAX"), help="rad/s")
    parser.add_argument("--horizon", type=float, default=0.5, help="collision look-ahead in seconds")
    parser.add_argument("--control-hz", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--backend", choices=("mujoco", "ramms"), default="mujoco", help="ramms needs the editor running"
    )
    parser.add_argument("--ramms-address", default="tcp://127.0.0.1:5559", help="URLab bridge RPC address")
    parser.add_argument("--camera", action="store_true", help="also record front-camera frames (ramms backend)")
    args = parser.parse_args(argv)

    configs = spread_configs(args.rovers, args.clutter_min, args.clutter_max, args.seed)
    profiles = spread_profiles(
        args.rovers,
        tuple(args.speed),
        tuple(args.range_noise),
        tuple(args.accel_noise),
        tuple(args.gyro_noise),
        args.seed,
    )
    meta = collect(
        configs,
        seconds=args.seconds,
        out_dir=args.out,
        control_hz=args.control_hz,
        horizon=args.horizon,
        seed=args.seed,
        wander=WanderParams(avoid_gain=args.avoid_gain),
        profiles=profiles,
        backend=args.backend,
        ramms_address=args.ramms_address,
        camera=args.camera,
    )

    print(f"wrote {args.out} ({meta['sim_seconds_per_wall_second']:.0f}x real time)")
    print(
        f"{'rover':>5} {'clutter':>7} {'speed':>5} {'rng noise':>9} {'coll/min':>8} {'pos rate':>8}"
        f" {'samples':>8} {'episodes':>8}"
    )
    for r in meta["rovers"]:
        print(
            f"{r['rover']:>5} {r['clutter']:>7.2f} {r['cruise_speed']:>5.2f} {r['range_noise']:>9.3f}"
            f" {r['collisions_per_min']:>8.1f} {r['positive_rate']:>8.3f} {r['valid_samples']:>8} {r['episodes']:>8}"
        )


if __name__ == "__main__":
    main()
