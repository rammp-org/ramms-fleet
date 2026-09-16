"""Closed-loop test: let the collision predictor drive, and count collisions.

Every other experiment scores a model that never touches the robot. Here the
model runs on board: each control step the rover forms the command it would
have used, scores it with its own model, and when the predicted risk crosses a
threshold it brakes and turns towards the freer side. Arenas, profiles, and the
exploration policy are exactly those of collection, so the only change is the
guard.

Safety alone is easy (stand still), so runs report collisions per minute
against distance covered, and collisions per 100 m, which is the number a fleet
operator would care about.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from ramms_fleet.collect import spread_configs, spread_profiles
from ramms_fleet.experiment import RiskModel
from ramms_fleet.fleet import MujocoFleet
from ramms_fleet.policy import CRUISE, WanderPolicy
from ramms_fleet.spec import RANGE_ANGLES_DEG, WanderParams
from ramms_fleet.world import EnvConfig


@dataclass(frozen=True)
class GuardParams:
    threshold: float = 0.5
    """Predicted collision probability above which the guard takes over."""
    brake: float = 0.25
    """Fraction of the commanded speed kept while guarding."""
    turn: float = 0.8
    """Guard turn rate as a fraction of the policy's maximum."""


class GuardedPolicy:
    """The exploration policy, with a model-driven brake and turn on top.

    The scored feature vector holds the command the policy intended, exactly as
    during collection; the guard then overrides that command.
    """

    def __init__(
        self,
        policy: WanderPolicy,
        risk: RiskModel | None,
        params: GuardParams = GuardParams(),
        wander: WanderParams = WanderParams(),
    ):
        self.policy = policy
        self.risk = risk
        self.params = params
        self.wander = wander
        self._sin = np.sin(np.deg2rad(RANGE_ANGLES_DEG))
        self.last_risk = np.zeros(len(policy.mode))
        self.guarded = np.zeros(len(policy.mode), dtype=bool)

    def reset(self, rovers, obs) -> None:
        self.policy.reset(rovers, obs)
        if self.risk is not None:
            self.risk.reset(np.asarray(rovers))

    def act(self, obs) -> np.ndarray:
        commands = self.policy.act(obs)
        if self.risk is None:
            return commands
        features = np.concatenate([obs.ranges, obs.accel, obs.gyro, obs.wheel_vel, commands], axis=1)
        self.last_risk = self.risk(features.astype(np.float32))
        # Only cruising rovers are guarded: reversing and spinning are the
        # scripted recovery, which is already a response to a collision.
        self.guarded = (self.last_risk > self.params.threshold) & (self.policy.mode == CRUISE)
        if self.guarded.any():
            # Turn away from the nearer side, the direction the weak avoidance term already prefers.
            closeness = np.clip(1 - obs.ranges / self.wander.avoid_distance, 0, 1)
            side = (closeness * self._sin).sum(axis=1)
            away = np.where(side >= 0, -1.0, 1.0) * self.params.turn * self.wander.max_turn_rate
            commands[self.guarded, 0] *= self.params.brake
            commands[self.guarded, 1] = away[self.guarded]
        return commands


def run_guard(
    configs: list[EnvConfig],
    seconds: float,
    model: str | None,
    seed: int = 0,
    control_hz: float = 20.0,
    profiles=None,
    wander: WanderParams = WanderParams(),
    guard: GuardParams = GuardParams(),
) -> dict:
    """Drives the fleet with `model` guarding (or nothing, when it is None) and returns per-rover metrics.

    `model` may contain `{rover}`, which expands to each rover's own file.
    """
    fleet = MujocoFleet(configs, control_hz=control_hz, seed=seed, profiles=profiles)
    n = fleet.num_rovers
    speeds = np.array([p.cruise_speed for p in fleet.profiles])
    policy = WanderPolicy(n, fleet.dt, wander, seed=seed + 1, cruise_speeds=speeds)
    risk = None
    if model is not None:
        paths = [Path(model.format(rover=f"{i:02d}")) for i in range(n)] if "{rover}" in model else [Path(model)] * n
        risk = _PerRoverRisk(paths, n)
    guarded_policy = GuardedPolicy(policy, risk, guard, wander)

    steps = round(seconds / fleet.dt)
    obs = fleet.reset()
    guarded_policy.reset(np.arange(n), obs)
    collisions = np.zeros(n, dtype=int)
    distance = np.zeros(n)
    guarded_steps = np.zeros(n, dtype=int)
    tips = np.zeros(n, dtype=int)
    previous_bump = obs.bump.copy()
    previous_pose = obs.pose[:, :2].copy()
    for _ in range(steps):
        commands = guarded_policy.act(obs)
        guarded_steps += guarded_policy.guarded
        obs = fleet.step(commands)
        collisions += obs.bump & ~previous_bump
        previous_bump = obs.bump.copy()
        # Resets teleport the rover, so only count movement between ordinary steps.
        moved = np.linalg.norm(obs.pose[:, :2] - previous_pose, axis=1)
        distance += np.where(moved < 0.5, moved, 0.0)
        previous_pose = obs.pose[:, :2].copy()
        tipped = np.flatnonzero(~obs.upright)
        if len(tipped):
            tips[tipped] += 1
            obs = fleet.reset(list(tipped))
            guarded_policy.reset(tipped, obs)
            previous_pose = obs.pose[:, :2].copy()
            previous_bump = obs.bump.copy()
    minutes = seconds / 60

    rovers = []
    for i in range(n):
        rovers.append(
            {
                "rover": i,
                "clutter": configs[i].clutter,
                "pedestrians": configs[i].pedestrians,
                "cruise_speed": fleet.profiles[i].cruise_speed,
                "collisions_per_min": float(collisions[i]) / minutes,
                "metres_per_min": float(distance[i]) / minutes,
                "collisions_per_100m": 100 * float(collisions[i]) / float(distance[i]) if distance[i] > 0 else None,
                "guarded_fraction": float(guarded_steps[i]) / steps,
                "tips": int(tips[i]),
            }
        )
    fleet_totals = {
        "collisions_per_min": float(np.mean([r["collisions_per_min"] for r in rovers])),
        "metres_per_min": float(np.mean([r["metres_per_min"] for r in rovers])),
        "collisions_per_100m": 100 * float(collisions.sum()) / float(distance.sum()),
        "guarded_fraction": float(guarded_steps.sum()) / (steps * n),
    }
    return {
        "model": model,
        "seed": seed,
        "seconds": seconds,
        "guard": asdict(guard),
        "fleet": fleet_totals,
        "rovers": rovers,
    }


class _PerRoverRisk:
    """One RiskModel per rover, so local-only models can drive their own rover."""

    def __init__(self, paths: list[Path], num_rovers: int):
        self.models = [RiskModel(p, 1) for p in paths]
        self.num_rovers = num_rovers

    def reset(self, rovers: np.ndarray) -> None:
        for i in np.atleast_1d(rovers):
            self.models[int(i)].reset(np.array([0]))

    def __call__(self, features: np.ndarray) -> np.ndarray:
        return np.array([m(features[i : i + 1])[0] for i, m in enumerate(self.models)])


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rovers", type=int, default=8)
    parser.add_argument("--seconds", type=float, default=600.0)
    parser.add_argument("--clutter-min", type=float, default=0.1)
    parser.add_argument("--clutter-max", type=float, default=1.2)
    parser.add_argument("--pedestrians", type=int, nargs=2, default=(0, 0), metavar=("MIN", "MAX"))
    parser.add_argument("--speed", type=float, nargs=2, default=(0.3, 0.3), metavar=("MIN", "MAX"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--model", default=None, help="model.pt driving every rover, or a path with {rover} for per-rover models"
    )
    parser.add_argument("--threshold", type=float, default=GuardParams.threshold)
    parser.add_argument("--brake", type=float, default=GuardParams.brake)
    parser.add_argument("--turn", type=float, default=GuardParams.turn)
    parser.add_argument("--out", type=Path, required=True, help="JSON file for the metrics")
    args = parser.parse_args(argv)

    configs = spread_configs(args.rovers, args.clutter_min, args.clutter_max, args.seed, tuple(args.pedestrians))
    profiles = spread_profiles(args.rovers, tuple(args.speed), seed=args.seed)
    report = run_guard(
        configs,
        args.seconds,
        args.model,
        seed=args.seed,
        profiles=profiles,
        guard=GuardParams(threshold=args.threshold, brake=args.brake, turn=args.turn),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    f = report["fleet"]
    print(
        f"{args.model or 'no guard'}: {f['collisions_per_min']:.1f} collisions/min, "
        f"{f['metres_per_min']:.1f} m/min, {f['collisions_per_100m']:.1f} per 100 m, "
        f"guarding {f['guarded_fraction'] * 100:.0f}% of steps"
    )


def summary_main(argv: list[str] | None = None) -> None:
    """Averages guard runs written as <dir>/<condition>/<method>/seed_<n>.json."""
    parser = argparse.ArgumentParser(description=summary_main.__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--out", type=Path, help="also write the markdown here")
    args = parser.parse_args(argv)

    METRICS = ("collisions_per_min", "collisions_per_100m", "metres_per_min", "guarded_fraction")
    summary: dict[str, dict[str, dict]] = {}
    for path in sorted(args.results.glob("*/*/seed_*.json")):
        condition, method = path.parent.parent.name, path.parent.name
        report = json.loads(path.read_text())
        runs = summary.setdefault(condition, {}).setdefault(method, {"seeds": [], **{m: [] for m in METRICS}})
        runs["seeds"].append(report["seed"])
        for m in METRICS:
            runs[m].append(report["fleet"][m])
    stats = {
        c: {
            k: {
                "seeds": len(v["seeds"]),
                **{
                    m: {"mean": float(np.mean(v[m])), "std": float(np.std(v[m], ddof=1)) if len(v[m]) > 1 else None}
                    for m in METRICS
                },
            }
            for k, v in methods.items()
        }
        for c, methods in summary.items()
    }
    (args.results / "summary.json").write_text(json.dumps(stats, indent=2))

    lines = []
    for condition, methods in stats.items():
        lines += [
            f"## {condition}",
            "",
            "| Guard | Collisions/min | Collisions/100 m | Metres/min | Guarding |",
            "|---|---:|---:|---:|---:|",
        ]
        for name, v in methods.items():

            def cell(metric: str, stats: dict = v) -> str:
                spread = stats[metric]["std"]
                return f"{stats[metric]['mean']:.1f}" + (f" ± {spread:.1f}" if spread is not None else "")

            guarding = f"{v['guarded_fraction']['mean'] * 100:.0f}%"
            row = [name, cell("collisions_per_min"), cell("collisions_per_100m"), cell("metres_per_min"), guarding]
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")
    markdown = "\n".join(lines)
    if args.out:
        args.out.write_text(markdown + "\n")
    print(markdown)


if __name__ == "__main__":
    main()
