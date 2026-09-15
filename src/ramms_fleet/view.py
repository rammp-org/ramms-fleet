"""Watches the fleet wander in the MuJoCo viewer.

All rovers share one world so they appear side by side. A rover's body turns
red while its bumper is in contact.
"""

from __future__ import annotations

import argparse
import time

import mujoco
import mujoco.viewer
import numpy as np

from ramms_fleet.collect import spread_configs
from ramms_fleet.fleet import MujocoFleet
from ramms_fleet.policy import WanderParams, WanderPolicy
from ramms_fleet.world import ROVER_PREFIX

BODY_RGBA = np.array([0.20, 0.45, 0.80, 1.0])
BUMP_RGBA = np.array([0.90, 0.15, 0.15, 1.0])


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rovers", type=int, default=8)
    parser.add_argument("--clutter-min", type=float, default=0.1)
    parser.add_argument("--clutter-max", type=float, default=1.2)
    parser.add_argument("--avoid-gain", type=float, default=WanderParams.avoid_gain)
    parser.add_argument("--speed", type=float, default=1.0, help="playback speed relative to real time")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pedestrians", type=int, nargs=2, default=(0, 0), metavar=("MIN", "MAX"))
    args = parser.parse_args(argv)

    configs = spread_configs(args.rovers, args.clutter_min, args.clutter_max, args.seed, tuple(args.pedestrians))
    fleet = MujocoFleet(configs, seed=args.seed, shared_world=True)
    model, data = fleet.worlds[0]
    policy = WanderPolicy(fleet.num_rovers, fleet.dt, WanderParams(avoid_gain=args.avoid_gain), seed=args.seed + 1)
    body_geoms = [model.geom(ROVER_PREFIX.format(index=i) + "body").id for i in range(fleet.num_rovers)]

    obs = fleet.reset()
    policy.reset(np.arange(fleet.num_rovers), obs)
    centers = np.array([r.cell.center for r in fleet.rovers])

    with mujoco.viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=False) as viewer:
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_RANGEFINDER] = True
        # Frame the whole grid of cells from above.
        extent = np.ptp(centers, axis=0).max() + fleet.layout.cell_size
        viewer.cam.lookat[:2] = centers.min(axis=0) + np.ptp(centers, axis=0) / 2
        viewer.cam.distance = 0.7 * extent / np.tan(np.deg2rad(model.vis.global_.fovy / 2))
        viewer.cam.elevation = -75
        viewer.cam.azimuth = 90

        while viewer.is_running():
            started = time.perf_counter()
            with viewer.lock():
                obs = fleet.step(policy.act(obs))
                tipped = np.flatnonzero(~obs.upright)
                if len(tipped):
                    obs = fleet.reset(list(tipped))
                    policy.reset(tipped, obs)
                for geom, bumping in zip(body_geoms, obs.bump, strict=True):
                    model.geom_rgba[geom] = BUMP_RGBA if bumping else BODY_RGBA
            viewer.sync()
            time.sleep(max(0.0, fleet.dt / args.speed - (time.perf_counter() - started)))


if __name__ == "__main__":
    main()
