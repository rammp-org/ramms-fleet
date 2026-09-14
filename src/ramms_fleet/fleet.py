"""Headless MuJoCo backend: N rovers stepped in lockstep.

By default each rover gets its own small model holding one arena cell. The
cells never interact, and independent models step faster than one combined
model because their contact problems are solved separately. It also mirrors
the federated setup, where every client owns its environment. `shared_world`
puts every cell in one model instead, which is what the viewer needs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import mujoco
import numpy as np

from ramms_fleet.spec import RANGE_ANGLES_DEG, RANGE_SENSORS, RoverParams, RoverProfile
from ramms_fleet.world import ROVER_PREFIX, ArenaLayout, Cell, EnvConfig, build_world

__all__ = ["RANGE_ANGLES_DEG", "RANGE_SENSORS", "FleetObs", "MujocoFleet", "RoverParams", "RoverProfile"]

_SENSOR_DISABLE_BIT = int(mujoco.mjtDisableBit.mjDSBL_SENSOR)


@dataclass
class FleetObs:
    """Batched observations, first axis is the rover index."""

    ranges: np.ndarray  # (N, 5) metres, ordered as RANGE_SENSORS
    accel: np.ndarray  # (N, 3) m/s^2, rover frame
    gyro: np.ndarray  # (N, 3) rad/s, rover frame
    wheel_vel: np.ndarray  # (N, 2) rad/s, left and right
    bump: np.ndarray  # (N,) bool
    pose: np.ndarray  # (N, 3) x, y relative to the cell centre, and yaw
    upright: np.ndarray  # (N,) bool, False once the rover has tipped over


class _RoverSim:
    """One rover's cell plus its addresses inside the model that holds it."""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, cell: Cell, index: int):
        self.model = model
        self.data = data
        self.cell = cell
        m = model
        p = ROVER_PREFIX.format(index=index)
        joint = m.joint(p + "root")
        self.qpos = joint.qposadr[0]
        self.dof = joint.dofadr[0]
        self.body = m.body(p + "chassis").id
        self.act = [m.actuator(p + "wheel_left").id, m.actuator(p + "wheel_right").id]
        self.range_adr = [m.sensor(p + s).adr[0] for s in RANGE_SENSORS]
        self.accel_adr = m.sensor(p + "accel").adr[0]
        self.gyro_adr = m.sensor(p + "gyro").adr[0]
        self.bump_adr = m.sensor(p + "bump").adr[0]
        self.wheel_adr = [m.sensor(p + "wheel_left_vel").adr[0], m.sensor(p + "wheel_right_vel").adr[0]]


class MujocoFleet:
    def __init__(
        self,
        configs: list[EnvConfig],
        control_hz: float = 20.0,
        seed: int = 0,
        layout: ArenaLayout = ArenaLayout(),
        params: RoverParams = RoverParams(),
        shared_world: bool = False,
        profiles: list[RoverProfile] | None = None,
    ):
        self.profiles = profiles or [RoverProfile()] * len(configs)
        if len(self.profiles) != len(configs):
            raise ValueError(f"{len(self.profiles)} profiles for {len(configs)} rovers")
        self.worlds: list[tuple[mujoco.MjModel, mujoco.MjData]] = []
        self.rovers: list[_RoverSim] = []
        groups = [configs] if shared_world else [[config] for config in configs]
        for group in groups:
            spec, cells = build_world(group, layout)
            model = spec.compile()
            data = mujoco.MjData(model)
            self.worlds.append((model, data))
            self.rovers += [_RoverSim(model, data, cell, i) for i, cell in enumerate(cells)]
        self.layout = layout
        self.params = params
        self.num_rovers = len(configs)
        timestep = self.worlds[0][0].opt.timestep
        self.substeps = max(1, round(1.0 / (control_hz * timestep)))
        self.dt = self.substeps * timestep
        self._rng = np.random.default_rng(seed)
        # A separate stream so noise-free fleets draw exactly as before.
        self._noise_rng = np.random.default_rng([seed, 1])
        self._range_noise = np.array([p.range_noise for p in self.profiles])
        self._accel_noise = np.array([p.accel_noise for p in self.profiles])
        self._gyro_noise = np.array([p.gyro_noise for p in self.profiles])

    def reset(self, rovers: list[int] | None = None) -> FleetObs:
        """Places rovers at random clear poses in their cells, at rest."""
        for i in range(self.num_rovers) if rovers is None else rovers:
            r = self.rovers[i]
            x, y, yaw = self._sample_clear_pose(r.cell)
            r.data.qpos[r.qpos : r.qpos + 7] = [
                x,
                y,
                self.params.wheel_radius,
                math.cos(yaw / 2),
                0,
                0,
                math.sin(yaw / 2),
            ]
            r.data.qvel[r.dof : r.dof + 6] = 0
            r.data.ctrl[r.act] = 0
        for model, data in self.worlds:
            mujoco.mj_forward(model, data)
        return self.observe()

    def step(self, commands: np.ndarray) -> FleetObs:
        """Applies (linear m/s, angular rad/s) body-velocity commands, shape (N, 2)."""
        half_track = self.params.track_width / 2
        wheels = (
            np.column_stack(
                [commands[:, 0] - commands[:, 1] * half_track, commands[:, 0] + commands[:, 1] * half_track]
            )
            / self.params.wheel_radius
        )
        for r, ctrl in zip(self.rovers, wheels, strict=True):
            r.data.ctrl[r.act] = ctrl
        for model, data in self.worlds:
            # Sensors are only read once per control step, and every
            # rangefinder ray is tested against every geom, so skip sensor
            # evaluation on all but the last substep.
            if self.substeps > 1:
                model.opt.disableflags |= _SENSOR_DISABLE_BIT
                mujoco.mj_step(model, data, nstep=self.substeps - 1)
                model.opt.disableflags &= ~_SENSOR_DISABLE_BIT
            mujoco.mj_step(model, data)
        return self.observe()

    def observe(self) -> FleetObs:
        n = self.num_rovers
        ranges = np.empty((n, len(RANGE_SENSORS)))
        accel = np.empty((n, 3))
        gyro = np.empty((n, 3))
        wheel_vel = np.empty((n, 2))
        bump = np.empty(n, dtype=bool)
        pose = np.empty((n, 3))
        upright = np.empty(n, dtype=bool)
        for i, r in enumerate(self.rovers):
            s = r.data.sensordata
            ranges[i] = s[r.range_adr]
            accel[i] = s[r.accel_adr : r.accel_adr + 3]
            gyro[i] = s[r.gyro_adr : r.gyro_adr + 3]
            wheel_vel[i] = s[r.wheel_adr]
            bump[i] = s[r.bump_adr] > self.params.bump_force
            x, y, _, qw, qx, qy, qz = r.data.qpos[r.qpos : r.qpos + 7]
            yaw = math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
            pose[i] = (x - r.cell.center[0], y - r.cell.center[1], yaw)
            # xmat[8]: z component of the chassis up axis in the world frame.
            upright[i] = r.data.xmat[r.body, 8] > 0.5
        ranges[(ranges < 0) | (ranges > self.params.max_range)] = self.params.max_range
        # Sensor noise corrupts what the rover observes (and so what it records
        # and how it drives); the bumper and pose stay exact.
        if self._range_noise.any():
            ranges += self._range_noise[:, None] * self._noise_rng.standard_normal(ranges.shape)
            np.clip(ranges, 0.0, self.params.max_range, out=ranges)
        if self._accel_noise.any():
            accel += self._accel_noise[:, None] * self._noise_rng.standard_normal(accel.shape)
        if self._gyro_noise.any():
            gyro += self._gyro_noise[:, None] * self._noise_rng.standard_normal(gyro.shape)
        return FleetObs(
            ranges=ranges, accel=accel, gyro=gyro, wheel_vel=wheel_vel, bump=bump, pose=pose, upright=upright
        )

    def _sample_clear_pose(self, cell: Cell, clearance: float = 0.3) -> tuple[float, float, float]:
        inner = self.layout.cell_size / 2 - self.layout.wall_thickness - clearance
        cx, cy = cell.center
        for _ in range(1000):
            x = cx + self._rng.uniform(-inner, inner)
            y = cy + self._rng.uniform(-inner, inner)
            if all(math.hypot(x - o.x, y - o.y) > o.radius + clearance for o in cell.obstacles):
                return x, y, self._rng.uniform(-math.pi, math.pi)
        raise RuntimeError(f"no clear pose in cell {cell.index} (clutter {cell.config.clutter})")
