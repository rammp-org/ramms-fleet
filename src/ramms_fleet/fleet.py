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

from ramms_fleet.crowd import Crowd, CrowdParams
from ramms_fleet.spec import RANGE_ANGLES_DEG, RANGE_SENSORS, RoverParams, RoverProfile
from ramms_fleet.world import ROVER_PREFIX, ArenaLayout, Cell, EnvConfig, Obstacle, build_world

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
    images: np.ndarray | None = None  # (N, H, W) uint8 grayscale front-camera frames, when the backend has a camera
    image_age: np.ndarray | None = None  # (N,) seconds of simulated time between each frame and this observation
    pedestrians: np.ndarray | None = (
        None  # (N, P, 2) pedestrian x, y relative to the cell centre, NaN past a cell's count
    )


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
        peds = range(cell.config.pedestrians)
        # (P, 2) addresses per pedestrian, x then y.
        self.ped_qpos = np.array(
            [[m.joint(f"{p}ped{k}_{a}").qposadr[0] for a in "xy"] for k in peds], dtype=int
        ).reshape(-1, 2)
        self.ped_dof = np.array([[m.joint(f"{p}ped{k}_{a}").dofadr[0] for a in "xy"] for k in peds], dtype=int).reshape(
            -1, 2
        )
        self.ped_act = np.array([[m.actuator(f"{p}ped{k}_{a}").id for a in "xy"] for k in peds], dtype=int).reshape(
            -1, 2
        )
        self.ped_home = np.array(cell.pedestrian_homes, dtype=float).reshape(-1, 2)


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
        crowd: CrowdParams = CrowdParams(),
    ):
        self.profiles = profiles or [RoverProfile()] * len(configs)
        if len(self.profiles) != len(configs):
            raise ValueError(f"{len(self.profiles)} profiles for {len(configs)} rovers")
        self.worlds: list[tuple[mujoco.MjModel, mujoco.MjData]] = []
        self.rovers: list[_RoverSim] = []
        groups = [configs] if shared_world else [[config] for config in configs]
        for group in groups:
            spec, cells = build_world(group, layout, crowd)
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
        self._noise = SensorNoise(self.profiles, seed, params.max_range)
        self.crowd = make_crowd([r.cell for r in self.rovers], layout, crowd, seed)

    def reset(self, rovers: list[int] | None = None) -> FleetObs:
        """Places rovers at random clear poses in their cells, at rest.

        A full reset also scatters the pedestrians; a partial reset leaves them
        walking and places the rover clear of them.
        """
        if rovers is None and self.crowd is not None:
            for i, r in enumerate(self.rovers):
                placed = self.crowd.place(i)
                r.data.qpos[r.ped_qpos] = placed - r.ped_home
                r.data.qvel[r.ped_dof] = 0
                r.data.ctrl[r.ped_act] = 0
        peds = self._pedestrian_positions()
        for i in range(self.num_rovers) if rovers is None else rovers:
            r = self.rovers[i]
            x, y, yaw = sample_clear_pose(
                r.cell, self.layout, self._rng, avoid=pedestrian_circles(r.cell, peds[i], self.crowd)
            )
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
        if self.crowd is not None:
            rover_xy = np.array([r.data.qpos[r.qpos : r.qpos + 2] - r.cell.center for r in self.rovers])
            walk = self.crowd.velocities(self._pedestrian_positions(), rover_xy, self.dt)
            for i, r in enumerate(self.rovers):
                r.data.ctrl[r.ped_act] = walk[i, : len(r.ped_act)]
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
        self._noise.apply(ranges, accel, gyro)
        return FleetObs(
            ranges=ranges,
            accel=accel,
            gyro=gyro,
            wheel_vel=wheel_vel,
            bump=bump,
            pose=pose,
            upright=upright,
            pedestrians=None if self.crowd is None else self._pedestrian_positions(),
        )

    def _pedestrian_positions(self) -> np.ndarray:
        size = 0 if self.crowd is None else self.crowd.size
        out = np.full((self.num_rovers, size, 2), np.nan)
        for i, r in enumerate(self.rovers):
            out[i, : len(r.ped_qpos)] = r.ped_home + r.data.qpos[r.ped_qpos]
        return out


class SensorNoise:
    """Per-rover Gaussian sensor noise, shared by every backend.

    Noise corrupts what the rover observes (and so what it records and how it
    drives); the bumper and pose stay exact. Its random stream is separate
    from pose sampling, so noise-free fleets draw exactly as before.
    """

    def __init__(self, profiles: list[RoverProfile], seed: int, max_range: float):
        self._rng = np.random.default_rng([seed, 1])
        self._range = np.array([p.range_noise for p in profiles])
        self._accel = np.array([p.accel_noise for p in profiles])
        self._gyro = np.array([p.gyro_noise for p in profiles])
        self._max_range = max_range

    def apply(self, ranges: np.ndarray, accel: np.ndarray, gyro: np.ndarray) -> None:
        """Adds noise in place; ranges stay within [0, max_range]."""
        if self._range.any():
            ranges += self._range[:, None] * self._rng.standard_normal(ranges.shape)
            np.clip(ranges, 0.0, self._max_range, out=ranges)
        if self._accel.any():
            accel += self._accel[:, None] * self._rng.standard_normal(accel.shape)
        if self._gyro.any():
            gyro += self._gyro[:, None] * self._rng.standard_normal(gyro.shape)


def make_crowd(cells: list[Cell], layout: ArenaLayout, params: CrowdParams, seed: int) -> Crowd | None:
    """The shared pedestrian model for a fleet, or None when no cell has pedestrians."""
    if not any(c.config.pedestrians for c in cells):
        return None
    obstacles = [np.array([[o.x - c.center[0], o.y - c.center[1], o.radius] for o in c.obstacles]) for c in cells]
    half = layout.cell_size / 2 - layout.wall_thickness
    return Crowd(obstacles, [c.config.pedestrians for c in cells], half, params, seed)


def pedestrian_circles(cell: Cell, positions: np.ndarray, crowd: Crowd | None) -> list[Obstacle]:
    """Pedestrians in one cell as world-frame obstacles, from cell-relative positions (P, 2)."""
    if crowd is None:
        return []
    cx, cy = cell.center
    return [Obstacle(x=cx + x, y=cy + y, radius=crowd.params.radius) for x, y in positions[: cell.config.pedestrians]]


def sample_clear_pose(
    cell: Cell,
    layout: ArenaLayout,
    rng: np.random.Generator,
    clearance: float = 0.3,
    avoid: list[Obstacle] = (),
) -> tuple[float, float, float]:
    """A random (x, y, yaw) in the cell, at least `clearance` from walls, obstacle bounds, and `avoid`."""
    inner = layout.cell_size / 2 - layout.wall_thickness - clearance
    cx, cy = cell.center
    circles = [*cell.obstacles, *avoid]
    for _ in range(1000):
        x = cx + rng.uniform(-inner, inner)
        y = cy + rng.uniform(-inner, inner)
        if all(math.hypot(x - o.x, y - o.y) > o.radius + clearance for o in circles):
            return x, y, rng.uniform(-math.pi, math.pi)
    raise RuntimeError(f"no clear pose in cell {cell.index} (clutter {cell.config.clutter})")
