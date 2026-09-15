"""Scripted pedestrians that share each rover's arena.

Pedestrians walk between random goals, pause, and steer around walls,
obstacles, and each other. Some also give rovers a wide berth; the rest walk
as if rovers were not there. The model only decides walking velocities: each
backend turns them into velocity-actuator commands on the pedestrian bodies,
so pedestrians push and get pushed like any other body and every sensor and
camera sees them.

All positions are relative to the cell centre. Kept free of MuJoCo imports so
both backends share one model.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

ROVER_RADIUS = 0.13
"""Bounding-circle radius of the rover chassis, for steering and clear placement."""


@dataclass(frozen=True)
class CrowdParams:
    radius: float = 0.07
    height: float = 0.45
    mass: float = 5.0
    drive_gain: float = 60.0
    """Velocity-actuator gain, N per m/s."""
    max_force: float = 15.0
    speed: tuple[float, float] = (0.15, 0.35)
    """Walking speed range in m/s; each pedestrian draws one."""
    pause_seconds: tuple[float, float] = (0.5, 3.0)
    avoid_distance: float = 0.3
    """Surface gap at which a pedestrian starts steering away."""
    attentive_fraction: float = 0.5
    """Share of pedestrians that steer around rovers."""
    goal_tolerance: float = 0.1
    stuck_seconds: float = 6.0
    """A pedestrian that makes no progress for this long picks a new goal."""


class Crowd:
    """Walking state for every cell's pedestrians.

    `obstacles` holds each cell's obstacle bounding circles as (x, y, radius)
    rows relative to the cell centre; `counts` is pedestrians per cell. Arrays
    are padded to the busiest cell.
    """

    def __init__(
        self,
        obstacles: list[np.ndarray],
        counts: list[int],
        half_extent: float,
        params: CrowdParams = CrowdParams(),
        seed: int = 0,
    ):
        self.params = params
        self.obstacles = [np.asarray(o, dtype=float).reshape(-1, 3) for o in obstacles]
        self.counts = list(counts)
        self.num_cells = len(counts)
        self.size = max(self.counts, default=0)
        self.half = half_extent
        self._rng = np.random.default_rng([seed, 3])
        shape = (self.num_cells, self.size)
        self.active = np.arange(self.size)[None, :] < np.array(self.counts)[:, None]
        self.speed = self._rng.uniform(*params.speed, size=shape)
        self.attentive = self._rng.random(shape) < params.attentive_fraction
        self.goal = np.zeros((*shape, 2))
        self.pause = np.zeros(shape)
        self._best = np.full(shape, np.inf)
        self._stalled = np.zeros(shape)

    def place(self, cell: int, avoid: list[tuple[float, float, float]] = ()) -> np.ndarray:
        """Random non-overlapping positions for one cell's pedestrians, shape (count, 2). Also picks fresh goals."""
        r = self.params.radius
        placed: list[tuple[float, float, float]] = list(avoid)
        out = np.zeros((self.counts[cell], 2))
        for k in range(self.counts[cell]):
            out[k] = self._sample_point(cell, clearance=r + 0.05, others=placed)
            placed.append((out[k, 0], out[k, 1], r))
            self._new_goal(cell, k, out[k])
            self.pause[cell, k] = 0.0
        return out

    def velocities(self, positions: np.ndarray, rovers: np.ndarray, dt: float) -> np.ndarray:
        """Walking velocities, shape (cells, size, 2), from positions (cells, size, 2) and rover xy (cells, 2)."""
        p = self.params
        out = np.zeros((self.num_cells, self.size, 2))
        for c in range(self.num_cells):
            k = self.counts[c]
            if k == 0:
                continue
            pos = positions[c, :k]
            to_goal = self.goal[c, :k] - pos
            dist = np.linalg.norm(to_goal, axis=1)

            # Arrive, wait, then walk to the next goal; give up on goals that stop getting closer.
            arrived = (dist < p.goal_tolerance) & (self.pause[c, :k] <= 0)
            improved = dist < self._best[c, :k] - 0.05
            self._best[c, :k] = np.where(improved, dist, self._best[c, :k])
            self._stalled[c, :k] = np.where(improved | (self.pause[c, :k] > 0), 0.0, self._stalled[c, :k] + dt)
            for j in np.flatnonzero(arrived):
                self.pause[c, j] = self._rng.uniform(*p.pause_seconds)
            waiting = self.pause[c, :k] > 0
            self.pause[c, :k] = np.maximum(self.pause[c, :k] - dt, 0.0)
            for j in np.flatnonzero((waiting & (self.pause[c, :k] <= 0)) | (self._stalled[c, :k] > p.stuck_seconds)):
                self._new_goal(c, j, pos[j])
                to_goal[j] = self.goal[c, j] - pos[j]
                dist[j] = np.linalg.norm(to_goal[j])

            speed = self.speed[c, :k]
            desired = np.where(waiting[:, None], 0.0, to_goal / np.maximum(dist, 1e-6)[:, None] * speed[:, None])
            push = self._wall_push(pos) + self._circle_push(pos, self.obstacles[c], desired)
            if k > 1:
                others = np.column_stack([pos, np.full(k, p.radius)])
                push += self._circle_push(pos, others, desired, skip_self=True)
            rover = np.array([[rovers[c, 0], rovers[c, 1], ROVER_RADIUS]])
            push += self.attentive[c, :k, None] * self._circle_push(pos, rover, desired)
            v = desired + push * speed[:, None]
            norm = np.linalg.norm(v, axis=1)
            limit = 1.5 * speed
            v *= np.minimum(1.0, limit / np.maximum(norm, 1e-9))[:, None]
            out[c, :k] = v
        return out

    # ---- steering terms, in units of the pedestrian's walking speed

    def _falloff(self, gap: np.ndarray) -> np.ndarray:
        return np.clip(1.0 - gap / self.params.avoid_distance, 0.0, 2.0) ** 2

    def _wall_push(self, pos: np.ndarray) -> np.ndarray:
        r = self.params.radius
        push = np.zeros_like(pos)
        for axis in (0, 1):
            push[:, axis] += self._falloff(pos[:, axis] + self.half - r)
            push[:, axis] -= self._falloff(self.half - pos[:, axis] - r)
        return push

    def _circle_push(self, pos: np.ndarray, circles: np.ndarray, desired: np.ndarray, skip_self: bool = False):
        """Pushes away from circles, plus a sideways term so a pedestrian walks around rather than stalling."""
        if len(circles) == 0:
            return np.zeros_like(pos)
        diff = pos[:, None, :] - circles[None, :, :2]
        centre = np.linalg.norm(diff, axis=2)
        gap = centre - circles[None, :, 2] - self.params.radius
        weight = self._falloff(gap)
        if skip_self:
            np.fill_diagonal(weight, 0.0)
        away = diff / np.maximum(centre, 1e-6)[:, :, None]
        side = np.stack([-away[..., 1], away[..., 0]], axis=2)
        # Turn to whichever side the pedestrian is already heading.
        sign = np.sign(np.einsum("pcd,pd->pc", side, desired))
        sign[sign == 0] = 1.0
        return (weight[:, :, None] * (away + 0.5 * sign[:, :, None] * side)).sum(axis=1)

    # ---- sampling

    def _new_goal(self, cell: int, k: int, pos: np.ndarray) -> None:
        # Goals at least a metre away keep pedestrians crossing the arena instead of shuffling in place.
        for _ in range(20):
            goal = self._sample_point(cell, clearance=self.params.radius + 0.1)
            if np.linalg.norm(goal - pos) > 1.0:
                break
        self.goal[cell, k] = goal
        self._best[cell, k] = np.inf
        self._stalled[cell, k] = 0.0

    def _sample_point(self, cell: int, clearance: float, others: list[tuple[float, float, float]] = ()) -> np.ndarray:
        inner = self.half - clearance
        circles = [*map(tuple, self.obstacles[cell]), *others]
        for _ in range(1000):
            x, y = self._rng.uniform(-inner, inner, size=2)
            if all(math.hypot(x - ox, y - oy) > orad + clearance for ox, oy, orad in circles):
                return np.array([x, y])
        raise RuntimeError(f"no clear pedestrian position in cell {cell}")


def home_positions(
    obstacles: np.ndarray, count: int, half_extent: float, params: CrowdParams, seed: int
) -> list[tuple[float, float]]:
    """Where each pedestrian body sits in the arena file, clear of obstacles, each other, and the rover's start.

    Resets move pedestrians away from here straight away; the homes only keep
    the model free of overlaps before the first reset.
    """
    crowd = Crowd([obstacles], [0], half_extent, params)
    crowd._rng = np.random.default_rng([seed, 4])
    placed = [(0.0, 0.0, ROVER_RADIUS + 0.2)]
    homes = []
    for _ in range(count):
        x, y = crowd._sample_point(0, clearance=params.radius + 0.05, others=placed)
        placed.append((x, y, params.radius))
        homes.append((float(x), float(y)))
    return homes
