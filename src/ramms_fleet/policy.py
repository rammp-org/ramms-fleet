"""Exploration policy for data collection.

Deliberately imperfect: rovers wander with a random turn rate and only weakly
steer away from obstacles, so they collide often enough to produce positive
collision labels. After a bump they back off and spin to a new heading.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ramms_fleet.fleet import RANGE_ANGLES_DEG, FleetObs

CRUISE, REVERSE, SPIN = 0, 1, 2


@dataclass(frozen=True)
class WanderParams:
    cruise_speed: float = 0.3
    reverse_speed: float = 0.15
    max_turn_rate: float = 2.5
    turn_noise: float = 1.5
    """Scale of the random turn-rate process, rad/s per sqrt(s)."""
    turn_reversion: float = 1.0
    avoid_gain: float = 0.3
    """0 drives blind; around 1 avoids most obstacles."""
    avoid_distance: float = 0.5
    reverse_seconds: float = 0.6
    spin_seconds: tuple[float, float] = (0.4, 1.2)
    stuck_seconds: float = 3.0


class WanderPolicy:
    def __init__(self, num_rovers: int, dt: float, params: WanderParams = WanderParams(), seed: int = 0):
        self.params = params
        self.dt = dt
        self._rng = np.random.default_rng(seed)
        self.mode = np.full(num_rovers, CRUISE)
        self._timer = np.zeros(num_rovers)
        self._turn = np.zeros(num_rovers)
        self._spin_dir = np.ones(num_rovers)
        self._anchor = np.zeros((num_rovers, 2))
        self._anchor_time = np.zeros(num_rovers)
        self._sin = np.sin(np.deg2rad(RANGE_ANGLES_DEG))

    def reset(self, rovers: np.ndarray | list[int], obs: FleetObs) -> None:
        self.mode[rovers] = CRUISE
        self._timer[rovers] = 0
        self._turn[rovers] = 0
        self._anchor[rovers] = obs.pose[rovers, :2]
        self._anchor_time[rovers] = 0

    def act(self, obs: FleetObs) -> np.ndarray:
        p, dt, n = self.params, self.dt, len(self.mode)
        self._timer -= dt

        # Stuck in cruise without a bump (wedged on an obstacle edge): back off.
        self._anchor_time += dt
        moved = np.linalg.norm(obs.pose[:, :2] - self._anchor, axis=1) > 0.05
        self._anchor[moved] = obs.pose[moved, :2]
        self._anchor_time[moved] = 0
        stuck = (self.mode == CRUISE) & (self._anchor_time > p.stuck_seconds)

        start_reverse = ((self.mode == CRUISE) & obs.bump) | stuck
        self.mode[start_reverse] = REVERSE
        self._timer[start_reverse] = p.reverse_seconds
        self._anchor_time[start_reverse] = 0

        start_spin = (self.mode == REVERSE) & (self._timer <= 0)
        self.mode[start_spin] = SPIN
        self._timer[start_spin] = self._rng.uniform(*p.spin_seconds, size=start_spin.sum())
        self._spin_dir[start_spin] = self._rng.choice([-1.0, 1.0], size=start_spin.sum())

        end_spin = (self.mode == SPIN) & (self._timer <= 0)
        self.mode[end_spin] = CRUISE
        self._turn[end_spin] = 0
        self._anchor[end_spin] = obs.pose[end_spin, :2]
        self._anchor_time[end_spin] = 0

        # Ornstein-Uhlenbeck turn rate plus a weak push away from the nearer side.
        self._turn += -p.turn_reversion * self._turn * dt + p.turn_noise * np.sqrt(dt) * self._rng.standard_normal(n)
        closeness = np.clip(1 - obs.ranges / p.avoid_distance, 0, 1)
        avoid = -p.avoid_gain * p.max_turn_rate * (closeness * self._sin).sum(axis=1)

        commands = np.zeros((n, 2))
        cruise = self.mode == CRUISE
        commands[cruise, 0] = p.cruise_speed
        commands[cruise, 1] = self._turn[cruise] + avoid[cruise]
        commands[self.mode == REVERSE, 0] = -p.reverse_speed
        spin = self.mode == SPIN
        commands[spin, 1] = self._spin_dir[spin] * p.max_turn_rate
        commands[:, 1] = np.clip(commands[:, 1], -p.max_turn_rate, p.max_turn_rate)
        return commands
