"""Rover, policy, and feature constants shared by simulation and learning.

Kept free of MuJoCo imports: every federated ClientApp task starts a fresh
process, and the learning side only needs these values.
"""

from __future__ import annotations

from dataclasses import dataclass

RANGE_SENSORS = ("range_m60", "range_m30", "range_0", "range_p30", "range_p60")
RANGE_ANGLES_DEG = (-60.0, -30.0, 0.0, 30.0, 60.0)

FEATURES = (
    *RANGE_SENSORS,
    "accel_x",
    "accel_y",
    "accel_z",
    "gyro_x",
    "gyro_y",
    "gyro_z",
    "wheel_left",
    "wheel_right",
    "cmd_v",
    "cmd_w",
)


@dataclass(frozen=True)
class RoverParams:
    """Must match assets/rover.xml."""

    wheel_radius: float = 0.035
    track_width: float = 0.19
    max_range: float = 2.0
    """Rangefinder readings beyond this, or with no hit, are reported as this."""
    bump_force: float = 0.1
    """Touch-sensor force in newtons above which the rover counts as bumping."""


@dataclass(frozen=True)
class RoverProfile:
    """Per-rover hardware and behaviour differences. The defaults are the ideal rover."""

    cruise_speed: float = 0.3
    """m/s while wandering."""
    range_noise: float = 0.0
    """Standard deviation of Gaussian rangefinder noise, metres."""
    accel_noise: float = 0.0
    """Standard deviation of accelerometer noise per axis, m/s^2."""
    gyro_noise: float = 0.0
    """Standard deviation of gyro noise per axis, rad/s."""


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
