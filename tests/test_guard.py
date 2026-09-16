import json

import numpy as np

from ramms_fleet.fleet import FleetObs
from ramms_fleet.guard import GuardedPolicy, GuardParams, run_guard
from ramms_fleet.policy import SPIN, WanderPolicy
from ramms_fleet.spec import WanderParams
from ramms_fleet.world import EnvConfig


class _FixedRisk:
    """Stands in for a model: returns the risk it was given."""

    def __init__(self, values):
        self.values = np.asarray(values, float)
        self.seen = None

    def reset(self, rovers):
        pass

    def __call__(self, features):
        self.seen = features
        return self.values


def _obs(n, ranges):
    return FleetObs(
        ranges=np.asarray(ranges, float),
        accel=np.zeros((n, 3)),
        gyro=np.zeros((n, 3)),
        wheel_vel=np.zeros((n, 2)),
        bump=np.zeros(n, dtype=bool),
        pose=np.zeros((n, 3)),
        upright=np.ones(n, dtype=bool),
    )


def test_guard_brakes_and_turns_away_only_above_the_threshold():
    n = 2
    policy = WanderPolicy(n, dt=0.05, seed=0)
    risk = _FixedRisk([0.9, 0.1])
    guard = GuardedPolicy(policy, risk, GuardParams(threshold=0.5, brake=0.25, turn=0.8))
    # Rover 0 has an obstacle to its right (negative angles), rover 1 is clear.
    obs = _obs(n, [[0.1, 0.1, 2.0, 2.0, 2.0], [2.0] * 5])
    plain = WanderPolicy(n, dt=0.05, seed=0).act(obs)
    commands = guard.act(obs)

    assert guard.guarded.tolist() == [True, False]
    assert commands[0, 0] == 0.25 * plain[0, 0]
    assert commands[0, 1] > 0  # turns left, away from the close right side
    np.testing.assert_allclose(commands[1], plain[1])
    # The scored features carry the command the policy intended, as during collection.
    np.testing.assert_allclose(risk.seen[:, -2:], plain, rtol=1e-6)


def test_guard_leaves_the_recovery_manoeuvre_alone():
    policy = WanderPolicy(1, dt=0.05, seed=0)
    policy.mode[:] = SPIN
    policy._timer[:] = 1.0
    guard = GuardedPolicy(policy, _FixedRisk([1.0]))
    commands = guard.act(_obs(1, [[2.0] * 5]))
    assert policy.mode[0] == SPIN
    assert not guard.guarded[0]
    assert abs(commands[0, 1]) == WanderParams().max_turn_rate


def test_run_guard_reports_distance_and_collisions_without_a_model():
    report = run_guard([EnvConfig(clutter=1.2, seed=1)], seconds=120, model=None, seed=0)
    assert report["fleet"]["guarded_fraction"] == 0
    assert report["fleet"]["metres_per_min"] > 1.0
    # Unguarded rovers in heavy clutter collide, and the rate is per metre travelled.
    assert report["rovers"][0]["collisions_per_min"] > 0
    assert json.dumps(report)
