import math

import numpy as np
import pytest

from ramms_fleet.collect import FEATURES, collect, collision_labels, spread_configs
from ramms_fleet.fleet import RANGE_SENSORS, MujocoFleet
from ramms_fleet.world import EnvConfig, build_world


def test_world_composes_multiple_rovers():
    spec, cells = build_world([EnvConfig(clutter=0.0), EnvConfig(clutter=1.0, seed=3)])
    model = spec.compile()
    assert model.nu == 4
    assert model.actuator("rover1/wheel_right") is not None
    assert len(cells[0].obstacles) == 0
    assert len(cells[1].obstacles) > 0


def drive(fleet: MujocoFleet, v: float, w: float, seconds: float):
    obs = None
    for _ in range(round(seconds / fleet.dt)):
        obs = fleet.step(np.array([[v, w]]))
    return obs


@pytest.fixture
def empty_fleet():
    fleet = MujocoFleet([EnvConfig(clutter=0.0)], seed=1)
    rover = fleet.rovers[0]
    fleet.reset()
    # Start at the cell centre facing +x so distances to the walls are known.
    rover.data.qpos[rover.qpos : rover.qpos + 7] = [0, 0, fleet.params.wheel_radius, 1, 0, 0, 0]
    return fleet


def test_forward_command_moves_along_heading(empty_fleet):
    obs = drive(empty_fleet, 0.3, 0.0, 2.0)
    x, y, yaw = obs.pose[0]
    assert x > 0.4
    assert abs(y) < 0.05
    assert abs(yaw) < 0.1
    assert obs.upright[0]


def test_positive_turn_rate_turns_left(empty_fleet):
    obs = drive(empty_fleet, 0.0, 1.0, 1.0)
    assert 0.6 < obs.pose[0, 2] < 1.4


def test_front_range_shrinks_towards_wall_and_bumps(empty_fleet):
    start = drive(empty_fleet, 0.0, 0.0, 0.1).ranges[0, RANGE_SENSORS.index("range_0")]
    assert start == pytest.approx(1.5 - 0.025 - 0.117, abs=0.03)
    obs = drive(empty_fleet, 0.3, 0.0, 6.0)
    assert obs.ranges[0, RANGE_SENSORS.index("range_0")] < 0.05
    assert obs.bump[0]


def test_reset_poses_are_clear_of_obstacles():
    fleet = MujocoFleet([EnvConfig(clutter=1.2, seed=s) for s in range(4)], seed=2)
    for _ in range(5):
        obs = fleet.reset()
        assert not obs.bump.any()
        for rover, (x, y, _) in zip(fleet.rovers, obs.pose, strict=True):
            for o in rover.cell.obstacles:
                assert math.hypot(x - (o.x - rover.cell.center[0]), y - (o.y - rover.cell.center[1])) > o.radius


def test_collision_labels_precede_onsets_within_episode():
    bump = np.array([0, 0, 0, 0, 1, 1, 0, 0, 1, 0], dtype=bool)
    episode = np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1])
    labels = collision_labels(bump, episode, horizon_steps=2)
    # Onset at 4 labels 2 and 3. Onset at 8 labels only 7: step 6 is in the previous episode.
    assert labels.tolist() == [0, 0, 1, 1, 0, 0, 0, 1, 0, 0]


def test_collect_writes_per_rover_datasets(tmp_path):
    meta = collect(spread_configs(2, 1.0, 1.2, seed=0), seconds=30, out_dir=tmp_path, seed=0)
    assert (tmp_path / "meta.json").exists()
    for i in range(2):
        data = np.load(tmp_path / f"rover_{i:02d}.npz")
        steps = round(30 / meta["dt"])
        assert data["features"].shape == (steps, len(FEATURES))
        assert data["label"].shape == data["valid"].shape == (steps,)
        assert not data["bump"][data["valid"]].any()


def test_collection_is_deterministic(tmp_path):
    configs = spread_configs(2, 0.5, 1.0, seed=4)
    collect(configs, seconds=10, out_dir=tmp_path / "a", seed=4)
    collect(configs, seconds=10, out_dir=tmp_path / "b", seed=4)
    for i in range(2):
        a = np.load(tmp_path / "a" / f"rover_{i:02d}.npz")
        b = np.load(tmp_path / "b" / f"rover_{i:02d}.npz")
        np.testing.assert_array_equal(a["features"], b["features"])
