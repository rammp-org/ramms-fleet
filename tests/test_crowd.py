import mujoco
import numpy as np

from ramms_fleet.collect import collect, near_pedestrian, spread_configs
from ramms_fleet.crowd import Crowd, CrowdParams
from ramms_fleet.fleet import MujocoFleet
from ramms_fleet.world import EnvConfig, build_cell_xml, build_world


def test_pedestrians_add_bodies_joints_and_actuators_in_order():
    xml, cell = build_cell_xml(EnvConfig(clutter=0.5, pedestrians=2), "arena")
    model = mujoco.MjModel.from_xml_string(xml)
    assert [model.actuator(i).name for i in range(model.nu)] == [
        "wheel_left",
        "wheel_right",
        "ped0_x",
        "ped0_y",
        "ped1_x",
        "ped1_y",
    ]
    # Free root (7) and two wheel hinges come first; the RAMMS backend relies on it.
    assert model.joint("ped0_x").qposadr[0] == 9
    assert len(cell.pedestrian_homes) == 2


def test_no_pedestrians_leaves_the_world_unchanged():
    spec, _ = build_world([EnvConfig(clutter=0.5)])
    assert spec.compile().nu == 2
    assert MujocoFleet([EnvConfig(clutter=0.5)]).reset().pedestrians is None


def test_crowd_walks_inside_the_arena_and_around_obstacles():
    obstacles = np.array([[0.0, 0.0, 0.3], [0.8, -0.6, 0.2]])
    half = 1.45
    crowd = Crowd([obstacles], [5], half, CrowdParams(), seed=0)
    pos = crowd.place(0)[None]
    rover = np.array([[-1.0, 1.0]])
    travelled = 0.0
    closest = np.inf
    for _ in range(4000):
        v = crowd.velocities(pos, rover, 0.05)
        pos = pos + v * 0.05
        travelled += np.linalg.norm(v * 0.05, axis=2).sum()
        gap = np.linalg.norm(pos[0, :, None] - obstacles[None, :, :2], axis=2) - obstacles[None, :, 2]
        closest = min(closest, gap.min())
    assert np.abs(pos).max() < half
    assert closest > 0.0
    assert travelled / 5 > 20.0  # metres each over 200 s


def test_fleet_pedestrians_move_and_repeat_with_the_seed():
    configs = [EnvConfig(clutter=0.4, seed=1, pedestrians=3), EnvConfig(clutter=0.4, seed=2)]

    def run():
        fleet = MujocoFleet(configs, seed=4)
        obs = fleet.reset()
        start = obs.pedestrians.copy()
        for _ in range(200):
            obs = fleet.step(np.array([[0.0, 0.0], [0.0, 0.0]]))
        return start, obs.pedestrians

    start, end = run()
    assert start.shape == (2, 3, 2)
    assert np.isnan(start[1]).all()
    assert np.linalg.norm(end[0] - start[0], axis=1).max() > 0.5
    np.testing.assert_array_equal(run()[1], end)


def test_spread_configs_shuffles_pedestrian_counts():
    configs = spread_configs(8, 0.1, 1.2, seed=0, pedestrians=(0, 7))
    assert sorted(c.pedestrians for c in configs) == list(range(8))
    assert [c.pedestrians for c in configs] != list(range(8))


def test_collect_records_pedestrians(tmp_path):
    configs = [EnvConfig(clutter=0.3, seed=1, pedestrians=4)]
    meta = collect(configs, seconds=10, out_dir=tmp_path, seed=0)
    data = np.load(tmp_path / "rover_00.npz")
    assert data["pedestrians"].shape == (200, 4, 2)
    assert meta["rovers"][0]["pedestrians"] == 4
    pose = np.zeros((3, 3))
    peds = np.array([[[0.1, 0.0]], [[1.0, 1.0]], [[np.nan, np.nan]]])
    assert near_pedestrian(pose, peds).tolist() == [True, False, False]
