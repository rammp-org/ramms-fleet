"""Integration tests against a running RAMMS editor.

Skipped unless RAMMS_BRIDGE is set, for example RAMMS_BRIDGE=tcp://127.0.0.1:5559.
They create or reuse the FleetArena level and start Play In Editor.
"""

import math
import os

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(not os.environ.get("RAMMS_BRIDGE"), reason="set RAMMS_BRIDGE to run against RAMMS")


@pytest.fixture(scope="module")
def ramms_fleet():
    from ramms_fleet.collect import spread_configs
    from ramms_fleet.ramms.fleet import RammsFleet

    fleet = RammsFleet(spread_configs(3, 0.4, 1.2, seed=5), seed=5, address=os.environ["RAMMS_BRIDGE"])
    yield fleet
    fleet.close()


def test_rangefinders_match_standalone_mujoco(ramms_fleet):
    import mujoco

    from ramms_fleet.spec import RANGE_SENSORS
    from ramms_fleet.world import build_cell_xml

    obs = ramms_fleet.reset()
    for i, cell in enumerate(ramms_fleet.cells):
        xml, _ = build_cell_xml(cell.config, ramms_fleet.names[i])
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
        x, y, yaw = obs.pose[i]
        data.qpos[:7] = [x, y, 0.035, math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
        mujoco.mj_forward(model, data)
        expected = np.array([data.sensordata[model.sensor(s).adr[0]] for s in RANGE_SENSORS])
        expected[(expected < 0) | (expected > 2.0)] = 2.0
        np.testing.assert_allclose(obs.ranges[i], expected, atol=1e-4)


def test_driving_moves_rovers_forward(ramms_fleet):
    obs = ramms_fleet.reset()
    start = obs.pose.copy()
    for _ in range(10):
        obs = ramms_fleet.step(np.tile([0.3, 0.0], (ramms_fleet.num_rovers, 1)))
    heading = np.column_stack([np.cos(start[:, 2]), np.sin(start[:, 2])])
    travelled = ((obs.pose[:, :2] - start[:, :2]) * heading).sum(axis=1)
    assert np.all(travelled > 0.05) or obs.bump.any()
