import numpy as np
import pytest
import torch

from ramms_fleet.collect import FEATURES, collect, spread_configs
from ramms_fleet.learning import average_precision, evaluate, load_rover, make_model, roc_auc, train


def test_roc_auc_matches_pairwise_definition():
    rng = np.random.default_rng(0)
    y = rng.random(200) < 0.3
    scores = np.round(rng.random(200), 1)  # rounding forces ties
    pos, neg = scores[y], scores[~y]
    pairwise = ((pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()) / (
        len(pos) * len(neg)
    )
    assert roc_auc(y.astype(float), scores) == pytest.approx(pairwise)


def test_average_precision_perfect_and_worst():
    y = np.array([1, 1, 0, 0, 0], dtype=float)
    assert average_precision(y, np.array([0.9, 0.8, 0.3, 0.2, 0.1])) == pytest.approx(1.0)
    # Positives ranked last: precision at their ranks is 1/4 and 2/5.
    assert average_precision(y, np.array([0.1, 0.2, 0.9, 0.8, 0.7])) == pytest.approx((1 / 4 + 2 / 5) / 2)


@pytest.fixture(scope="module")
def rover_file(tmp_path_factory):
    out = tmp_path_factory.mktemp("run")
    collect(spread_configs(1, 1.0, 1.0, seed=7), seconds=120, out_dir=out, seed=7)
    return out / "rover_00.npz"


def test_load_rover_windows_and_split(rover_file):
    data = load_rover(rover_file, history=3, test_fraction=0.25)
    assert data.input_dim == 3 * len(FEATURES)
    assert len(data.x_train) > 3 * len(data.x_test) * 0.8
    assert set(np.unique(data.y_train)) <= {0.0, 1.0}
    assert data.y_train.mean() > 0


def test_training_learns_collision_signal(rover_file):
    torch.manual_seed(0)
    data = load_rover(rover_file)
    model = make_model(data.input_dim)
    before = evaluate(model, data.x_train, data.y_train)["auroc"]
    train(model, data.x_train, data.y_train, epochs=15, seed=0)
    after = evaluate(model, data.x_train, data.y_train)["auroc"]
    assert after > max(before, 0.8)


def test_distance_labels_use_travel_not_time():
    from ramms_fleet.labels import distance_labels

    steps = 40
    episode = np.zeros(steps, dtype=int)
    bump = np.zeros(steps, dtype=bool)
    bump[20] = True
    slow = np.column_stack([np.arange(steps) * 0.01, np.zeros(steps), np.zeros(steps)])  # 0.01 m per step
    fast = np.column_stack([np.arange(steps) * 0.05, np.zeros(steps), np.zeros(steps)])  # 0.05 m per step
    # 0.15 m of travel is 15 steps for the slow rover and 3 for the fast one.
    assert np.flatnonzero(distance_labels(bump, episode, slow, 0.15)).tolist() == list(range(5, 20))
    assert np.flatnonzero(distance_labels(bump, episode, fast, 0.15)).tolist() == [17, 18, 19]
    # Capped window, and no labels across an episode boundary.
    assert distance_labels(bump, episode, np.zeros((steps, 3)), 0.15, max_steps=4).sum() == 4
    split = episode.copy()
    split[18:] = 1
    assert np.flatnonzero(distance_labels(bump, split, slow, 0.15)).tolist() == [18, 19]
