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
