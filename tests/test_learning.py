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


def _camera_file(path, steps=3000, seed=0):
    """A synthetic run where only the camera reveals the label: bright frames precede collisions."""
    rng = np.random.default_rng(seed)
    label = rng.random(steps) < 0.2
    images = rng.integers(0, 60, size=(steps, 48, 64), dtype=np.uint8)
    images[label, 10:38, 20:44] = 220
    age = np.full(steps, 0.0, dtype=np.float32)
    age[::10] = 1.0  # every tenth frame is stale
    np.savez_compressed(
        path,
        features=rng.normal(size=(steps, len(FEATURES))).astype(np.float32),
        label=label,
        valid=np.ones(steps, dtype=bool),
        bump=np.zeros(steps, dtype=bool),
        episode=np.zeros(steps, dtype=np.int32),
        pose=np.zeros((steps, 3), dtype=np.float32),
        images=images,
        image_age=age,
    )


def test_camera_inputs_share_samples_and_learn_from_frames(tmp_path):
    from ramms_fleet.experiment import load_model, save_model

    path = tmp_path / "rover_00.npz"
    _camera_file(path)
    features_only = load_rover(path, inputs="features")
    camera = load_rover(path, inputs="camera")
    both = load_rover(path, inputs="both")
    # Stale frames are dropped for every choice of inputs.
    assert len(features_only.y_train) == len(camera.y_train) == len(both.y_train) < 0.95 * 3000 * 0.8
    assert camera.img_train.shape[1:] == (1, 24, 32) and features_only.img_train is None

    torch.manual_seed(0)
    model = make_model(camera.input_dim, 32, "camera")
    train(model, camera.x_train, camera.y_train, epochs=3, seed=0, images=camera.img_train)
    assert evaluate(model, camera.x_test, camera.y_test, camera.img_test)["auroc"] > 0.95

    save_model(tmp_path / "m.pt", model, camera.input_dim, 32, 4, "camera")
    loaded, history, inputs = load_model(tmp_path / "m.pt")
    assert (history, inputs) == (4, "camera")
    np.testing.assert_allclose(
        evaluate(loaded, camera.x_test, camera.y_test, camera.img_test)["auroc"],
        evaluate(model, camera.x_test, camera.y_test, camera.img_test)["auroc"],
    )


def test_camera_inputs_need_frames(rover_file):
    with pytest.raises(ValueError, match="no camera frames"):
        load_rover(rover_file, inputs="both")


def _moving_camera_file(path, steps=3000, seed=0):
    """A synthetic run where only movement between frames reveals the label: the bright patch slides.

    One bright patch drifts along a random walk. It is positive exactly when the
    patch moved on this step, which no single frame can show: the patch sits at
    an arbitrary column either way.
    """
    rng = np.random.default_rng(seed)
    images = rng.integers(0, 60, size=(steps, 48, 64), dtype=np.uint8)
    step_size = rng.choice([0, 7], size=steps, p=[0.7, 0.3])
    column = (8 + np.cumsum(step_size)) % 40
    label = step_size > 0
    for t in range(steps):
        images[t, 10:38, column[t] : column[t] + 16] = 220
    np.savez_compressed(
        path,
        features=rng.normal(size=(steps, len(FEATURES))).astype(np.float32),
        label=label,
        valid=np.ones(steps, dtype=bool),
        bump=np.zeros(steps, dtype=bool),
        episode=np.zeros(steps, dtype=np.int32),
        pose=np.zeros((steps, 3), dtype=np.float32),
        images=images,
        image_age=np.zeros(steps, dtype=np.float32),
    )


def test_frame_history_stacks_the_feature_window(tmp_path):
    from ramms_fleet.experiment import load_model, save_model

    path = tmp_path / "rover_00.npz"
    _camera_file(path)
    one, many = load_rover(path, inputs="camera"), load_rover(path, inputs="camera-history")
    # Same samples as every other choice of inputs, one channel per step of the window.
    assert len(one.y_train) == len(many.y_train)
    assert many.img_train.shape[1:] == (4, 24, 32)
    # The newest frame is last, and matches the single-frame loader.
    np.testing.assert_allclose(many.img_train[:, 3], one.img_train[:, 0])

    model = make_model(many.input_dim, 32, "camera-history")
    assert model.cnn[0].in_channels == 4
    save_model(tmp_path / "m.pt", model, many.input_dim, 32, 4, "camera-history")
    loaded, history, inputs = load_model(tmp_path / "m.pt")
    assert (history, inputs) == (4, "camera-history")
    np.testing.assert_allclose(
        evaluate(loaded, many.x_test, many.y_test, many.img_test)["auroc"],
        evaluate(model, many.x_test, many.y_test, many.img_test)["auroc"],
    )


def test_frame_history_learns_motion_a_single_frame_cannot(tmp_path):
    path = tmp_path / "rover_00.npz"
    _moving_camera_file(path)

    def auroc(inputs):
        data = load_rover(path, inputs=inputs)
        torch.manual_seed(0)
        model = make_model(data.input_dim, 32, inputs)
        train(model, data.x_train, data.y_train, epochs=4, seed=0, images=data.img_train)
        return evaluate(model, data.x_test, data.y_test, data.img_test)["auroc"]

    assert auroc("camera-history") > 0.9
    assert auroc("camera") < 0.7
