"""Collision-prediction model, per-rover datasets, training, and metrics.

Nothing here knows about Flower; the federated apps, the baselines, and the
evaluation script all use these functions, so every method trains and is scored
the same way.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from ramms_fleet.labels import distance_labels
from ramms_fleet.spec import RANGE_SENSORS, RoverParams, WanderParams

# Fixed per-feature scales from physical limits rather than dataset statistics,
# so normalizing never needs information from other rovers.
FEATURE_SCALE = np.array(
    [RoverParams.max_range] * len(RANGE_SENSORS)
    + [10.0, 10.0, 10.0]  # accel, m/s^2 (gravity included)
    + [3.0, 3.0, 3.0]  # gyro, rad/s
    + [25.0, 25.0]  # wheel velocity, rad/s
    + [WanderParams.cruise_speed, WanderParams.max_turn_rate],  # commanded v, w
    dtype=np.float32,
)


@dataclass
class RoverData:
    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    img_train: np.ndarray | None = None
    """(N, C, H, W) camera frames in [0, 1] when the model uses the camera, oldest frame first."""
    img_test: np.ndarray | None = None

    @property
    def input_dim(self) -> int:
        return self.x_train.shape[1]


LABELS = ("time", "distance")
INPUTS = ("features", "camera", "both", "camera-history", "both-history")
"""What the model sees. The "-history" choices stack the same window of frames the features cover."""
# Camera frames are average-pooled by this factor on load (64x48 recorded, 32x24 to the
# model), which keeps CNN training on the CPU fast.
CAMERA_DOWNSAMPLE = 2


def camera_frames(inputs: str, history: int) -> int:
    """How many camera frames the model stacks: the feature window, or just the latest frame."""
    if inputs == "features":
        return 0
    return history if inputs.endswith("-history") else 1


def load_rover(
    path: Path | str,
    history: int = 4,
    test_fraction: float = 0.2,
    label: str = "time",
    horizon_m: float = 0.15,
    inputs: str = "features",
    max_image_age: float = 0.1,
) -> RoverData:
    """Loads one rover's file as windowed samples with a chronological split.

    A sample at step t stacks the scaled features of steps t-history+1..t, all in
    the same episode, and is kept only if step t is valid. The last
    `test_fraction` of the run is held out, so test samples never sit next to
    training samples in time.

    `label="time"` uses the labels stored at collection (collision within the
    run's time horizon); `label="distance"` relabels from the recorded poses:
    collision within `horizon_m` metres of travel.

    `inputs` picks what the model sees: the feature window, camera frames, or
    both. The "-history" choices stack every frame of the window as a channel,
    so the model can see movement; the others use the latest frame only. When
    the file has camera frames, samples with a frame in the window older than
    `max_image_age` seconds are dropped for every choice of inputs, so all
    models are scored on the same samples.
    """
    if inputs not in INPUTS:
        raise ValueError(f"inputs must be one of {INPUTS}, not {inputs!r}")
    data = np.load(path)
    features = data["features"] / FEATURE_SCALE
    valid, episode = data["valid"], data["episode"]
    if label == "time":
        labels = data["label"]
    elif label == "distance":
        labels = distance_labels(data["bump"], episode, data["pose"], horizon_m)
    else:
        raise ValueError(f"label must be one of {LABELS}, not {label!r}")
    steps, dim = features.shape

    ends = np.arange(history - 1, steps)
    same_episode = episode[ends] == episode[ends - history + 1]
    keep = valid[ends] & same_episode
    has_images = "images" in data.files
    if has_images:
        age = data["image_age"]
        for k in range(history):
            keep &= age[ends - k] <= max_image_age
    elif inputs != "features":
        raise ValueError(f"{path} has no camera frames; collect with --camera to use inputs={inputs!r}")
    ends = ends[keep]
    windows = np.stack([features[ends - k] for k in range(history - 1, -1, -1)], axis=1)
    x = windows.reshape(len(ends), history * dim).astype(np.float32)
    y = labels[ends].astype(np.float32)

    split = ends < round(steps * (1 - test_fraction))
    img = None
    channels = camera_frames(inputs, history)
    if channels:
        stack = np.stack([data["images"][ends - k] for k in range(channels - 1, -1, -1)], axis=1)
        frames = stack.astype(np.float32) / 255.0
        k = CAMERA_DOWNSAMPLE
        if k > 1:
            n, c, h, w = frames.shape
            frames = frames[:, :, : h - h % k, : w - w % k].reshape(n, c, h // k, k, w // k, k).mean(axis=(3, 5))
        img = frames
    return RoverData(
        x_train=x[split],
        y_train=y[split],
        x_test=x[~split],
        y_test=y[~split],
        img_train=None if img is None else img[split],
        img_test=None if img is None else img[~split],
    )


class CollisionNet(nn.Module):
    """Collision predictor that can see the camera: a small CNN, optionally fused with the feature MLP."""

    def __init__(self, input_dim: int, hidden: int = 64, inputs: str = "both", frames: int = 1):
        super().__init__()
        self.inputs = inputs
        self.cnn = nn.Sequential(
            nn.Conv2d(frames, 8, 5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(8, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((3, 4)),
            nn.Flatten(),
            nn.Linear(16 * 3 * 4, hidden),
            nn.ReLU(),
        )
        self.mlp = (
            nn.Sequential(nn.Linear(input_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
            if inputs.startswith("both")
            else None
        )
        width = hidden * (2 if self.mlp is not None else 1)
        self.head = nn.Sequential(nn.Linear(width, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, x: torch.Tensor, img: torch.Tensor) -> torch.Tensor:
        parts = [self.cnn(img)]
        if self.mlp is not None:
            parts.append(self.mlp(x))
        return self.head(torch.cat(parts, dim=1))


def forward(model: nn.Module, x: torch.Tensor, img: torch.Tensor | None) -> torch.Tensor:
    """Calls a feature-only model with x, and a camera model with x and the frame."""
    return model(x, img) if isinstance(model, CollisionNet) else model(x)


def make_model(input_dim: int, hidden: int = 64, inputs: str = "features", history: int = 4) -> nn.Module:
    if inputs != "features":
        return CollisionNet(input_dim, hidden, inputs, camera_frames(inputs, history))
    return nn.Sequential(
        nn.Linear(input_dim, hidden),
        nn.ReLU(),
        nn.Linear(hidden, hidden),
        nn.ReLU(),
        nn.Linear(hidden, 1),
    )


def train(
    model: nn.Module,
    x: np.ndarray,
    y: np.ndarray,
    epochs: int,
    lr: float = 1e-3,
    batch_size: int = 256,
    seed: int = 0,
    proximal_mu: float = 0.0,
    images: np.ndarray | None = None,
) -> float:
    """Trains in place and returns the mean loss of the last epoch.

    The positive class is re-weighted by this dataset's own negative/positive
    ratio. `proximal_mu` adds the FedProx penalty towards the starting weights.
    """
    generator = torch.Generator().manual_seed(seed)
    xt, yt = torch.from_numpy(x), torch.from_numpy(y)
    it = None if images is None else torch.from_numpy(images)
    positives = float(yt.sum())
    pos_weight = torch.tensor((len(yt) - positives) / max(positives, 1.0))
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    anchor = [p.detach().clone() for p in model.parameters()] if proximal_mu > 0 else None

    model.train()
    last = 0.0
    for _ in range(epochs):
        order = torch.randperm(len(xt), generator=generator)
        total = 0.0
        for start in range(0, len(xt), batch_size):
            batch = order[start : start + batch_size]
            logits = forward(model, xt[batch], None if it is None else it[batch])
            loss = loss_fn(logits.squeeze(1), yt[batch])
            if anchor is not None:
                loss = loss + proximal_mu / 2 * sum(
                    ((p - a) ** 2).sum() for p, a in zip(model.parameters(), anchor, strict=True)
                )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item() * len(batch)
        last = total / max(len(xt), 1)
    return last


@torch.no_grad()
def predict(model: nn.Module, x: np.ndarray, images: np.ndarray | None = None, batch_size: int = 2048) -> np.ndarray:
    model.eval()
    out = []
    for start in range(0, len(x), batch_size):
        xb = torch.from_numpy(x[start : start + batch_size])
        ib = None if images is None else torch.from_numpy(images[start : start + batch_size])
        out.append(torch.sigmoid(forward(model, xb, ib).squeeze(1)).numpy())
    return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)


def evaluate(model: nn.Module, x: np.ndarray, y: np.ndarray, images: np.ndarray | None = None) -> dict[str, float]:
    scores = predict(model, x, images)
    eps = 1e-7
    bce = -np.mean(y * np.log(scores + eps) + (1 - y) * np.log(1 - scores + eps))
    return {
        "auprc": average_precision(y, scores),
        "auroc": roc_auc(y, scores),
        "bce": float(bce),
        "positive_rate": float(y.mean()) if len(y) else 0.0,
        "num_examples": int(len(y)),
    }


def roc_auc(y: np.ndarray, scores: np.ndarray) -> float:
    """Area under the ROC curve via the rank-sum statistic, ties averaged."""
    positives = y > 0.5
    n_pos, n_neg = int(positives.sum()), int((~positives).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(1, len(scores) + 1)
    # Average the ranks of tied scores.
    _, first, counts = np.unique(sorted_scores, return_index=True, return_counts=True)
    for start, count in zip(first[counts > 1], counts[counts > 1], strict=True):
        ranks[order[start : start + count]] = start + (count + 1) / 2
    return float((ranks[positives].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def average_precision(y: np.ndarray, scores: np.ndarray) -> float:
    """Area under the precision-recall curve as a step function over thresholds."""
    positives = y > 0.5
    n_pos = int(positives.sum())
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    hits = positives[order]
    true_pos = np.cumsum(hits)
    precision = true_pos / np.arange(1, len(hits) + 1)
    return float((precision * hits).sum() / n_pos)
