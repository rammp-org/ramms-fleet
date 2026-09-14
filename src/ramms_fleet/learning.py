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

    @property
    def input_dim(self) -> int:
        return self.x_train.shape[1]


def load_rover(path: Path | str, history: int = 4, test_fraction: float = 0.2) -> RoverData:
    """Loads one rover's file as windowed samples with a chronological split.

    A sample at step t stacks the scaled features of steps t-history+1..t, all in
    the same episode, and is kept only if step t is valid. The last
    `test_fraction` of the run is held out, so test samples never sit next to
    training samples in time.
    """
    data = np.load(path)
    features = data["features"] / FEATURE_SCALE
    labels, valid, episode = data["label"], data["valid"], data["episode"]
    steps, dim = features.shape

    ends = np.arange(history - 1, steps)
    same_episode = episode[ends] == episode[ends - history + 1]
    ends = ends[valid[ends] & same_episode]
    windows = np.stack([features[ends - k] for k in range(history - 1, -1, -1)], axis=1)
    x = windows.reshape(len(ends), history * dim).astype(np.float32)
    y = labels[ends].astype(np.float32)

    split = ends < round(steps * (1 - test_fraction))
    return RoverData(x_train=x[split], y_train=y[split], x_test=x[~split], y_test=y[~split])


def make_model(input_dim: int, hidden: int = 64) -> nn.Module:
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
) -> float:
    """Trains in place and returns the mean loss of the last epoch.

    The positive class is re-weighted by this dataset's own negative/positive
    ratio. `proximal_mu` adds the FedProx penalty towards the starting weights.
    """
    generator = torch.Generator().manual_seed(seed)
    xt, yt = torch.from_numpy(x), torch.from_numpy(y)
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
            loss = loss_fn(model(xt[batch]).squeeze(1), yt[batch])
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
def predict(model: nn.Module, x: np.ndarray) -> np.ndarray:
    model.eval()
    return torch.sigmoid(model(torch.from_numpy(x)).squeeze(1)).numpy()


def evaluate(model: nn.Module, x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    scores = predict(model, x)
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
