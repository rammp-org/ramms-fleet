"""Self-supervised collision labels computed from a rover's recorded run.

Kept free of MuJoCo and torch imports so the learning side can relabel data
cheaply inside every ClientApp process.
"""

from __future__ import annotations

import numpy as np


def bump_onsets(bump: np.ndarray, episode: np.ndarray) -> np.ndarray:
    """Rising edges of `bump`, restarting at every episode boundary."""
    previous = np.concatenate([[False], bump[:-1]])
    same_episode = np.concatenate([[False], episode[1:] == episode[:-1]])
    return bump & ~(previous & same_episode)


def collision_labels(bump: np.ndarray, episode: np.ndarray, horizon_steps: int) -> np.ndarray:
    """Marks step t positive when a bump onset occurs in (t, t + horizon_steps].

    Labels never look across an episode boundary.
    """
    labels = np.zeros(len(bump), dtype=bool)
    for t in np.flatnonzero(bump_onsets(bump, episode)):
        window = slice(max(0, t - horizon_steps), t)
        labels[window] |= episode[window] == episode[t]
    return labels


def distance_labels(
    bump: np.ndarray, episode: np.ndarray, pose: np.ndarray, horizon_m: float, max_steps: int = 100
) -> np.ndarray:
    """Marks step t positive when a bump onset follows within `horizon_m` of travel.

    Travel is the path length of the recorded (x, y) positions. Unlike a time
    horizon, this means the same thing for a slow and a fast rover. The window
    is also capped at `max_steps` so a rover sitting still before a collision
    does not label a long stretch of time. Labels never cross an episode boundary.
    """
    step = np.linalg.norm(np.diff(pose[:, :2], axis=0), axis=1)
    step = np.concatenate([[0.0], step])
    step[np.concatenate([[True], episode[1:] != episode[:-1]])] = 0.0
    travelled = np.cumsum(step)

    labels = np.zeros(len(bump), dtype=bool)
    for t in np.flatnonzero(bump_onsets(bump, episode)):
        # The tolerance keeps a step exactly horizon_m back despite accumulated float error.
        start = int(np.searchsorted(travelled, travelled[t] - horizon_m - 1e-9, side="left"))
        window = slice(max(start, t - max_steps, 0), t)
        labels[window] |= episode[window] == episode[t]
    return labels
