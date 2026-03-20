from __future__ import annotations

import numpy as np

from minefield_rl.constants import ACTION_DELTAS


def normalize_heatmap(counts: np.ndarray) -> np.ndarray:
    counts = counts.astype(np.float32)
    if counts.max() <= 0:
        return np.zeros_like(counts)
    return counts / counts.max()


def visit_counts_to_cell_heatmap(
    agent_pos: tuple[int, int],
    visit_counts: np.ndarray,
    grid_shape: tuple[int, int],
) -> np.ndarray:
    heatmap = np.zeros(grid_shape, dtype=np.float32)
    for action, count in enumerate(visit_counts):
        row = agent_pos[0] + int(ACTION_DELTAS[action][0])
        col = agent_pos[1] + int(ACTION_DELTAS[action][1])
        if 0 <= row < grid_shape[0] and 0 <= col < grid_shape[1]:
            heatmap[row, col] += float(count)
    return normalize_heatmap(heatmap)
