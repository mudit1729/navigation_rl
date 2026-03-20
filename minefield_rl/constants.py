from __future__ import annotations

from enum import IntEnum

import numpy as np


class CellType(IntEnum):
    EMPTY = 0
    WALL = 1
    MINE = 2
    START = 3
    EXIT = 4


OBS_OUTSIDE = 0
OBS_EMPTY = 1
OBS_WALL = 2
OBS_MINE = 3
OBS_EXIT = 4


ACTION_DELTAS = np.asarray(
    [
        (-1, 0),   # N
        (-1, 1),   # NE
        (0, 1),    # E
        (1, 1),    # SE
        (1, 0),    # S
        (1, -1),   # SW
        (0, -1),   # W
        (-1, -1),  # NW
    ],
    dtype=np.int64,
)

ACTION_NAMES = ("N", "NE", "E", "SE", "S", "SW", "W", "NW")

NUM_ACTIONS = len(ACTION_NAMES)
