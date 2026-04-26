from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(slots=True)
class EnvSnapshot:
    grid: np.ndarray
    start_pos: tuple[int, int]
    exit_pos: tuple[int, int]
    agent_pos: tuple[int, int]
    health: int
    steps: int
    max_steps: int
    prev_distance: float
    episode_seed: int
    total_reward_raw: float
    total_reward_clipped: float
    mine_hits: int
    recent_positions: list[tuple[int, int]]
    revisit_window: int
    wall_density: float
    mine_density: float
    generation_max_attempts: int
    blocked_fraction_target: float | None
    dispersion: str
    trajectory: list[tuple[int, int]]
    trajectory_counts: np.ndarray
    outcome: str | None
    terminated: bool
    truncated: bool
    info_cache: dict[str, Any]


def clone_snapshot(snapshot: EnvSnapshot) -> EnvSnapshot:
    return EnvSnapshot(
        grid=np.array(snapshot.grid, copy=True),
        start_pos=tuple(snapshot.start_pos),
        exit_pos=tuple(snapshot.exit_pos),
        agent_pos=tuple(snapshot.agent_pos),
        health=int(snapshot.health),
        steps=int(snapshot.steps),
        max_steps=int(snapshot.max_steps),
        prev_distance=float(snapshot.prev_distance),
        episode_seed=int(snapshot.episode_seed),
        total_reward_raw=float(snapshot.total_reward_raw),
        total_reward_clipped=float(snapshot.total_reward_clipped),
        mine_hits=int(snapshot.mine_hits),
        recent_positions=list(snapshot.recent_positions),
        revisit_window=int(snapshot.revisit_window),
        wall_density=float(snapshot.wall_density),
        mine_density=float(snapshot.mine_density),
        generation_max_attempts=int(snapshot.generation_max_attempts),
        blocked_fraction_target=(
            None
            if snapshot.blocked_fraction_target is None
            else float(snapshot.blocked_fraction_target)
        ),
        dispersion=str(snapshot.dispersion),
        trajectory=list(snapshot.trajectory),
        trajectory_counts=np.array(snapshot.trajectory_counts, copy=True),
        outcome=snapshot.outcome,
        terminated=bool(snapshot.terminated),
        truncated=bool(snapshot.truncated),
        info_cache=dict(snapshot.info_cache),
    )
