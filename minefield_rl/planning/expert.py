from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Any

import numpy as np

from minefield_rl.constants import ACTION_DELTAS, CellType
from minefield_rl.env.minefield_env import MinefieldEnv


@dataclass(slots=True)
class ExpertPlan:
    actions: list[int]
    path: list[tuple[int, int]]
    expected_mine_hits: int
    total_cost: float


class ExpertPathPlanner:
    def __init__(self, mine_cost: float = 6.0) -> None:
        self.mine_cost = mine_cost

    def plan(self, env: MinefieldEnv) -> ExpertPlan | None:
        start_state = (env.agent_pos[0], env.agent_pos[1], env.health)
        goal = env.exit_pos
        frontier: list[tuple[float, float, tuple[int, int, int]]] = []
        heapq.heappush(frontier, (0.0, 0.0, start_state))

        came_from: dict[tuple[int, int, int], tuple[tuple[int, int, int], int]] = {}
        cost_so_far: dict[tuple[int, int, int], float] = {start_state: 0.0}

        best_goal_state: tuple[int, int, int] | None = None
        while frontier:
            _, current_cost, current_state = heapq.heappop(frontier)
            row, col, health = current_state
            if (row, col) == goal:
                best_goal_state = current_state
                break

            current_pos = (row, col)
            for action, (d_row, d_col) in enumerate(ACTION_DELTAS):
                next_pos = (row + int(d_row), col + int(d_col))
                if not env._is_valid_move(current_pos, next_pos):
                    continue

                cell = CellType(env.grid[next_pos])
                mine_hit = int(cell == CellType.MINE)
                next_health = health - mine_hit
                if next_health <= 0:
                    continue

                step_cost = 1.0 + self.mine_cost * mine_hit
                next_state = (next_pos[0], next_pos[1], next_health)
                new_cost = current_cost + step_cost
                if new_cost >= cost_so_far.get(next_state, float("inf")):
                    continue

                cost_so_far[next_state] = new_cost
                priority = new_cost + self._heuristic(next_pos, goal)
                heapq.heappush(frontier, (priority, new_cost, next_state))
                came_from[next_state] = (current_state, action)

        if best_goal_state is None:
            return None

        actions: list[int] = []
        path: list[tuple[int, int]] = [(best_goal_state[0], best_goal_state[1])]
        current = best_goal_state
        while current != start_state:
            previous, action = came_from[current]
            actions.append(action)
            path.append((previous[0], previous[1]))
            current = previous
        actions.reverse()
        path.reverse()
        expected_mine_hits = max(0, env.health - best_goal_state[2])
        return ExpertPlan(
            actions=actions,
            path=path,
            expected_mine_hits=expected_mine_hits,
            total_cost=cost_so_far[best_goal_state],
        )

    def _heuristic(self, current: tuple[int, int], goal: tuple[int, int]) -> float:
        delta_row = abs(goal[0] - current[0])
        delta_col = abs(goal[1] - current[1])
        diagonal = min(delta_row, delta_col)
        straight = max(delta_row, delta_col) - diagonal
        return 1.41421356237 * diagonal + straight
