from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from minefield_rl.constants import CellType


@dataclass(slots=True)
class MapSpec:
    grid: np.ndarray
    start_pos: tuple[int, int]
    exit_pos: tuple[int, int]
    seed: int
    generation_attempts: int
    used_fallback: bool


class MapGenerator:
    def __init__(
        self,
        size: int,
        wall_density: float = 0.15,
        mine_density: float = 0.12,
        max_attempts: int = 10,
        endpoint_clear_radius: int = 1,
        blocked_fraction_target: float | None = None,
        path_max_health: int = 3,
        dispersion: str = "clustered",
    ) -> None:
        self.size = size
        self.wall_density = wall_density
        self.mine_density = mine_density
        self.max_attempts = max_attempts
        self.endpoint_clear_radius = max(0, int(endpoint_clear_radius))
        self.blocked_fraction_target = None if blocked_fraction_target is None else float(blocked_fraction_target)
        self.path_max_health = max(1, int(path_max_health))
        if dispersion not in ("clustered", "dispersed"):
            raise ValueError(f"dispersion must be 'clustered' or 'dispersed', got {dispersion!r}")
        self.dispersion = dispersion

    def generate(self, rng: np.random.Generator, seed: int) -> MapSpec:
        for attempt in range(1, self.max_attempts + 1):
            grid = self._generate_candidate(rng)
            start_pos, exit_pos = self._sample_endpoints(grid, rng)
            if start_pos == exit_pos:
                continue

            grid = np.array(grid, copy=True)
            self._clear_endpoint_zone(grid, start_pos)
            self._clear_endpoint_zone(grid, exit_pos)
            self._adjust_blocked_fraction(grid, rng, start_pos, exit_pos)
            grid[start_pos] = CellType.START
            grid[exit_pos] = CellType.EXIT

            if self._has_safe_path(grid, start_pos, exit_pos):
                return MapSpec(
                    grid=grid,
                    start_pos=start_pos,
                    exit_pos=exit_pos,
                    seed=seed,
                    generation_attempts=attempt,
                    used_fallback=False,
                )

        safe_grid, start_pos, exit_pos = self._fallback_template(seed)
        return MapSpec(
            grid=safe_grid,
            start_pos=start_pos,
            exit_pos=exit_pos,
            seed=seed,
            generation_attempts=self.max_attempts,
            used_fallback=True,
        )

    def _generate_candidate(self, rng: np.random.Generator) -> np.ndarray:
        grid = np.full((self.size, self.size), CellType.EMPTY, dtype=np.int8)

        if self.dispersion == "dispersed":
            wall_mask = rng.random((self.size, self.size)) < self.wall_density
            grid[wall_mask] = CellType.WALL
            non_wall = np.argwhere(grid != CellType.WALL)
            mine_target = min(len(non_wall), int(round(self.size * self.size * self.mine_density)))
            if mine_target > 0 and len(non_wall) > 0:
                chosen = non_wall[rng.permutation(len(non_wall))[:mine_target]]
                grid[chosen[:, 0], chosen[:, 1]] = CellType.MINE
        else:
            wall_noise = rng.random((self.size, self.size)) < self.wall_density
            grid[wall_noise] = CellType.WALL
            grid = self._smooth_walls(grid, rng)

            mine_mask = self._grow_mines(rng)
            grid[(grid != CellType.WALL) & mine_mask] = CellType.MINE

        # Keep borders slightly more navigable.
        grid[0, :] = np.where(grid[0, :] == CellType.WALL, CellType.EMPTY, grid[0, :])
        grid[:, 0] = np.where(grid[:, 0] == CellType.WALL, CellType.EMPTY, grid[:, 0])
        grid[-1, :] = np.where(grid[-1, :] == CellType.WALL, CellType.EMPTY, grid[-1, :])
        grid[:, -1] = np.where(grid[:, -1] == CellType.WALL, CellType.EMPTY, grid[:, -1])
        return grid

    def _smooth_walls(self, grid: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        smoothed = np.array(grid, copy=True)
        for _ in range(2):
            next_grid = np.array(smoothed, copy=True)
            for row in range(self.size):
                for col in range(self.size):
                    wall_neighbors = self._neighbor_count(smoothed, row, col, CellType.WALL)
                    if wall_neighbors >= 5:
                        next_grid[row, col] = CellType.WALL
                    elif wall_neighbors <= 2:
                        next_grid[row, col] = CellType.EMPTY
                    elif rng.random() < 0.05:
                        next_grid[row, col] = CellType.WALL
            smoothed = next_grid
        return smoothed

    def _grow_mines(self, rng: np.random.Generator) -> np.ndarray:
        target = max(1, int(self.size * self.size * self.mine_density))
        mine_mask = np.zeros((self.size, self.size), dtype=bool)
        seeds = max(2, target // 8)
        frontier: list[tuple[int, int]] = []

        for _ in range(seeds):
            row = int(rng.integers(0, self.size))
            col = int(rng.integers(0, self.size))
            frontier.append((row, col))
            mine_mask[row, col] = True

        while mine_mask.sum() < target and frontier:
            row, col = frontier.pop(int(rng.integers(0, len(frontier))))
            for d_row, d_col in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                if mine_mask.sum() >= target:
                    break
                if rng.random() > 0.55:
                    continue
                new_row = row + d_row
                new_col = col + d_col
                if 0 <= new_row < self.size and 0 <= new_col < self.size and not mine_mask[new_row, new_col]:
                    mine_mask[new_row, new_col] = True
                    frontier.append((new_row, new_col))

        if mine_mask.sum() < target:
            remaining = np.argwhere(~mine_mask)
            rng.shuffle(remaining)
            for row, col in remaining[: target - int(mine_mask.sum())]:
                mine_mask[row, col] = True
        return mine_mask

    def _sample_endpoints(
        self, grid: np.ndarray, rng: np.random.Generator
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        top_left = [
            (row, col)
            for row in range(max(1, self.size // 2))
            for col in range(max(1, self.size // 2))
            if CellType(grid[row, col]) == CellType.EMPTY
        ]
        bottom_right = [
            (row, col)
            for row in range(self.size // 2, self.size)
            for col in range(self.size // 2, self.size)
            if CellType(grid[row, col]) == CellType.EMPTY
        ]
        if not top_left or not bottom_right:
            return (0, 0), (self.size - 1, self.size - 1)
        start_pos = top_left[int(rng.integers(0, len(top_left)))]
        exit_pos = bottom_right[int(rng.integers(0, len(bottom_right)))]
        return start_pos, exit_pos

    def _clear_endpoint_zone(self, grid: np.ndarray, center: tuple[int, int]) -> None:
        radius = self.endpoint_clear_radius
        if radius <= 0:
            return
        row_center, col_center = center
        for row in range(max(0, row_center - radius), min(self.size, row_center + radius + 1)):
            for col in range(max(0, col_center - radius), min(self.size, col_center + radius + 1)):
                grid[row, col] = CellType.EMPTY

    def _adjust_blocked_fraction(
        self,
        grid: np.ndarray,
        rng: np.random.Generator,
        start_pos: tuple[int, int],
        exit_pos: tuple[int, int],
    ) -> None:
        if self.blocked_fraction_target is None:
            return

        target_fraction = float(np.clip(self.blocked_fraction_target, 0.0, 1.0))
        protected = self._endpoint_protection_mask(start_pos, exit_pos)
        adjustable = ~protected
        adjustable_count = int(adjustable.sum())
        if adjustable_count <= 0:
            return

        target_blocked = min(int(round(target_fraction * grid.size)), adjustable_count)
        blocked_mask = (grid == CellType.WALL) | (grid == CellType.MINE)
        blocked_adjustable = blocked_mask & adjustable

        total_density = max(self.wall_density + self.mine_density, 1e-8)
        desired_wall_share = float(self.wall_density / total_density)
        target_walls = int(round(target_blocked * desired_wall_share))
        target_mines = target_blocked - target_walls

        wall_positions = np.argwhere((grid == CellType.WALL) & adjustable)
        mine_positions = np.argwhere((grid == CellType.MINE) & adjustable)

        if len(wall_positions) > target_walls:
            remove = wall_positions[rng.permutation(len(wall_positions))[: len(wall_positions) - target_walls]]
            grid[remove[:, 0], remove[:, 1]] = CellType.EMPTY
        if len(mine_positions) > target_mines:
            remove = mine_positions[rng.permutation(len(mine_positions))[: len(mine_positions) - target_mines]]
            grid[remove[:, 0], remove[:, 1]] = CellType.EMPTY

        empty_positions = np.argwhere((grid == CellType.EMPTY) & adjustable)
        current_walls = int(((grid == CellType.WALL) & adjustable).sum())
        missing_walls = max(0, target_walls - current_walls)
        if missing_walls > 0 and len(empty_positions) > 0:
            chosen = empty_positions[rng.permutation(len(empty_positions))[:missing_walls]]
            grid[chosen[:, 0], chosen[:, 1]] = CellType.WALL

        empty_positions = np.argwhere((grid == CellType.EMPTY) & adjustable)
        current_mines = int(((grid == CellType.MINE) & adjustable).sum())
        missing_mines = max(0, target_mines - current_mines)
        if missing_mines > 0 and len(empty_positions) > 0:
            chosen = empty_positions[rng.permutation(len(empty_positions))[:missing_mines]]
            grid[chosen[:, 0], chosen[:, 1]] = CellType.MINE

    def _endpoint_protection_mask(
        self,
        start_pos: tuple[int, int],
        exit_pos: tuple[int, int],
    ) -> np.ndarray:
        mask = np.zeros((self.size, self.size), dtype=bool)
        for center in (start_pos, exit_pos):
            row_center, col_center = center
            for row in range(max(0, row_center - self.endpoint_clear_radius), min(self.size, row_center + self.endpoint_clear_radius + 1)):
                for col in range(max(0, col_center - self.endpoint_clear_radius), min(self.size, col_center + self.endpoint_clear_radius + 1)):
                    mask[row, col] = True
        return mask

    def _has_safe_path(
        self, grid: np.ndarray, start_pos: tuple[int, int], exit_pos: tuple[int, int]
    ) -> bool:
        queue: deque[tuple[int, int, int]] = deque([(start_pos[0], start_pos[1], self.path_max_health)])
        seen = {(start_pos[0], start_pos[1], self.path_max_health)}

        while queue:
            row, col, health = queue.popleft()
            if (row, col) == exit_pos:
                return True
            for d_row, d_col in (
                (-1, 0),
                (1, 0),
                (0, -1),
                (0, 1),
                (-1, -1),
                (-1, 1),
                (1, -1),
                (1, 1),
            ):
                new_row = row + d_row
                new_col = col + d_col
                if not (0 <= new_row < self.size and 0 <= new_col < self.size):
                    continue
                cell = CellType(grid[new_row, new_col])
                if cell == CellType.WALL:
                    continue
                if d_row != 0 and d_col != 0 and self._diagonal_blocked(grid, row, col, d_row, d_col):
                    continue
                next_health = health
                if cell == CellType.MINE:
                    if health <= 1:
                        continue
                    next_health = health - 1
                state = (new_row, new_col, next_health)
                if state in seen:
                    continue
                seen.add(state)
                queue.append(state)
        return False

    def _diagonal_blocked(
        self, grid: np.ndarray, row: int, col: int, d_row: int, d_col: int
    ) -> bool:
        side_a = (row + d_row, col)
        side_b = (row, col + d_col)
        for check_row, check_col in (side_a, side_b):
            if not (0 <= check_row < self.size and 0 <= check_col < self.size):
                return True
            if CellType(grid[check_row, check_col]) == CellType.WALL:
                return True
        return False

    def _neighbor_count(self, grid: np.ndarray, row: int, col: int, target: CellType) -> int:
        count = 0
        for d_row in (-1, 0, 1):
            for d_col in (-1, 0, 1):
                if d_row == 0 and d_col == 0:
                    continue
                new_row = row + d_row
                new_col = col + d_col
                if not (0 <= new_row < self.size and 0 <= new_col < self.size):
                    count += 1 if target == CellType.WALL else 0
                    continue
                count += int(CellType(grid[new_row, new_col]) == target)
        return count

    def _fallback_template(self, seed: int) -> tuple[np.ndarray, tuple[int, int], tuple[int, int]]:
        grid = np.full((self.size, self.size), CellType.EMPTY, dtype=np.int8)
        for row in range(1, self.size - 1):
            if row % 4 == 0:
                grid[row, 1 : self.size - 2] = CellType.WALL
                gap = min(self.size - 2, 2 + (row * 3) % max(2, self.size - 3))
                grid[row, gap : gap + 2] = CellType.EMPTY

        for col in range(2, self.size - 2, 5):
            for row in range(1, self.size - 1):
                if grid[row, col] == CellType.EMPTY and abs(row - col) > 2:
                    grid[row, col] = CellType.MINE

        start_pos = (1, 1)
        exit_pos = (self.size - 2, self.size - 2)
        self._clear_endpoint_zone(grid, start_pos)
        self._clear_endpoint_zone(grid, exit_pos)
        grid[start_pos] = CellType.START
        grid[exit_pos] = CellType.EXIT
        return grid, start_pos, exit_pos
