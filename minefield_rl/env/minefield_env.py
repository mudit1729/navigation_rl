from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from minefield_rl.constants import (
    ACTION_DELTAS,
    ACTION_NAMES,
    CellType,
    NUM_ACTIONS,
    OBS_EMPTY,
    OBS_EXIT,
    OBS_MINE,
    OBS_OUTSIDE,
    OBS_WALL,
)
from minefield_rl.env.env_copy import EnvSnapshot, clone_snapshot
from minefield_rl.env.map_generator import MapGenerator


@dataclass(slots=True)
class RewardBreakdown:
    progress: float = 0.0
    health: float = 0.0
    action: float = 0.0
    terminal: float = 0.0

    @property
    def total(self) -> float:
        return self.progress + self.health + self.action + self.terminal


class MinefieldEnv(gym.Env[np.ndarray, int]):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 15}

    def __init__(
        self,
        size: int = 20,
        view_radius: int = 4,
        wall_density: float = 0.15,
        mine_density: float = 0.12,
        max_health: int = 3,
        max_steps: int | None = None,
        max_steps_factor: int = 5,
        generation_max_attempts: int = 10,
        endpoint_clear_radius: int = 1,
        blocked_fraction_target: float | None = None,
        dispersion: str = "clustered",
        revisit_window: int = 10,
        reward_clip: tuple[float, float] = (-2.0, 2.0),
        progress_scale: float = 0.1,
        invalid_move_penalty: float = 0.05,
        living_penalty: float = 0.005,
        mine_hit_penalty: float = 1.0,
        success_reward: float = 10.0,
        death_penalty: float = 5.0,
        timeout_penalty: float = 1.0,
        revisit_penalty: float = 0.03,
        loop_penalty: float = 0.08,
        oscillation_penalty: float = 0.12,
        episode_profiles: list[dict[str, Any]] | None = None,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        self.size = size
        self.view_radius = view_radius
        self.max_health = max_health
        self.max_steps_override = max_steps
        self.base_max_steps_override = max_steps
        self.max_steps_factor = max_steps_factor
        self.base_wall_density = wall_density
        self.base_mine_density = mine_density
        self.base_generation_max_attempts = generation_max_attempts
        self.base_endpoint_clear_radius = endpoint_clear_radius
        self.base_blocked_fraction_target = blocked_fraction_target
        self.base_dispersion = dispersion
        self.base_revisit_window = revisit_window
        self.base_max_steps = int(max_steps if max_steps is not None else max_steps_factor * size * size)
        self.generation_max_attempts = generation_max_attempts
        self.max_steps = self.base_max_steps
        self.revisit_window = revisit_window
        self.reward_clip = reward_clip
        self.progress_scale = progress_scale
        self.invalid_move_penalty = invalid_move_penalty
        self.living_penalty = living_penalty
        self.mine_hit_penalty = mine_hit_penalty
        self.success_reward = success_reward
        self.death_penalty = death_penalty
        self.timeout_penalty = timeout_penalty
        self.revisit_penalty = revisit_penalty
        self.loop_penalty = loop_penalty
        self.oscillation_penalty = oscillation_penalty
        self.episode_profiles = [dict(profile) for profile in (episode_profiles or [])]
        self.base_seed = int(seed if seed is not None else np.random.SeedSequence().entropy)

        self.generator = MapGenerator(
            size=size,
            wall_density=wall_density,
            mine_density=mine_density,
            max_attempts=generation_max_attempts,
            endpoint_clear_radius=endpoint_clear_radius,
            blocked_fraction_target=blocked_fraction_target,
            path_max_health=max_health,
            dispersion=dispersion,
        )
        window = 2 * view_radius + 1
        self.observation_space = spaces.Box(
            low=0.0,
            high=4.0,
            shape=(2, window, window),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(NUM_ACTIONS)

        self.episode_index = 0
        self.episode_seed = self.base_seed
        self.grid = np.zeros((size, size), dtype=np.int8)
        self.start_pos = (0, 0)
        self.exit_pos = (size - 1, size - 1)
        self.agent_pos = self.start_pos
        self.health = max_health
        self.steps = 0
        self.prev_distance = 0.0
        self.total_reward_raw = 0.0
        self.total_reward_clipped = 0.0
        self.mine_hits = 0
        self.outcome: str | None = None
        self.terminated = False
        self.truncated = False
        self.recent_positions: deque[tuple[int, int]] = deque(maxlen=revisit_window)
        self.trajectory: list[tuple[int, int]] = []
        self.trajectory_counts = np.zeros((size, size), dtype=np.int32)
        self.info_cache: dict[str, Any] = {}
        self._rng = np.random.default_rng(self.base_seed)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "MinefieldEnv":
        env_cfg = config.get("env", config)
        reward_cfg = env_cfg.get("reward", {})
        return cls(
            size=env_cfg.get("size", 20),
            view_radius=env_cfg.get("view_radius", 4),
            wall_density=env_cfg.get("wall_density", 0.15),
            mine_density=env_cfg.get("mine_density", 0.12),
            max_health=env_cfg.get("max_health", 3),
            max_steps=env_cfg.get("max_steps"),
            max_steps_factor=env_cfg.get("max_steps_factor", 5),
            generation_max_attempts=env_cfg.get("generation_max_attempts", 10),
            endpoint_clear_radius=env_cfg.get("endpoint_clear_radius", 1),
            blocked_fraction_target=env_cfg.get("blocked_fraction_target"),
            dispersion=env_cfg.get("dispersion", "clustered"),
            revisit_window=env_cfg.get("revisit_window", 10),
            reward_clip=tuple(reward_cfg.get("clip", [-2.0, 2.0])),
            progress_scale=reward_cfg.get("progress_scale", 0.1),
            invalid_move_penalty=reward_cfg.get("invalid_move_penalty", 0.05),
            living_penalty=reward_cfg.get("living_penalty", 0.005),
            mine_hit_penalty=reward_cfg.get("mine_hit_penalty", 1.0),
            success_reward=reward_cfg.get("success_reward", 10.0),
            death_penalty=reward_cfg.get("death_penalty", 5.0),
            timeout_penalty=reward_cfg.get("timeout_penalty", 1.0),
            revisit_penalty=reward_cfg.get("revisit_penalty", 0.03),
            loop_penalty=reward_cfg.get("loop_penalty", 0.08),
            oscillation_penalty=reward_cfg.get("oscillation_penalty", 0.12),
            episode_profiles=env_cfg.get("episode_profiles"),
            seed=env_cfg.get("seed"),
        )

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self.episode_index += 1
        episode_seed = int(seed if seed is not None else self._rng.integers(0, 2**31 - 1))
        self.episode_seed = episode_seed
        episode_rng = np.random.default_rng(episode_seed)
        active_profile = self._apply_episode_profile(episode_rng)

        spec = self.generator.generate(episode_rng, episode_seed)
        self.grid = spec.grid
        self.start_pos = spec.start_pos
        self.exit_pos = spec.exit_pos
        self.agent_pos = self.start_pos
        self.health = self.max_health
        self.steps = 0
        self.prev_distance = self._distance(self.agent_pos, self.exit_pos)
        self.total_reward_raw = 0.0
        self.total_reward_clipped = 0.0
        self.mine_hits = 0
        self.outcome = None
        self.terminated = False
        self.truncated = False
        self.recent_positions = deque([self.agent_pos], maxlen=self.revisit_window)
        self.trajectory = [self.agent_pos]
        self.trajectory_counts.fill(0)
        self.trajectory_counts[self.agent_pos] = 1
        self.info_cache = {
            "map_seed": self.episode_seed,
            "generation_attempts": spec.generation_attempts,
            "used_fallback_map": spec.used_fallback,
            "profile_name": active_profile["name"],
            "wall_density": active_profile["wall_density"],
            "mine_density": active_profile["mine_density"],
            "blocked_fraction_target": active_profile.get("blocked_fraction_target"),
            "dispersion": active_profile["dispersion"],
            "episode_max_steps": active_profile["max_steps"],
        }
        return self._get_observation(), self._get_info(RewardBreakdown())

    def _apply_episode_profile(self, episode_rng: np.random.Generator) -> dict[str, Any]:
        profile_name = "default"
        wall_density = self.base_wall_density
        mine_density = self.base_mine_density
        generation_max_attempts = self.base_generation_max_attempts
        blocked_fraction_target = self.base_blocked_fraction_target
        dispersion = self.base_dispersion
        revisit_window = self.base_revisit_window
        max_steps = self.base_max_steps

        if self.episode_profiles:
            profile_index = int(episode_rng.integers(0, len(self.episode_profiles)))
            profile = dict(self.episode_profiles[profile_index])
            profile_name = str(profile.get("name", f"profile_{profile_index}"))
            wall_density = float(profile.get("wall_density", wall_density))
            mine_density = float(profile.get("mine_density", mine_density))
            generation_max_attempts = int(profile.get("generation_max_attempts", generation_max_attempts))
            blocked_fraction_target = profile.get("blocked_fraction_target", blocked_fraction_target)
            dispersion = str(profile.get("dispersion", dispersion))
            revisit_window = int(profile.get("revisit_window", revisit_window))
            max_steps = int(profile.get("max_steps", max_steps))

        self.generator.wall_density = wall_density
        self.generator.mine_density = mine_density
        self.generator.max_attempts = generation_max_attempts
        self.generator.endpoint_clear_radius = self.base_endpoint_clear_radius
        self.generator.blocked_fraction_target = blocked_fraction_target
        self.generator.dispersion = dispersion
        self.generator.path_max_health = self.max_health
        self.generation_max_attempts = generation_max_attempts
        self.revisit_window = revisit_window
        self.max_steps = max_steps

        return {
            "name": profile_name,
            "wall_density": wall_density,
            "mine_density": mine_density,
            "generation_max_attempts": generation_max_attempts,
            "blocked_fraction_target": blocked_fraction_target,
            "dispersion": dispersion,
            "revisit_window": revisit_window,
            "max_steps": max_steps,
        }

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self.terminated or self.truncated:
            raise RuntimeError("Episode is done. Call reset() before step().")

        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action index: {action}")

        self.steps += 1
        reward = RewardBreakdown()
        invalid_move = False
        mine_hit = False
        loop_penalty_applied = False
        oscillation_penalty_applied = False

        current_pos = self.agent_pos
        next_pos = self._candidate_position(current_pos, action)
        if not self._is_valid_move(current_pos, next_pos):
            next_pos = current_pos
            invalid_move = True
            reward.action -= self.invalid_move_penalty

        self.agent_pos = next_pos
        reward.progress += self.progress_scale * (self.prev_distance - self._distance(next_pos, self.exit_pos)) / self.size
        self.prev_distance = self._distance(next_pos, self.exit_pos)

        reward.action -= self.living_penalty
        recent_positions = list(self.recent_positions)
        if next_pos in recent_positions:
            reward.action -= self.revisit_penalty
        if recent_positions.count(next_pos) >= 2:
            reward.action -= self.loop_penalty
            loop_penalty_applied = True
        if (
            len(recent_positions) >= 3
            and recent_positions[-3] == current_pos
            and recent_positions[-2] == next_pos
            and recent_positions[-1] == current_pos
        ):
            reward.action -= self.oscillation_penalty
            oscillation_penalty_applied = True

        cell = CellType(self.grid[next_pos])
        if cell == CellType.MINE:
            self.health -= 1
            self.mine_hits += 1
            mine_hit = True
            reward.health -= self.mine_hit_penalty

        self.recent_positions.append(next_pos)
        self.trajectory.append(next_pos)
        self.trajectory_counts[next_pos] += 1

        if next_pos == self.exit_pos:
            self.terminated = True
            self.outcome = "success"
            reward.terminal += self.success_reward
        elif self.health <= 0:
            self.terminated = True
            self.outcome = "death"
            reward.terminal -= self.death_penalty
        elif self.steps >= self.max_steps:
            self.truncated = True
            self.outcome = "timeout"
            reward.terminal -= self.timeout_penalty

        raw_reward = reward.total
        clipped_reward = float(np.clip(raw_reward, self.reward_clip[0], self.reward_clip[1]))
        self.total_reward_raw += raw_reward
        self.total_reward_clipped += clipped_reward

        info = self._get_info(reward)
        info.update(
            {
                "invalid_move": invalid_move,
                "mine_hit": mine_hit,
                "loop_penalty_applied": loop_penalty_applied,
                "oscillation_penalty_applied": oscillation_penalty_applied,
                "action_name": ACTION_NAMES[action],
            }
        )
        return self._get_observation(), clipped_reward, self.terminated, self.truncated, info

    def _get_observation(self) -> np.ndarray:
        diameter = 2 * self.view_radius + 1
        obs = np.zeros((2, diameter, diameter), dtype=np.float32)
        center_row, center_col = self.agent_pos

        for view_row, row_offset in enumerate(range(-self.view_radius, self.view_radius + 1)):
            for view_col, col_offset in enumerate(range(-self.view_radius, self.view_radius + 1)):
                target_row = center_row + row_offset
                target_col = center_col + col_offset

                if not (0 <= target_row < self.size and 0 <= target_col < self.size):
                    obs[0, view_row, view_col] = OBS_OUTSIDE
                    obs[1, view_row, view_col] = 0.0
                    continue

                obs[1, view_row, view_col] = 1.0
                distance = np.hypot(row_offset, col_offset)
                if distance > self.view_radius:
                    obs[0, view_row, view_col] = OBS_OUTSIDE
                    continue

                cell = CellType(self.grid[target_row, target_col])
                if cell == CellType.WALL:
                    obs[0, view_row, view_col] = OBS_WALL
                elif cell == CellType.MINE:
                    obs[0, view_row, view_col] = OBS_MINE
                elif cell == CellType.EXIT:
                    obs[0, view_row, view_col] = OBS_EXIT
                else:
                    obs[0, view_row, view_col] = OBS_EMPTY
        return obs

    def _get_info(self, reward: RewardBreakdown) -> dict[str, Any]:
        info = {
            "health": self.health,
            "max_health": self.max_health,
            "steps": self.steps,
            "max_steps": self.max_steps,
            "agent_pos": self.agent_pos,
            "start_pos": self.start_pos,
            "exit_pos": self.exit_pos,
            "map_seed": self.episode_seed,
            "mine_hits": self.mine_hits,
            "outcome": self.outcome,
            "episode_reward_raw": self.total_reward_raw,
            "episode_reward_clipped": self.total_reward_clipped,
            "reward_terms": {
                "progress": reward.progress,
                "health": reward.health,
                "action": reward.action,
                "terminal": reward.terminal,
                "raw_total": reward.total,
            },
            "generation_attempts": self.info_cache.get("generation_attempts", 1),
            "used_fallback_map": self.info_cache.get("used_fallback_map", False),
            "profile_name": self.info_cache.get("profile_name", "default"),
            "wall_density": self.info_cache.get("wall_density", self.generator.wall_density),
            "mine_density": self.info_cache.get("mine_density", self.generator.mine_density),
            "blocked_fraction_target": self.info_cache.get("blocked_fraction_target", self.generator.blocked_fraction_target),
            "dispersion": self.info_cache.get("dispersion", self.generator.dispersion),
            "episode_max_steps": self.info_cache.get("episode_max_steps", self.max_steps),
        }
        self.info_cache.update(info)
        return info

    def _candidate_position(self, current_pos: tuple[int, int], action: int) -> tuple[int, int]:
        delta_row, delta_col = ACTION_DELTAS[action]
        return current_pos[0] + int(delta_row), current_pos[1] + int(delta_col)

    def _is_valid_move(
        self, current_pos: tuple[int, int], next_pos: tuple[int, int]
    ) -> bool:
        next_row, next_col = next_pos
        if not (0 <= next_row < self.size and 0 <= next_col < self.size):
            return False
        if CellType(self.grid[next_row, next_col]) == CellType.WALL:
            return False

        d_row = next_row - current_pos[0]
        d_col = next_col - current_pos[1]
        if d_row != 0 and d_col != 0:
            side_a = (current_pos[0] + d_row, current_pos[1])
            side_b = (current_pos[0], current_pos[1] + d_col)
            for row, col in (side_a, side_b):
                if not (0 <= row < self.size and 0 <= col < self.size):
                    return False
                if CellType(self.grid[row, col]) == CellType.WALL:
                    return False
        return True

    def _distance(self, left: tuple[int, int], right: tuple[int, int]) -> float:
        return float(np.hypot(left[0] - right[0], left[1] - right[1]))

    def snapshot(self) -> EnvSnapshot:
        return EnvSnapshot(
            grid=np.array(self.grid, copy=True),
            start_pos=tuple(self.start_pos),
            exit_pos=tuple(self.exit_pos),
            agent_pos=tuple(self.agent_pos),
            health=int(self.health),
            steps=int(self.steps),
            max_steps=int(self.max_steps),
            prev_distance=float(self.prev_distance),
            episode_seed=int(self.episode_seed),
            total_reward_raw=float(self.total_reward_raw),
            total_reward_clipped=float(self.total_reward_clipped),
            mine_hits=int(self.mine_hits),
            recent_positions=list(self.recent_positions),
            revisit_window=int(self.revisit_window),
            wall_density=float(self.generator.wall_density),
            mine_density=float(self.generator.mine_density),
            generation_max_attempts=int(self.generation_max_attempts),
            blocked_fraction_target=(
                None
                if self.generator.blocked_fraction_target is None
                else float(self.generator.blocked_fraction_target)
            ),
            dispersion=str(self.generator.dispersion),
            trajectory=list(self.trajectory),
            trajectory_counts=np.array(self.trajectory_counts, copy=True),
            outcome=self.outcome,
            terminated=bool(self.terminated),
            truncated=bool(self.truncated),
            info_cache=dict(self.info_cache),
        )

    def restore(self, snapshot: EnvSnapshot) -> None:
        state = clone_snapshot(snapshot)
        self.grid = state.grid
        self.start_pos = state.start_pos
        self.exit_pos = state.exit_pos
        self.agent_pos = state.agent_pos
        self.health = state.health
        self.steps = state.steps
        self.max_steps = state.max_steps
        self.prev_distance = state.prev_distance
        self.episode_seed = state.episode_seed
        self.total_reward_raw = state.total_reward_raw
        self.total_reward_clipped = state.total_reward_clipped
        self.mine_hits = state.mine_hits
        self.revisit_window = state.revisit_window
        self.generation_max_attempts = state.generation_max_attempts
        self.generator.wall_density = state.wall_density
        self.generator.mine_density = state.mine_density
        self.generator.max_attempts = state.generation_max_attempts
        self.generator.blocked_fraction_target = state.blocked_fraction_target
        self.generator.dispersion = state.dispersion
        self.recent_positions = deque(state.recent_positions, maxlen=self.revisit_window)
        self.trajectory = state.trajectory
        self.trajectory_counts = state.trajectory_counts
        self.outcome = state.outcome
        self.terminated = state.terminated
        self.truncated = state.truncated
        self.info_cache = state.info_cache

    def copy(self) -> "MinefieldEnv":
        env = MinefieldEnv(
            size=self.size,
            view_radius=self.view_radius,
            wall_density=self.generator.wall_density,
            mine_density=self.generator.mine_density,
            max_health=self.max_health,
            max_steps=self.max_steps_override,
            max_steps_factor=self.max_steps_factor,
            generation_max_attempts=self.base_generation_max_attempts,
            endpoint_clear_radius=self.base_endpoint_clear_radius,
            blocked_fraction_target=self.generator.blocked_fraction_target,
            dispersion=self.generator.dispersion,
            revisit_window=self.base_revisit_window,
            reward_clip=self.reward_clip,
            progress_scale=self.progress_scale,
            invalid_move_penalty=self.invalid_move_penalty,
            living_penalty=self.living_penalty,
            mine_hit_penalty=self.mine_hit_penalty,
            success_reward=self.success_reward,
            death_penalty=self.death_penalty,
            timeout_penalty=self.timeout_penalty,
            revisit_penalty=self.revisit_penalty,
            loop_penalty=self.loop_penalty,
            oscillation_penalty=self.oscillation_penalty,
            episode_profiles=self.episode_profiles,
            seed=self.base_seed,
        )
        env.restore(self.snapshot())
        return env

    def visible_mask(self) -> np.ndarray:
        mask = np.zeros((self.size, self.size), dtype=bool)
        row_center, col_center = self.agent_pos
        for row in range(self.size):
            for col in range(self.size):
                if np.hypot(row - row_center, col - col_center) <= self.view_radius:
                    mask[row, col] = True
        return mask

    def full_state(self) -> dict[str, Any]:
        return {
            "grid": np.array(self.grid, copy=True),
            "agent_pos": self.agent_pos,
            "start_pos": self.start_pos,
            "exit_pos": self.exit_pos,
            "health": self.health,
            "steps": self.steps,
            "max_steps": self.max_steps,
            "trajectory": list(self.trajectory),
            "trajectory_counts": np.array(self.trajectory_counts, copy=True),
            "visible_mask": self.visible_mask(),
            "map_seed": self.episode_seed,
            "outcome": self.outcome,
        }
