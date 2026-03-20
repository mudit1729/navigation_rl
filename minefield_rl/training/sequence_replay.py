from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(slots=True)
class SequenceBatch:
    obs: np.ndarray
    health: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_obs: np.ndarray
    next_health: np.ndarray
    dones: np.ndarray
    mask: np.ndarray
    returns: np.ndarray


@dataclass(slots=True)
class PolicyBatch:
    obs: np.ndarray
    health: np.ndarray
    policy_targets: np.ndarray
    value_targets: np.ndarray
    mask: np.ndarray


class SequenceReplayBuffer:
    def __init__(self, capacity: int, sequence_length: int, gamma: float = 0.99) -> None:
        self.capacity = capacity
        self.sequence_length = sequence_length
        self.gamma = gamma
        self.sequences: deque[dict[str, np.ndarray]] = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self.sequences)

    def add_episode(self, transitions: list[dict[str, Any]]) -> None:
        if not transitions:
            return

        running_return = 0.0
        returns: list[float] = []
        for transition in reversed(transitions):
            running_return = float(transition["reward"]) + self.gamma * running_return
            returns.append(np.tanh(running_return / 10.0))
        returns.reverse()

        augmented = []
        for index, transition in enumerate(transitions):
            item = dict(transition)
            item["return"] = returns[index]
            augmented.append(item)

        for start in range(len(augmented)):
            stop = min(len(augmented), start + self.sequence_length)
            window = augmented[start:stop]
            self.sequences.append(self._pack_window(window))

    def sample(self, batch_size: int) -> SequenceBatch:
        if batch_size > len(self.sequences):
            raise ValueError("Not enough sequences in buffer")
        indices = np.random.choice(len(self.sequences), size=batch_size, replace=False)
        batch = [self.sequences[index] for index in indices]
        return SequenceBatch(
            obs=np.stack([item["obs"] for item in batch]),
            health=np.stack([item["health"] for item in batch]),
            actions=np.stack([item["actions"] for item in batch]),
            rewards=np.stack([item["rewards"] for item in batch]),
            next_obs=np.stack([item["next_obs"] for item in batch]),
            next_health=np.stack([item["next_health"] for item in batch]),
            dones=np.stack([item["dones"] for item in batch]),
            mask=np.stack([item["mask"] for item in batch]),
            returns=np.stack([item["returns"] for item in batch]),
        )

    def _pack_window(self, window: list[dict[str, Any]]) -> dict[str, np.ndarray]:
        obs_shape = window[0]["obs"].shape
        next_obs_shape = window[0]["next_obs"].shape
        seq_len = self.sequence_length
        obs = np.zeros((seq_len, *obs_shape), dtype=np.float32)
        health = np.zeros((seq_len,), dtype=np.float32)
        actions = np.zeros((seq_len,), dtype=np.int64)
        rewards = np.zeros((seq_len,), dtype=np.float32)
        next_obs = np.zeros((seq_len, *next_obs_shape), dtype=np.float32)
        next_health = np.zeros((seq_len,), dtype=np.float32)
        dones = np.ones((seq_len,), dtype=np.float32)
        mask = np.zeros((seq_len,), dtype=np.float32)
        returns = np.zeros((seq_len,), dtype=np.float32)

        for index, item in enumerate(window):
            obs[index] = item["obs"]
            health[index] = item["health"]
            actions[index] = item["action"]
            rewards[index] = item["reward"]
            next_obs[index] = item["next_obs"]
            next_health[index] = item["next_health"]
            dones[index] = item["done"]
            returns[index] = item["return"]
            mask[index] = 1.0

        return {
            "obs": obs,
            "health": health,
            "actions": actions,
            "rewards": rewards,
            "next_obs": next_obs,
            "next_health": next_health,
            "dones": dones,
            "mask": mask,
            "returns": returns,
        }


class PolicyReplayBuffer:
    def __init__(self, capacity: int, sequence_length: int, action_dim: int) -> None:
        self.capacity = capacity
        self.sequence_length = sequence_length
        self.action_dim = action_dim
        self.sequences: deque[dict[str, np.ndarray]] = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self.sequences)

    def add_episode(self, records: list[dict[str, Any]]) -> None:
        if not records:
            return
        for start in range(len(records)):
            stop = min(len(records), start + self.sequence_length)
            self.sequences.append(self._pack_window(records[start:stop]))

    def sample(self, batch_size: int) -> PolicyBatch:
        if batch_size > len(self.sequences):
            raise ValueError("Not enough policy sequences in buffer")
        indices = np.random.choice(len(self.sequences), size=batch_size, replace=False)
        batch = [self.sequences[index] for index in indices]
        return PolicyBatch(
            obs=np.stack([item["obs"] for item in batch]),
            health=np.stack([item["health"] for item in batch]),
            policy_targets=np.stack([item["policy_targets"] for item in batch]),
            value_targets=np.stack([item["value_targets"] for item in batch]),
            mask=np.stack([item["mask"] for item in batch]),
        )

    def _pack_window(self, window: list[dict[str, Any]]) -> dict[str, np.ndarray]:
        obs_shape = window[0]["obs"].shape
        seq_len = self.sequence_length
        obs = np.zeros((seq_len, *obs_shape), dtype=np.float32)
        health = np.zeros((seq_len,), dtype=np.float32)
        policy_targets = np.zeros((seq_len, self.action_dim), dtype=np.float32)
        value_targets = np.zeros((seq_len,), dtype=np.float32)
        mask = np.zeros((seq_len,), dtype=np.float32)

        for index, item in enumerate(window):
            obs[index] = item["obs"]
            health[index] = item["health"]
            policy_targets[index] = item["policy_target"]
            value_targets[index] = item["value_target"]
            mask[index] = 1.0

        return {
            "obs": obs,
            "health": health,
            "policy_targets": policy_targets,
            "value_targets": value_targets,
            "mask": mask,
        }
