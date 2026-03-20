from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class PPOSequenceBatch:
    obs: np.ndarray
    health: np.ndarray
    actions: np.ndarray
    old_log_probs: np.ndarray
    old_values: np.ndarray
    returns: np.ndarray
    advantages: np.ndarray
    episode_starts: np.ndarray
    init_hidden: np.ndarray


class RecurrentRolloutBuffer:
    def __init__(
        self,
        rollout_steps: int,
        num_envs: int,
        obs_shape: tuple[int, ...],
        gru_layers: int,
        hidden_dim: int,
        sequence_length: int,
    ) -> None:
        if rollout_steps % sequence_length != 0:
            raise ValueError("rollout_steps must be divisible by sequence_length for recurrent PPO minibatching")
        self.rollout_steps = rollout_steps
        self.num_envs = num_envs
        self.obs_shape = obs_shape
        self.gru_layers = gru_layers
        self.hidden_dim = hidden_dim
        self.sequence_length = sequence_length

        self.obs = np.zeros((rollout_steps, num_envs, *obs_shape), dtype=np.float32)
        self.health = np.zeros((rollout_steps, num_envs), dtype=np.float32)
        self.actions = np.zeros((rollout_steps, num_envs), dtype=np.int64)
        self.log_probs = np.zeros((rollout_steps, num_envs), dtype=np.float32)
        self.rewards = np.zeros((rollout_steps, num_envs), dtype=np.float32)
        self.dones = np.zeros((rollout_steps, num_envs), dtype=np.float32)
        self.values = np.zeros((rollout_steps, num_envs), dtype=np.float32)
        self.episode_starts = np.zeros((rollout_steps, num_envs), dtype=np.float32)
        self.hiddens = np.zeros((rollout_steps, gru_layers, num_envs, hidden_dim), dtype=np.float32)
        self.advantages = np.zeros((rollout_steps, num_envs), dtype=np.float32)
        self.returns = np.zeros((rollout_steps, num_envs), dtype=np.float32)

    def add(
        self,
        step: int,
        obs: np.ndarray,
        health: np.ndarray,
        actions: np.ndarray,
        log_probs: np.ndarray,
        rewards: np.ndarray,
        dones: np.ndarray,
        values: np.ndarray,
        episode_starts: np.ndarray,
        hiddens: np.ndarray,
    ) -> None:
        self.obs[step] = obs
        self.health[step] = health
        self.actions[step] = actions
        self.log_probs[step] = log_probs
        self.rewards[step] = rewards
        self.dones[step] = dones
        self.values[step] = values
        self.episode_starts[step] = episode_starts
        self.hiddens[step] = hiddens

    def compute_returns_and_advantages(
        self,
        last_values: np.ndarray,
        gamma: float,
        gae_lambda: float,
    ) -> None:
        last_advantage = np.zeros(self.num_envs, dtype=np.float32)
        for step in range(self.rollout_steps - 1, -1, -1):
            if step == self.rollout_steps - 1:
                next_values = last_values
            else:
                next_values = self.values[step + 1]
            next_non_terminal = 1.0 - self.dones[step]
            delta = self.rewards[step] + gamma * next_values * next_non_terminal - self.values[step]
            last_advantage = delta + gamma * gae_lambda * next_non_terminal * last_advantage
            self.advantages[step] = last_advantage
        self.returns = self.advantages + self.values

    def iterate_minibatches(
        self,
        num_minibatches: int,
        update_epochs: int,
        normalize_advantages: bool = True,
    ):
        chunk_keys = [
            (env_index, start)
            for env_index in range(self.num_envs)
            for start in range(0, self.rollout_steps, self.sequence_length)
        ]
        chunk_count = len(chunk_keys)
        if chunk_count % num_minibatches != 0:
            raise ValueError("num_minibatches must divide the total recurrent chunk count")

        advantages = np.array(self.advantages, copy=True)
        if normalize_advantages:
            advantages = (advantages - advantages.mean()) / max(advantages.std(), 1e-8)

        minibatch_size = chunk_count // num_minibatches
        for _ in range(update_epochs):
            np.random.shuffle(chunk_keys)
            for start_index in range(0, chunk_count, minibatch_size):
                selected = chunk_keys[start_index : start_index + minibatch_size]
                obs_batch = []
                health_batch = []
                actions_batch = []
                old_log_probs_batch = []
                old_values_batch = []
                returns_batch = []
                advantages_batch = []
                episode_starts_batch = []
                init_hidden_batch = []

                for env_index, seq_start in selected:
                    seq_slice = slice(seq_start, seq_start + self.sequence_length)
                    obs_batch.append(self.obs[seq_slice, env_index])
                    health_batch.append(self.health[seq_slice, env_index])
                    actions_batch.append(self.actions[seq_slice, env_index])
                    old_log_probs_batch.append(self.log_probs[seq_slice, env_index])
                    old_values_batch.append(self.values[seq_slice, env_index])
                    returns_batch.append(self.returns[seq_slice, env_index])
                    advantages_batch.append(advantages[seq_slice, env_index])
                    episode_starts_batch.append(self.episode_starts[seq_slice, env_index])
                    init_hidden_batch.append(self.hiddens[seq_start, :, env_index, :])

                yield PPOSequenceBatch(
                    obs=np.stack(obs_batch, axis=0),
                    health=np.stack(health_batch, axis=0),
                    actions=np.stack(actions_batch, axis=0),
                    old_log_probs=np.stack(old_log_probs_batch, axis=0),
                    old_values=np.stack(old_values_batch, axis=0),
                    returns=np.stack(returns_batch, axis=0),
                    advantages=np.stack(advantages_batch, axis=0),
                    episode_starts=np.stack(episode_starts_batch, axis=0),
                    init_hidden=np.stack(init_hidden_batch, axis=1),
                )
