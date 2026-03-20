from __future__ import annotations

import csv
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.distributions import Categorical
from torch.nn import functional as F
from torch.optim import Adam
from tqdm import trange

from minefield_rl.env.minefield_env import MinefieldEnv, RewardBreakdown
from minefield_rl.models.mcts import AlphaZeroMCTS
from minefield_rl.models.rppo import RecurrentPPOActorCritic
from minefield_rl.utils import ensure_dir


@dataclass(slots=True)
class GRPOTrajectory:
    obs: np.ndarray
    health: np.ndarray
    actions: np.ndarray
    old_log_probs: np.ndarray
    policy_targets: np.ndarray
    episode_starts: np.ndarray
    outcome: str
    reward_raw: float
    reward_clipped: float
    length: int
    health_remaining: int
    mine_hits: int
    map_seed: int
    group_index: int
    env_steps: int


def save_grpo_checkpoint(
    path: str | Path,
    model: RecurrentPPOActorCritic,
    optimizer: torch.optim.Optimizer,
    update: int,
    env_steps: int,
    episode: int,
    config: dict[str, Any],
) -> None:
    torch.save(
        {
            "agent": "ppo",
            "training_stage": "grpo",
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "update": update,
            "env_steps": env_steps,
            "episode": episode,
            "config": config,
        },
        Path(path),
    )


def _make_batch(
    trajectories: list[GRPOTrajectory],
    advantages: list[float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    max_len = max(len(trajectory.actions) for trajectory in trajectories)
    batch_size = len(trajectories)
    obs_shape = trajectories[0].obs.shape[1:]

    obs = np.zeros((batch_size, max_len, *obs_shape), dtype=np.float32)
    health = np.zeros((batch_size, max_len), dtype=np.float32)
    actions = np.zeros((batch_size, max_len), dtype=np.int64)
    old_log_probs = np.zeros((batch_size, max_len), dtype=np.float32)
    policy_targets = np.zeros((batch_size, max_len, trajectories[0].policy_targets.shape[-1]), dtype=np.float32)
    episode_starts = np.zeros((batch_size, max_len), dtype=np.float32)
    advantage_values = np.zeros((batch_size, max_len), dtype=np.float32)
    mask = np.zeros((batch_size, max_len), dtype=np.float32)

    for batch_index, (trajectory, advantage) in enumerate(zip(trajectories, advantages)):
        length = len(trajectory.actions)
        obs[batch_index, :length] = trajectory.obs
        health[batch_index, :length] = trajectory.health
        actions[batch_index, :length] = trajectory.actions
        old_log_probs[batch_index, :length] = trajectory.old_log_probs
        policy_targets[batch_index, :length] = trajectory.policy_targets
        episode_starts[batch_index, :length] = trajectory.episode_starts
        advantage_values[batch_index, :length] = advantage
        mask[batch_index, :length] = 1.0

    return obs, health, actions, old_log_probs, policy_targets, episode_starts, advantage_values, mask


def _compute_group_advantages(
    trajectories: list[GRPOTrajectory],
    group_rollouts: int,
    normalize_advantages: bool,
) -> list[float]:
    advantages: list[float] = [0.0 for _ in trajectories]
    for start_index in range(0, len(trajectories), group_rollouts):
        group = trajectories[start_index : start_index + group_rollouts]
        returns = np.asarray([trajectory.reward_raw for trajectory in group], dtype=np.float32)
        if normalize_advantages and len(group) > 1:
            std = float(returns.std())
            if std > 1e-6:
                normalized = (returns - float(returns.mean())) / std
            else:
                normalized = returns - float(returns.mean())
        else:
            normalized = returns
        for offset, value in enumerate(normalized.tolist()):
            advantages[start_index + offset] = float(value)
    return advantages


def _collect_trajectory(
    env: MinefieldEnv,
    model: RecurrentPPOActorCritic,
    search: AlphaZeroMCTS | None,
    device: torch.device,
    simulations: int,
    group_index: int,
) -> GRPOTrajectory:
    observation = env._get_observation()
    info = env._get_info(RewardBreakdown())
    hidden = model.init_hidden(1, device)
    hidden_mcts: np.ndarray | None = None
    terminated = False
    truncated = False

    obs_seq: list[np.ndarray] = []
    health_seq: list[float] = []
    action_seq: list[int] = []
    old_log_prob_seq: list[float] = []
    policy_targets_seq: list[np.ndarray] = []
    episode_starts_seq: list[float] = []

    while not (terminated or truncated):
        normalized_health = info["health"] / env.max_health
        obs_tensor = torch.from_numpy(observation).float().unsqueeze(0).to(device)
        health_tensor = torch.tensor([normalized_health], dtype=torch.float32, device=device)
        episode_start = torch.tensor([1.0 if info["steps"] == 0 else 0.0], dtype=torch.float32, device=device)

        with torch.no_grad():
            step_out = model.forward_step(
                obs_tensor,
                health_tensor,
                episode_start=episode_start,
                hidden=hidden,
            )
            step_logits = step_out.logits[0]
            policy = torch.softmax(step_logits, dim=-1).cpu().numpy().astype(np.float32)

        if search is not None and simulations > 0:
            result = search.search(
                env=env,
                observation=observation,
                health=info["health"],
                hidden_in=hidden_mcts,
                simulations=simulations,
                training=True,
            )
            action = int(result.chosen_action)
            policy_target = result.policy.astype(np.float32)
            hidden = torch.from_numpy(result.root_hidden).float().to(device)
            hidden_mcts = result.root_hidden
        else:
            with torch.no_grad():
                action_out = model.act(
                    obs_tensor,
                    health_tensor,
                    episode_start=episode_start,
                    hidden=hidden,
                    deterministic=False,
                )
            action = int(action_out.actions[0].cpu().item())
            policy_target = policy
            hidden = action_out.hidden.detach()
            hidden_mcts = hidden.detach().cpu().numpy().astype(np.float32)

        log_prob = float(torch.log_softmax(step_logits, dim=-1)[action].cpu().item())
        obs_seq.append(observation.astype(np.float32))
        health_seq.append(float(normalized_health))
        action_seq.append(action)
        old_log_prob_seq.append(log_prob)
        policy_targets_seq.append(policy_target)
        episode_starts_seq.append(1.0 if info["steps"] == 0 else 0.0)

        observation, _, terminated, truncated, info = env.step(action)

    return GRPOTrajectory(
        obs=np.asarray(obs_seq, dtype=np.float32),
        health=np.asarray(health_seq, dtype=np.float32),
        actions=np.asarray(action_seq, dtype=np.int64),
        old_log_probs=np.asarray(old_log_prob_seq, dtype=np.float32),
        policy_targets=np.asarray(policy_targets_seq, dtype=np.float32),
        episode_starts=np.asarray(episode_starts_seq, dtype=np.float32),
        outcome=str(info["outcome"]),
        reward_raw=float(info["episode_reward_raw"]),
        reward_clipped=float(info["episode_reward_clipped"]),
        length=int(info["steps"]),
        health_remaining=int(info["health"]),
        mine_hits=int(info["mine_hits"]),
        map_seed=int(info["map_seed"]),
        group_index=int(group_index),
        env_steps=int(info["steps"]),
    )


def train_grpo(
    config: dict[str, Any],
    device: str = "cpu",
    map_size: int | None = None,
    checkpoint_dir: str | Path | None = None,
    log_dir: str | Path | None = None,
    initial_checkpoint: str | Path | None = None,
    total_updates: int | None = None,
    mcts_simulations: int | None = None,
    max_episode_steps: int | None = None,
) -> dict[str, Any]:
    grpo_cfg = config["grpo"]
    env_cfg = dict(config["env"])
    if map_size is not None:
        env_cfg["size"] = map_size
    if max_episode_steps is not None:
        env_cfg["max_steps"] = max_episode_steps
    local_config = dict(config)
    local_config["env"] = env_cfg

    checkpoint_root = ensure_dir(checkpoint_dir or "minefield_rl/checkpoints")
    log_root = ensure_dir(log_dir or "minefield_rl/logs")
    csv_path = log_root / "grpo_train.csv"

    device_torch = torch.device(device)
    model = RecurrentPPOActorCritic(local_config).to(device_torch)
    if initial_checkpoint is not None:
        checkpoint = torch.load(initial_checkpoint, map_location=device_torch)
        model.load_state_dict(checkpoint["model_state_dict"])

    reference_model = RecurrentPPOActorCritic(local_config).to(device_torch)
    reference_model.load_state_dict(model.state_dict())
    reference_model.eval()
    for parameter in reference_model.parameters():
        parameter.requires_grad_(False)

    optimizer = Adam(model.parameters(), lr=float(grpo_cfg["learning_rate"]), eps=1e-5)
    search = AlphaZeroMCTS(model, local_config, device=device)

    writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(log_dir=str(log_root / "tensorboard" / "grpo"))
    except Exception:
        writer = None

    fieldnames = [
        "episode",
        "update",
        "env_steps",
        "reward_raw",
        "reward_clipped",
        "length",
        "outcome",
        "health_remaining",
        "mine_hits",
        "success_rate_100",
        "policy_loss",
        "entropy",
        "kl",
        "distill_loss",
        "map_seed",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        csv.DictWriter(handle, fieldnames=fieldnames).writeheader()

    groups_per_update = int(grpo_cfg["groups_per_update"])
    group_rollouts = int(grpo_cfg["group_rollouts"])
    updates = int(total_updates if total_updates is not None else grpo_cfg["total_updates"])
    epochs_per_update = int(grpo_cfg["epochs_per_update"])
    batch_size = int(grpo_cfg["batch_size"])
    clip_coef = float(grpo_cfg["clip_coef"])
    entropy_coef = float(grpo_cfg["entropy_coef"])
    kl_coef = float(grpo_cfg["kl_coef"])
    policy_target_coef = float(grpo_cfg["policy_target_coef"])
    grad_clip = float(grpo_cfg["grad_clip_norm"])
    checkpoint_interval = int(grpo_cfg["checkpoint_interval_updates"])
    normalize_advantages = bool(grpo_cfg.get("normalize_group_advantages", True))
    simulations = int(mcts_simulations if mcts_simulations is not None else grpo_cfg["mcts_simulations"])

    rolling_successes: deque[int] = deque(maxlen=100)
    best_success_rate = -1.0
    env_steps = 0
    episode_idx = 0
    last_policy_loss = np.nan
    last_entropy = np.nan
    last_kl = np.nan
    last_distill_loss = np.nan

    with trange(updates, desc="GRPO training", unit="update") as progress:
        for update in range(1, updates + 1):
            trajectories: list[GRPOTrajectory] = []

            for group_index in range(groups_per_update):
                seed_source = np.random.default_rng()
                group_seed = int(seed_source.integers(0, 2**31 - 1))
                for _ in range(group_rollouts):
                    env = MinefieldEnv.from_config({"env": env_cfg})
                    env.reset(seed=group_seed)
                    trajectory = _collect_trajectory(
                        env=env,
                        model=model,
                        search=search,
                        device=device_torch,
                        simulations=simulations,
                        group_index=group_index,
                    )
                    trajectories.append(trajectory)
                    episode_idx += 1
                    env_steps += trajectory.env_steps
                    rolling_successes.append(1 if trajectory.outcome == "success" else 0)
                    success_rate = float(np.mean(rolling_successes)) if rolling_successes else 0.0
                    row = {
                        "episode": episode_idx,
                        "update": update,
                        "env_steps": env_steps,
                        "reward_raw": trajectory.reward_raw,
                        "reward_clipped": trajectory.reward_clipped,
                        "length": trajectory.length,
                        "outcome": trajectory.outcome,
                        "health_remaining": trajectory.health_remaining,
                        "mine_hits": trajectory.mine_hits,
                        "success_rate_100": success_rate,
                        "policy_loss": last_policy_loss,
                        "entropy": last_entropy,
                        "kl": last_kl,
                        "distill_loss": last_distill_loss,
                        "map_seed": trajectory.map_seed,
                    }
                    with csv_path.open("a", newline="", encoding="utf-8") as handle:
                        csv.DictWriter(handle, fieldnames=fieldnames).writerow(row)

                    if writer is not None:
                        writer.add_scalar("episode/reward_raw", trajectory.reward_raw, episode_idx)
                        writer.add_scalar("episode/length", trajectory.length, episode_idx)
                        writer.add_scalar("episode/success_rate_100", success_rate, episode_idx)

            advantages = _compute_group_advantages(
                trajectories=trajectories,
                group_rollouts=group_rollouts,
                normalize_advantages=normalize_advantages,
            )
            model.train()
            indices = np.arange(len(trajectories))
            for _ in range(epochs_per_update):
                np.random.shuffle(indices)
                for start_index in range(0, len(indices), batch_size):
                    batch_indices = indices[start_index : start_index + batch_size]
                    batch_trajectories = [trajectories[index] for index in batch_indices]
                    batch_advantages = [advantages[index] for index in batch_indices]
                    (
                        obs,
                        health,
                        actions,
                        old_log_probs,
                        policy_targets,
                        episode_starts,
                        advantage_values,
                        mask,
                    ) = _make_batch(batch_trajectories, batch_advantages)
                    obs_tensor = torch.from_numpy(obs).float().to(device_torch)
                    health_tensor = torch.from_numpy(health).float().to(device_torch)
                    actions_tensor = torch.from_numpy(actions).long().to(device_torch)
                    old_log_probs_tensor = torch.from_numpy(old_log_probs).float().to(device_torch)
                    policy_targets_tensor = torch.from_numpy(policy_targets).float().to(device_torch)
                    episode_starts_tensor = torch.from_numpy(episode_starts).float().to(device_torch)
                    advantage_tensor = torch.from_numpy(advantage_values).float().to(device_torch)
                    mask_tensor = torch.from_numpy(mask).float().to(device_torch)
                    hidden = model.init_hidden(len(batch_trajectories), device_torch)

                    output = model(obs_tensor, health_tensor, episode_starts_tensor, hidden)
                    dist = Categorical(logits=output.logits)
                    new_log_probs = dist.log_prob(actions_tensor)
                    entropy = dist.entropy()

                    with torch.no_grad():
                        reference_output = reference_model(obs_tensor, health_tensor, episode_starts_tensor, hidden)

                    log_ratio = new_log_probs - old_log_probs_tensor
                    ratio = log_ratio.exp()
                    unclipped = ratio * advantage_tensor
                    clipped = torch.clamp(ratio, 1.0 - clip_coef, 1.0 + clip_coef) * advantage_tensor
                    policy_loss = -(torch.minimum(unclipped, clipped) * mask_tensor).sum() / mask_tensor.sum().clamp_min(1.0)

                    entropy_term = (entropy * mask_tensor).sum() / mask_tensor.sum().clamp_min(1.0)

                    current_log_probs = F.log_softmax(output.logits, dim=-1)
                    ref_log_probs = F.log_softmax(reference_output.logits, dim=-1)
                    ref_probs = ref_log_probs.exp()
                    kl = torch.sum(ref_probs * (ref_log_probs - current_log_probs), dim=-1)
                    kl = (kl * mask_tensor).sum() / mask_tensor.sum().clamp_min(1.0)

                    distill = -torch.sum(policy_targets_tensor * current_log_probs, dim=-1)
                    distill = (distill * mask_tensor).sum() / mask_tensor.sum().clamp_min(1.0)

                    loss = policy_loss + kl_coef * kl + policy_target_coef * distill - entropy_coef * entropy_term

                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    optimizer.step()

                    last_policy_loss = float(policy_loss.item())
                    last_entropy = float(entropy_term.item())
                    last_kl = float(kl.item())
                    last_distill_loss = float(distill.item())

            success_rate = float(np.mean(rolling_successes)) if rolling_successes else 0.0
            if writer is not None:
                writer.add_scalar("train/policy_loss", last_policy_loss, env_steps)
                writer.add_scalar("train/entropy", last_entropy, env_steps)
                writer.add_scalar("train/kl", last_kl, env_steps)
                writer.add_scalar("train/distill_loss", last_distill_loss, env_steps)
                writer.add_scalar("train/success_rate_100", success_rate, env_steps)

            if success_rate >= best_success_rate:
                best_success_rate = success_rate
                save_grpo_checkpoint(
                    checkpoint_root / "ppo_grpo_best.pt",
                    model,
                    optimizer,
                    update,
                    env_steps,
                    episode_idx,
                    local_config,
                )

            if update % checkpoint_interval == 0:
                save_grpo_checkpoint(
                    checkpoint_root / f"ppo_grpo_update_{update}.pt",
                    model,
                    optimizer,
                    update,
                    env_steps,
                    episode_idx,
                    local_config,
                )

            progress.update(1)
            progress.set_postfix(
                success_rate=f"{success_rate:.3f}",
                policy_loss=f"{last_policy_loss:.4f}" if not np.isnan(last_policy_loss) else "nan",
                kl=f"{last_kl:.4f}" if not np.isnan(last_kl) else "nan",
            )

    save_grpo_checkpoint(
        checkpoint_root / "ppo_grpo_latest.pt",
        model,
        optimizer,
        updates,
        env_steps,
        episode_idx,
        local_config,
    )
    if writer is not None:
        writer.close()

    return {
        "updates": updates,
        "episodes": episode_idx,
        "env_steps": env_steps,
        "best_success_rate_100": best_success_rate,
        "checkpoint": str(checkpoint_root / "ppo_grpo_latest.pt"),
        "best_checkpoint": str(checkpoint_root / "ppo_grpo_best.pt"),
        "log_csv": str(csv_path),
    }
