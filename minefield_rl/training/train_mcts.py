from __future__ import annotations

import csv
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam

from minefield_rl.env.minefield_env import MinefieldEnv
from minefield_rl.models.drqn import DRQN
from minefield_rl.models.mcts import AlphaZeroMCTS
from minefield_rl.training.sequence_replay import PolicyReplayBuffer
from minefield_rl.utils import discount_cumsum, ensure_dir


def save_mcts_checkpoint(
    path: str | Path,
    model: DRQN,
    optimizer: torch.optim.Optimizer,
    episode: int,
    config: dict[str, Any],
) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "episode": episode,
            "config": config,
        },
        Path(path),
    )


def train_mcts(
    config: dict[str, Any],
    device: str = "cpu",
    map_size: int | None = None,
    checkpoint_dir: str | Path | None = None,
    log_dir: str | Path | None = None,
    initial_checkpoint: str | Path | None = None,
    episodes: int | None = None,
) -> dict[str, Any]:
    mcts_cfg = config["mcts"]
    env_cfg = dict(config["env"])
    if map_size is not None:
        env_cfg["size"] = map_size
    env = MinefieldEnv.from_config({"env": env_cfg})

    checkpoint_root = ensure_dir(checkpoint_dir or "minefield_rl/checkpoints")
    log_root = ensure_dir(log_dir or "minefield_rl/logs")
    csv_path = log_root / "mcts_train.csv"

    model = DRQN(config).to(device)
    if initial_checkpoint is not None:
        checkpoint = torch.load(initial_checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
    optimizer = Adam(
        model.parameters(),
        lr=float(mcts_cfg["lr"]),
        weight_decay=float(mcts_cfg.get("weight_decay", 0.0)),
    )
    search = AlphaZeroMCTS(model, config, device=device)
    replay = PolicyReplayBuffer(
        capacity=int(mcts_cfg["replay_capacity"]),
        sequence_length=int(mcts_cfg["sequence_length"]),
        action_dim=config["model"]["action_dim"],
    )

    writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(log_dir=str(log_root / "tensorboard" / "mcts"))
    except Exception:
        writer = None

    fieldnames = [
        "episode",
        "reward_raw",
        "reward_clipped",
        "length",
        "outcome",
        "health_remaining",
        "mine_hits",
        "success_rate_100",
        "loss",
        "map_seed",
    ]
    if not csv_path.exists():
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=fieldnames).writeheader()

    batch_size = int(mcts_cfg["batch_size"])
    grad_clip = float(mcts_cfg["grad_clip_norm"])
    checkpoint_interval = int(mcts_cfg["checkpoint_interval_episodes"])
    gamma = float(mcts_cfg["gamma"])
    total_episodes = int(episodes if episodes is not None else mcts_cfg.get("train_episodes", 100))
    warmup_episodes = int(mcts_cfg["warmup_episodes"])

    rolling_successes: deque[int] = deque(maxlen=100)
    best_success_rate = -1.0
    last_loss = np.nan

    for episode_idx in range(1, total_episodes + 1):
        observation, info = env.reset()
        rnn_hidden: np.ndarray | None = None
        trajectory_records: list[dict[str, Any]] = []
        raw_rewards: list[float] = []
        terminated = False
        truncated = False

        while not (terminated or truncated):
            result = search.search(
                env=env,
                observation=observation,
                health=info["health"],
                hidden_in=rnn_hidden,
                simulations=int(mcts_cfg["simulations_train"]),
                training=True,
            )
            action = result.chosen_action
            next_observation, reward, terminated, truncated, next_info = env.step(action)
            raw_rewards.append(float(next_info["reward_terms"]["raw_total"]))
            trajectory_records.append(
                {
                    "obs": observation.astype(np.float32),
                    "health": np.float32(info["health"] / env.max_health),
                    "policy_target": result.policy.astype(np.float32),
                }
            )
            observation = next_observation
            info = next_info
            rnn_hidden = result.root_hidden

        discounted_returns = discount_cumsum(raw_rewards, gamma)
        for index, record in enumerate(trajectory_records):
            record["value_target"] = np.float32(np.tanh(discounted_returns[index] / 10.0))
        replay.add_episode(trajectory_records)

        if episode_idx > warmup_episodes and len(replay) >= batch_size:
            model.train()
            batch = replay.sample(batch_size)
            obs_batch = torch.from_numpy(batch.obs).float().to(device)
            health_batch = torch.from_numpy(batch.health).float().to(device)
            policy_targets = torch.from_numpy(batch.policy_targets).float().to(device)
            value_targets = torch.from_numpy(batch.value_targets).float().to(device)
            mask = torch.from_numpy(batch.mask).float().to(device)

            output = model(obs_batch, health_batch)
            log_probs = F.log_softmax(output.q_values, dim=-1)
            policy_loss = -torch.sum(policy_targets * log_probs, dim=-1)
            policy_loss = (policy_loss * mask).sum() / mask.sum().clamp_min(1.0)
            value_loss = F.mse_loss(output.values, value_targets, reduction="none")
            value_loss = (value_loss * mask).sum() / mask.sum().clamp_min(1.0)
            loss = policy_loss + value_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            last_loss = float(loss.item())

            if writer is not None:
                writer.add_scalar("train/loss", last_loss, episode_idx)

        rolling_successes.append(1 if info["outcome"] == "success" else 0)
        success_rate = float(np.mean(rolling_successes)) if rolling_successes else 0.0
        row = {
            "episode": episode_idx,
            "reward_raw": info["episode_reward_raw"],
            "reward_clipped": info["episode_reward_clipped"],
            "length": info["steps"],
            "outcome": info["outcome"],
            "health_remaining": info["health"],
            "mine_hits": info["mine_hits"],
            "success_rate_100": success_rate,
            "loss": last_loss,
            "map_seed": info["map_seed"],
        }
        with csv_path.open("a", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=fieldnames).writerow(row)

        if writer is not None:
            writer.add_scalar("episode/reward_raw", float(info["episode_reward_raw"]), episode_idx)
            writer.add_scalar("episode/length", float(info["steps"]), episode_idx)
            writer.add_scalar("episode/success_rate_100", success_rate, episode_idx)

        if episode_idx % checkpoint_interval == 0:
            save_mcts_checkpoint(
                checkpoint_root / f"mcts_episode_{episode_idx}.pt",
                model,
                optimizer,
                episode_idx,
                config,
            )

        if success_rate >= best_success_rate:
            best_success_rate = success_rate
            save_mcts_checkpoint(
                checkpoint_root / "mcts_best.pt",
                model,
                optimizer,
                episode_idx,
                config,
            )

    save_mcts_checkpoint(
        checkpoint_root / "mcts_latest.pt",
        model,
        optimizer,
        total_episodes,
        config,
    )
    if writer is not None:
        writer.close()

    return {
        "episodes": total_episodes,
        "best_success_rate_100": best_success_rate,
        "checkpoint": str(checkpoint_root / "mcts_latest.pt"),
        "log_csv": str(csv_path),
    }
