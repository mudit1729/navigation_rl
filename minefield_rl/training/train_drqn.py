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
from tqdm import trange

from minefield_rl.env.minefield_env import MinefieldEnv
from minefield_rl.models.drqn import DRQN
from minefield_rl.training.sequence_replay import SequenceReplayBuffer
from minefield_rl.utils import ensure_dir


def linear_schedule(start: float, end: float, step: int, decay_steps: int) -> float:
    if step >= decay_steps:
        return end
    fraction = step / max(decay_steps, 1)
    return start + fraction * (end - start)


def save_drqn_checkpoint(
    path: str | Path,
    model: DRQN,
    optimizer: torch.optim.Optimizer,
    step: int,
    episode: int,
    config: dict[str, Any],
) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "step": step,
            "episode": episode,
            "config": config,
        },
        Path(path),
    )


def train_drqn(
    config: dict[str, Any],
    device: str = "cpu",
    map_size: int | None = None,
    checkpoint_dir: str | Path | None = None,
    log_dir: str | Path | None = None,
) -> dict[str, Any]:
    training_cfg = config["drqn"]
    env_config = dict(config["env"])
    if map_size is not None:
        env_config["size"] = map_size
    env = MinefieldEnv.from_config({"env": env_config})

    checkpoint_root = ensure_dir(checkpoint_dir or "minefield_rl/checkpoints")
    log_root = ensure_dir(log_dir or "minefield_rl/logs")
    csv_path = log_root / "drqn_train.csv"

    model = DRQN(config).to(device)
    target_model = DRQN(config).to(device)
    target_model.load_state_dict(model.state_dict())
    target_model.eval()

    optimizer = Adam(model.parameters(), lr=float(training_cfg["lr"]))
    replay = SequenceReplayBuffer(
        capacity=int(training_cfg["replay_capacity"]),
        sequence_length=int(training_cfg["sequence_length"]),
        gamma=float(training_cfg["gamma"]),
    )

    writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(log_dir=str(log_root / "tensorboard" / "drqn"))
    except Exception:
        writer = None

    fieldnames = [
        "episode",
        "env_steps",
        "epsilon",
        "reward_raw",
        "reward_clipped",
        "length",
        "outcome",
        "health_remaining",
        "mine_hits",
        "success_rate_100",
        "loss",
        "mean_q",
        "map_seed",
    ]
    if not csv_path.exists():
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=fieldnames).writeheader()

    total_steps = int(training_cfg["train_steps"])
    batch_size = int(training_cfg["batch_size"])
    gamma = float(training_cfg["gamma"])
    grad_clip = float(training_cfg["grad_clip_norm"])
    train_interval = int(training_cfg.get("train_interval", 1))
    warmup_steps = int(training_cfg["warmup_steps"])
    target_update_interval = int(training_cfg["target_update_interval"])
    checkpoint_interval = int(training_cfg["checkpoint_interval_steps"])
    value_aux_weight = float(training_cfg.get("value_aux_weight", 0.0))

    env_steps = 0
    episode_idx = 0
    rolling_successes: deque[int] = deque(maxlen=100)
    best_success_rate = -1.0
    last_loss = np.nan

    with trange(total_steps, desc="DRQN training", unit="step") as progress:
        while env_steps < total_steps:
            observation, info = env.reset()
            episode_idx += 1
            hidden = model.init_hidden(1, torch.device(device))
            episode_transitions: list[dict[str, Any]] = []
            episode_q_values: list[float] = []

            terminated = False
            truncated = False
            while not (terminated or truncated):
                epsilon = linear_schedule(
                    float(training_cfg["epsilon_start"]),
                    float(training_cfg["epsilon_end"]),
                    env_steps,
                    int(training_cfg["epsilon_decay_steps"]),
                )
                normalized_health = info["health"] / env.max_health
                obs_tensor = torch.from_numpy(observation).float().unsqueeze(0).to(device)
                health_tensor = torch.tensor([[normalized_health]], dtype=torch.float32, device=device)

                model.eval()
                with torch.no_grad():
                    output = model.forward_step(obs_tensor, health_tensor, hidden)
                    q_values = output.q_values[0].cpu().numpy()
                    hidden = output.hidden.detach()

                if np.random.random() < epsilon:
                    action = int(env.action_space.sample())
                else:
                    action = int(np.argmax(q_values))

                next_observation, reward, terminated, truncated, next_info = env.step(action)
                done = float(terminated or truncated)
                episode_q_values.append(float(np.mean(q_values)))
                episode_transitions.append(
                    {
                        "obs": observation.astype(np.float32),
                        "health": np.float32(normalized_health),
                        "action": action,
                        "reward": np.float32(reward),
                        "next_obs": next_observation.astype(np.float32),
                        "next_health": np.float32(next_info["health"] / env.max_health),
                        "done": np.float32(done),
                    }
                )
                observation = next_observation
                info = next_info
                env_steps += 1
                progress.update(1)

                if env_steps >= warmup_steps and len(replay) >= batch_size and env_steps % train_interval == 0:
                    model.train()
                    batch = replay.sample(batch_size)
                    obs_batch = torch.from_numpy(batch.obs).float().to(device)
                    health_batch = torch.from_numpy(batch.health).float().to(device)
                    actions_batch = torch.from_numpy(batch.actions).long().to(device)
                    rewards_batch = torch.from_numpy(batch.rewards).float().to(device)
                    next_obs_batch = torch.from_numpy(batch.next_obs).float().to(device)
                    next_health_batch = torch.from_numpy(batch.next_health).float().to(device)
                    dones_batch = torch.from_numpy(batch.dones).float().to(device)
                    mask_batch = torch.from_numpy(batch.mask).float().to(device)
                    returns_batch = torch.from_numpy(batch.returns).float().to(device)

                    online_out = model(obs_batch, health_batch)
                    chosen_q = online_out.q_values.gather(-1, actions_batch.unsqueeze(-1)).squeeze(-1)
                    with torch.no_grad():
                        target_out = target_model(next_obs_batch, next_health_batch)
                        target_q = target_out.q_values.max(dim=-1).values
                        td_target = rewards_batch + gamma * (1.0 - dones_batch) * target_q

                    td_loss = F.smooth_l1_loss(chosen_q, td_target, reduction="none")
                    loss = (td_loss * mask_batch).sum() / mask_batch.sum().clamp_min(1.0)

                    if value_aux_weight > 0.0:
                        value_loss = F.mse_loss(online_out.values, returns_batch, reduction="none")
                        loss = loss + value_aux_weight * (value_loss * mask_batch).sum() / mask_batch.sum().clamp_min(1.0)

                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    optimizer.step()
                    last_loss = float(loss.item())

                    if writer is not None:
                        writer.add_scalar("train/loss", last_loss, env_steps)

                if env_steps % target_update_interval == 0:
                    target_model.load_state_dict(model.state_dict())

                if env_steps % checkpoint_interval == 0 and env_steps > 0:
                    save_drqn_checkpoint(
                        checkpoint_root / f"drqn_step_{env_steps}.pt",
                        model,
                        optimizer,
                        env_steps,
                        episode_idx,
                        config,
                    )

                if env_steps >= total_steps:
                    break

            replay.add_episode(episode_transitions)
            rolling_successes.append(1 if info["outcome"] == "success" else 0)
            success_rate = float(np.mean(rolling_successes)) if rolling_successes else 0.0
            mean_q = float(np.mean(episode_q_values)) if episode_q_values else 0.0
            row = {
                "episode": episode_idx,
                "env_steps": env_steps,
                "epsilon": epsilon,
                "reward_raw": info["episode_reward_raw"],
                "reward_clipped": info["episode_reward_clipped"],
                "length": info["steps"],
                "outcome": info["outcome"],
                "health_remaining": info["health"],
                "mine_hits": info["mine_hits"],
                "success_rate_100": success_rate,
                "loss": last_loss,
                "mean_q": mean_q,
                "map_seed": info["map_seed"],
            }
            with csv_path.open("a", newline="", encoding="utf-8") as handle:
                csv.DictWriter(handle, fieldnames=fieldnames).writerow(row)

            if writer is not None:
                writer.add_scalar("episode/reward_raw", float(info["episode_reward_raw"]), episode_idx)
                writer.add_scalar("episode/length", float(info["steps"]), episode_idx)
                writer.add_scalar("episode/success_rate_100", success_rate, episode_idx)
                writer.add_scalar("episode/mean_q", mean_q, episode_idx)

            if success_rate >= best_success_rate:
                best_success_rate = success_rate
                save_drqn_checkpoint(
                    checkpoint_root / "drqn_best.pt",
                    model,
                    optimizer,
                    env_steps,
                    episode_idx,
                    config,
                )

            if env_steps >= total_steps:
                break

    save_drqn_checkpoint(
        checkpoint_root / "drqn_latest.pt",
        model,
        optimizer,
        env_steps,
        episode_idx,
        config,
    )
    if writer is not None:
        writer.close()

    return {
        "episodes": episode_idx,
        "env_steps": env_steps,
        "best_success_rate_100": best_success_rate,
        "checkpoint": str(checkpoint_root / "drqn_latest.pt"),
        "log_csv": str(csv_path),
    }
