from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F
from torch.optim import Adam
from tqdm import trange

from minefield_rl.env.minefield_env import MinefieldEnv
from minefield_rl.models.rppo import RecurrentPPOActorCritic
from minefield_rl.planning.expert import ExpertPathPlanner
from minefield_rl.utils import discount_cumsum, ensure_dir


@dataclass(slots=True)
class EpisodeDemo:
    obs: np.ndarray
    health: np.ndarray
    actions: np.ndarray
    returns: np.ndarray
    episode_starts: np.ndarray


def save_imitation_checkpoint(
    path: str | Path,
    model: RecurrentPPOActorCritic,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    demos_collected: int,
    config: dict[str, Any],
) -> None:
    torch.save(
        {
            "agent": "ppo",
            "training_stage": "imitation",
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "demos_collected": demos_collected,
            "config": config,
        },
        Path(path),
    )


def _generate_demonstrations(
    config: dict[str, Any],
    demo_episodes: int,
    mine_cost: float,
) -> tuple[list[EpisodeDemo], dict[str, float]]:
    env = MinefieldEnv.from_config(config)
    planner = ExpertPathPlanner(mine_cost=mine_cost)
    episodes: list[EpisodeDemo] = []
    lengths: list[int] = []
    rewards: list[float] = []
    mine_hits: list[int] = []
    skipped = 0

    for _ in trange(demo_episodes, desc="Collecting demos", unit="episode"):
        observation, info = env.reset()
        plan = planner.plan(env)
        if plan is None or not plan.actions:
            skipped += 1
            continue

        obs_seq: list[np.ndarray] = []
        health_seq: list[float] = []
        action_seq: list[int] = []
        reward_seq: list[float] = []
        episode_starts: list[float] = []
        terminated = False
        truncated = False

        for step_index, action in enumerate(plan.actions):
            obs_seq.append(observation.astype(np.float32))
            health_seq.append(info["health"] / env.max_health)
            action_seq.append(int(action))
            episode_starts.append(1.0 if step_index == 0 else 0.0)

            observation, reward, terminated, truncated, info = env.step(action)
            reward_seq.append(float(reward))
            if terminated or truncated:
                break

        if not action_seq:
            skipped += 1
            continue

        returns = np.asarray(discount_cumsum(reward_seq, gamma=float(config["ppo"]["gamma"])), dtype=np.float32)
        returns = np.tanh(returns / 10.0)
        episodes.append(
            EpisodeDemo(
                obs=np.asarray(obs_seq, dtype=np.float32),
                health=np.asarray(health_seq, dtype=np.float32),
                actions=np.asarray(action_seq, dtype=np.int64),
                returns=returns,
                episode_starts=np.asarray(episode_starts, dtype=np.float32),
            )
        )
        lengths.append(len(action_seq))
        rewards.append(float(info["episode_reward_raw"]))
        mine_hits.append(int(info["mine_hits"]))

    summary = {
        "collected": float(len(episodes)),
        "skipped": float(skipped),
        "mean_length": float(np.mean(lengths)) if lengths else 0.0,
        "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
        "mean_mine_hits": float(np.mean(mine_hits)) if mine_hits else 0.0,
    }
    return episodes, summary


def _make_batch(episodes: list[EpisodeDemo]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    max_len = max(len(episode.actions) for episode in episodes)
    batch_size = len(episodes)
    obs_shape = episodes[0].obs.shape[1:]
    obs = np.zeros((batch_size, max_len, *obs_shape), dtype=np.float32)
    health = np.zeros((batch_size, max_len), dtype=np.float32)
    actions = np.zeros((batch_size, max_len), dtype=np.int64)
    returns = np.zeros((batch_size, max_len), dtype=np.float32)
    episode_starts = np.zeros((batch_size, max_len), dtype=np.float32)
    mask = np.zeros((batch_size, max_len), dtype=np.float32)

    for batch_index, episode in enumerate(episodes):
        length = len(episode.actions)
        obs[batch_index, :length] = episode.obs
        health[batch_index, :length] = episode.health
        actions[batch_index, :length] = episode.actions
        returns[batch_index, :length] = episode.returns
        episode_starts[batch_index, :length] = episode.episode_starts
        mask[batch_index, :length] = 1.0

    return obs, health, actions, returns, episode_starts, mask


def train_imitation(
    config: dict[str, Any],
    device: str = "cpu",
    map_size: int | None = None,
    checkpoint_dir: str | Path | None = None,
    log_dir: str | Path | None = None,
    initial_checkpoint: str | Path | None = None,
) -> dict[str, Any]:
    il_cfg = config["imitation"]
    local_config = dict(config)
    if map_size is not None:
        local_env = dict(config["env"])
        local_env["size"] = map_size
        local_config = dict(config)
        local_config["env"] = local_env

    checkpoint_root = ensure_dir(checkpoint_dir or "minefield_rl/checkpoints")
    log_root = ensure_dir(log_dir or "minefield_rl/logs")
    csv_path = log_root / "imitation_train.csv"

    episodes, demo_summary = _generate_demonstrations(
        config=local_config,
        demo_episodes=int(il_cfg["demo_episodes"]),
        mine_cost=float(il_cfg["expert_mine_cost"]),
    )
    if not episodes:
        raise RuntimeError("No expert demonstrations were collected")

    model = RecurrentPPOActorCritic(local_config).to(device)
    if initial_checkpoint is not None:
        ckpt = torch.load(str(initial_checkpoint), map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
    optimizer = Adam(model.parameters(), lr=float(il_cfg["learning_rate"]), eps=1e-5)
    batch_size = int(il_cfg["batch_size"])
    epochs = int(il_cfg["epochs"])
    value_coef = float(il_cfg["value_coef"])
    grad_clip = float(il_cfg["grad_clip_norm"])
    checkpoint_interval = int(il_cfg["checkpoint_interval_epochs"])

    writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(log_dir=str(log_root / "tensorboard" / "imitation"))
    except Exception:
        writer = None

    fieldnames = [
        "epoch",
        "policy_loss",
        "value_loss",
        "accuracy",
        "demo_episodes",
        "mean_demo_length",
        "mean_demo_reward",
        "mean_demo_mine_hits",
        "skipped_demo_episodes",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        csv.DictWriter(handle, fieldnames=fieldnames).writeheader()

    best_accuracy = -1.0
    for epoch in trange(1, epochs + 1, desc="Imitation training", unit="epoch"):
        random.shuffle(episodes)
        epoch_policy_loss = 0.0
        epoch_value_loss = 0.0
        epoch_correct = 0.0
        epoch_count = 0.0
        batch_counter = 0

        for start_index in range(0, len(episodes), batch_size):
            batch_episodes = episodes[start_index : start_index + batch_size]
            obs, health, actions, returns, episode_starts, mask = _make_batch(batch_episodes)
            obs_tensor = torch.from_numpy(obs).float().to(device)
            health_tensor = torch.from_numpy(health).float().to(device)
            actions_tensor = torch.from_numpy(actions).long().to(device)
            returns_tensor = torch.from_numpy(returns).float().to(device)
            episode_start_tensor = torch.from_numpy(episode_starts).float().to(device)
            mask_tensor = torch.from_numpy(mask).float().to(device)
            hidden = model.init_hidden(len(batch_episodes), torch.device(device))

            output = model(obs_tensor, health_tensor, episode_start_tensor, hidden)
            logits = output.logits
            log_probs = F.log_softmax(logits, dim=-1)
            selected_log_probs = log_probs.gather(-1, actions_tensor.unsqueeze(-1)).squeeze(-1)
            policy_loss = -(selected_log_probs * mask_tensor).sum() / mask_tensor.sum().clamp_min(1.0)
            value_loss = F.mse_loss(output.values, returns_tensor, reduction="none")
            value_loss = (value_loss * mask_tensor).sum() / mask_tensor.sum().clamp_min(1.0)
            loss = policy_loss + value_coef * value_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            predictions = logits.argmax(dim=-1)
            correct = ((predictions == actions_tensor).float() * mask_tensor).sum().item()
            count = mask_tensor.sum().item()
            epoch_policy_loss += float(policy_loss.item())
            epoch_value_loss += float(value_loss.item())
            epoch_correct += correct
            epoch_count += count
            batch_counter += 1

        mean_policy_loss = epoch_policy_loss / max(batch_counter, 1)
        mean_value_loss = epoch_value_loss / max(batch_counter, 1)
        accuracy = epoch_correct / max(epoch_count, 1.0)
        row = {
            "epoch": epoch,
            "policy_loss": mean_policy_loss,
            "value_loss": mean_value_loss,
            "accuracy": accuracy,
            "demo_episodes": int(demo_summary["collected"]),
            "mean_demo_length": demo_summary["mean_length"],
            "mean_demo_reward": demo_summary["mean_reward"],
            "mean_demo_mine_hits": demo_summary["mean_mine_hits"],
            "skipped_demo_episodes": int(demo_summary["skipped"]),
        }
        with csv_path.open("a", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=fieldnames).writerow(row)

        if writer is not None:
            writer.add_scalar("train/policy_loss", mean_policy_loss, epoch)
            writer.add_scalar("train/value_loss", mean_value_loss, epoch)
            writer.add_scalar("train/accuracy", accuracy, epoch)

        if accuracy >= best_accuracy:
            best_accuracy = accuracy
            save_imitation_checkpoint(
                checkpoint_root / "ppo_il_best.pt",
                model,
                optimizer,
                epoch,
                int(demo_summary["collected"]),
                local_config,
            )
        if epoch % checkpoint_interval == 0:
            save_imitation_checkpoint(
                checkpoint_root / f"ppo_il_epoch_{epoch}.pt",
                model,
                optimizer,
                epoch,
                int(demo_summary["collected"]),
                local_config,
            )

    save_imitation_checkpoint(
        checkpoint_root / "ppo_il_latest.pt",
        model,
        optimizer,
        epochs,
        int(demo_summary["collected"]),
        local_config,
    )
    if writer is not None:
        writer.close()

    return {
        "demo_episodes": int(demo_summary["collected"]),
        "skipped_demo_episodes": int(demo_summary["skipped"]),
        "mean_demo_length": demo_summary["mean_length"],
        "best_accuracy": best_accuracy,
        "checkpoint": str(checkpoint_root / "ppo_il_latest.pt"),
        "log_csv": str(csv_path),
    }
