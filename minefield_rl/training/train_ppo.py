from __future__ import annotations

import csv
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.distributions import Categorical
from torch.nn import functional as F
from torch.optim import Adam
from tqdm import trange

from minefield_rl.env.minefield_env import MinefieldEnv
from minefield_rl.models.rppo import RecurrentPPOActorCritic
from minefield_rl.training.ppo_rollout import RecurrentRolloutBuffer
from minefield_rl.utils import ensure_dir


def save_ppo_checkpoint(
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
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "update": update,
            "env_steps": env_steps,
            "episode": episode,
            "config": config,
        },
        Path(path),
    )


def train_ppo(
    config: dict[str, Any],
    device: str = "cpu",
    map_size: int | None = None,
    checkpoint_dir: str | Path | None = None,
    log_dir: str | Path | None = None,
    initial_checkpoint: str | Path | None = None,
) -> dict[str, Any]:
    ppo_cfg = config["ppo"]
    env_cfg = dict(config["env"])
    if map_size is not None:
        env_cfg["size"] = map_size

    checkpoint_root = ensure_dir(checkpoint_dir or "minefield_rl/checkpoints")
    log_root = ensure_dir(log_dir or "minefield_rl/logs")
    csv_path = log_root / "ppo_train.csv"

    num_envs = int(ppo_cfg["num_envs"])
    rollout_steps = int(ppo_cfg["rollout_steps"])
    total_timesteps = int(ppo_cfg["total_timesteps"])
    sequence_length = int(ppo_cfg["sequence_length"])
    updates = max(1, total_timesteps // (num_envs * rollout_steps))
    gamma = float(ppo_cfg["gamma"])
    gae_lambda = float(ppo_cfg["gae_lambda"])
    clip_coef = float(ppo_cfg["clip_coef"])
    ent_coef = float(ppo_cfg["entropy_coef"])
    vf_coef = float(ppo_cfg["value_coef"])
    grad_clip = float(ppo_cfg["grad_clip_norm"])
    checkpoint_interval = int(ppo_cfg["checkpoint_interval_updates"])
    num_minibatches = int(ppo_cfg["num_minibatches"])
    update_epochs = int(ppo_cfg["update_epochs"])
    normalize_advantages = bool(ppo_cfg.get("normalize_advantages", True))
    target_kl = ppo_cfg.get("target_kl")

    envs = [MinefieldEnv.from_config({"env": env_cfg}) for _ in range(num_envs)]
    observations: list[np.ndarray] = []
    infos: list[dict[str, Any]] = []
    for env in envs:
        observation, info = env.reset()
        observations.append(observation)
        infos.append(info)

    model = RecurrentPPOActorCritic(config).to(device)
    if initial_checkpoint is not None:
        checkpoint = torch.load(initial_checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
    optimizer = Adam(model.parameters(), lr=float(ppo_cfg["learning_rate"]), eps=1e-5)
    rollout = RecurrentRolloutBuffer(
        rollout_steps=rollout_steps,
        num_envs=num_envs,
        obs_shape=observations[0].shape,
        gru_layers=model.gru_layers,
        hidden_dim=model.hidden_dim,
        sequence_length=sequence_length,
    )

    writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(log_dir=str(log_root / "tensorboard" / "ppo"))
    except Exception:
        writer = None

    fieldnames = [
        "episode",
        "env_steps",
        "reward_raw",
        "reward_clipped",
        "length",
        "outcome",
        "health_remaining",
        "mine_hits",
        "success_rate_100",
        "policy_loss",
        "value_loss",
        "entropy",
        "approx_kl",
        "map_seed",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        csv.DictWriter(handle, fieldnames=fieldnames).writeheader()

    hidden = model.init_hidden(num_envs, torch.device(device))
    episode_starts = np.ones(num_envs, dtype=np.float32)
    rolling_successes: deque[int] = deque(maxlen=100)
    best_success_rate = -1.0
    episode_idx = 0
    env_steps = 0
    last_policy_loss = np.nan
    last_value_loss = np.nan
    last_entropy = np.nan
    last_approx_kl = np.nan

    with trange(updates, desc="RPPO training", unit="update") as progress:
        for update in range(1, updates + 1):
            for step in range(rollout_steps):
                obs_batch = np.stack(observations, axis=0).astype(np.float32)
                health_batch = np.asarray([info["health"] / env.max_health for env, info in zip(envs, infos)], dtype=np.float32)
                hidden_np = hidden.detach().cpu().numpy().astype(np.float32)

                with torch.no_grad():
                    obs_tensor = torch.from_numpy(obs_batch).float().to(device)
                    health_tensor = torch.from_numpy(health_batch).float().to(device)
                    episode_start_tensor = torch.from_numpy(episode_starts).float().to(device)
                    action_out = model.act(
                        obs_tensor,
                        health_tensor,
                        episode_start=episode_start_tensor,
                        hidden=hidden,
                        deterministic=False,
                    )
                    actions = action_out.actions.cpu().numpy()
                    log_probs = action_out.log_probs.cpu().numpy().astype(np.float32)
                    values = action_out.values.cpu().numpy().astype(np.float32)
                    hidden = action_out.hidden.detach()

                rewards = np.zeros(num_envs, dtype=np.float32)
                dones = np.zeros(num_envs, dtype=np.float32)
                next_episode_starts = np.zeros(num_envs, dtype=np.float32)

                for env_index, env in enumerate(envs):
                    next_observation, reward, terminated, truncated, next_info = env.step(int(actions[env_index]))
                    done = float(terminated or truncated)
                    rewards[env_index] = reward
                    dones[env_index] = done
                    observations[env_index] = next_observation
                    infos[env_index] = next_info

                    if done:
                        episode_idx += 1
                        rolling_successes.append(1 if next_info["outcome"] == "success" else 0)
                        success_rate = float(np.mean(rolling_successes)) if rolling_successes else 0.0
                        row = {
                            "episode": episode_idx,
                            "env_steps": env_steps + env_index + 1,
                            "reward_raw": next_info["episode_reward_raw"],
                            "reward_clipped": next_info["episode_reward_clipped"],
                            "length": next_info["steps"],
                            "outcome": next_info["outcome"],
                            "health_remaining": next_info["health"],
                            "mine_hits": next_info["mine_hits"],
                            "success_rate_100": success_rate,
                            "policy_loss": last_policy_loss,
                            "value_loss": last_value_loss,
                            "entropy": last_entropy,
                            "approx_kl": last_approx_kl,
                            "map_seed": next_info["map_seed"],
                        }
                        with csv_path.open("a", newline="", encoding="utf-8") as handle:
                            csv.DictWriter(handle, fieldnames=fieldnames).writerow(row)

                        if writer is not None:
                            writer.add_scalar("episode/reward_raw", float(next_info["episode_reward_raw"]), episode_idx)
                            writer.add_scalar("episode/length", float(next_info["steps"]), episode_idx)
                            writer.add_scalar("episode/success_rate_100", success_rate, episode_idx)

                        reset_observation, reset_info = env.reset()
                        observations[env_index] = reset_observation
                        infos[env_index] = reset_info
                        next_episode_starts[env_index] = 1.0
                        hidden[:, env_index] = 0.0

                rollout.add(
                    step=step,
                    obs=obs_batch,
                    health=health_batch,
                    actions=actions.astype(np.int64),
                    log_probs=log_probs,
                    rewards=rewards,
                    dones=dones,
                    values=values,
                    episode_starts=episode_starts,
                    hiddens=hidden_np,
                )
                episode_starts = next_episode_starts
                env_steps += num_envs

            with torch.no_grad():
                obs_batch = np.stack(observations, axis=0).astype(np.float32)
                health_batch = np.asarray([info["health"] / env.max_health for env, info in zip(envs, infos)], dtype=np.float32)
                obs_tensor = torch.from_numpy(obs_batch).float().to(device)
                health_tensor = torch.from_numpy(health_batch).float().to(device)
                episode_start_tensor = torch.from_numpy(episode_starts).float().to(device)
                bootstrap_out = model.forward_step(obs_tensor, health_tensor, episode_start_tensor, hidden)
                last_values = bootstrap_out.values.cpu().numpy().astype(np.float32)

            rollout.compute_returns_and_advantages(last_values=last_values, gamma=gamma, gae_lambda=gae_lambda)

            early_stop = False
            for batch in rollout.iterate_minibatches(
                num_minibatches=num_minibatches,
                update_epochs=update_epochs,
                normalize_advantages=normalize_advantages,
            ):
                obs_tensor = torch.from_numpy(batch.obs).float().to(device)
                health_tensor = torch.from_numpy(batch.health).float().to(device)
                actions_tensor = torch.from_numpy(batch.actions).long().to(device)
                old_log_probs_tensor = torch.from_numpy(batch.old_log_probs).float().to(device)
                old_values_tensor = torch.from_numpy(batch.old_values).float().to(device)
                returns_tensor = torch.from_numpy(batch.returns).float().to(device)
                advantages_tensor = torch.from_numpy(batch.advantages).float().to(device)
                episode_start_tensor = torch.from_numpy(batch.episode_starts).float().to(device)
                init_hidden_tensor = torch.from_numpy(batch.init_hidden).float().to(device)

                output = model(obs_tensor, health_tensor, episode_start_tensor, init_hidden_tensor)
                dist = Categorical(logits=output.logits)
                new_log_probs = dist.log_prob(actions_tensor)
                entropy = dist.entropy()

                log_ratio = new_log_probs - old_log_probs_tensor
                ratio = log_ratio.exp()
                unclipped = -advantages_tensor * ratio
                clipped = -advantages_tensor * torch.clamp(ratio, 1.0 - clip_coef, 1.0 + clip_coef)
                policy_loss = torch.max(unclipped, clipped).mean()

                value_unclipped = (output.values - returns_tensor) ** 2
                value_clipped = old_values_tensor + torch.clamp(output.values - old_values_tensor, -clip_coef, clip_coef)
                value_clipped_loss = (value_clipped - returns_tensor) ** 2
                value_loss = 0.5 * torch.max(value_unclipped, value_clipped_loss).mean()

                entropy_loss = entropy.mean()
                loss = policy_loss + vf_coef * value_loss - ent_coef * entropy_loss

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

                last_policy_loss = float(policy_loss.item())
                last_value_loss = float(value_loss.item())
                last_entropy = float(entropy_loss.item())
                last_approx_kl = float(((ratio - 1.0) - log_ratio).mean().item())

                if target_kl is not None and last_approx_kl > float(target_kl):
                    early_stop = True
                    break

            success_rate = float(np.mean(rolling_successes)) if rolling_successes else 0.0
            if writer is not None:
                writer.add_scalar("train/policy_loss", last_policy_loss, env_steps)
                writer.add_scalar("train/value_loss", last_value_loss, env_steps)
                writer.add_scalar("train/entropy", last_entropy, env_steps)
                writer.add_scalar("train/approx_kl", last_approx_kl, env_steps)
                writer.add_scalar("train/success_rate_100", success_rate, env_steps)

            if success_rate >= best_success_rate:
                best_success_rate = success_rate
                save_ppo_checkpoint(
                    checkpoint_root / "ppo_best.pt",
                    model,
                    optimizer,
                    update,
                    env_steps,
                    episode_idx,
                    config,
                )

            if update % checkpoint_interval == 0:
                save_ppo_checkpoint(
                    checkpoint_root / f"ppo_update_{update}.pt",
                    model,
                    optimizer,
                    update,
                    env_steps,
                    episode_idx,
                    config,
                )

            progress.update(1)
            progress.set_postfix(
                success_rate=f"{success_rate:.3f}",
                policy_loss=f"{last_policy_loss:.4f}" if not np.isnan(last_policy_loss) else "nan",
                value_loss=f"{last_value_loss:.4f}" if not np.isnan(last_value_loss) else "nan",
            )
            if early_stop:
                continue

    save_ppo_checkpoint(
        checkpoint_root / "ppo_latest.pt",
        model,
        optimizer,
        updates,
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
        "checkpoint": str(checkpoint_root / "ppo_latest.pt"),
        "log_csv": str(csv_path),
    }
