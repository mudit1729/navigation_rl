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
from minefield_rl.models.mcts import AlphaZeroMCTS
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
    training_stage: str = "ppo",
) -> None:
    torch.save(
        {
            "agent": "ppo",
            "training_stage": training_stage,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "update": update,
            "env_steps": env_steps,
            "episode": episode,
            "config": config,
        },
        Path(path),
    )


def _build_writer(log_root: Path, subdir: str):
    try:
        from torch.utils.tensorboard import SummaryWriter

        return SummaryWriter(log_dir=str(log_root / "tensorboard" / subdir))
    except Exception:
        return None


def _write_episode_row(csv_path: Path, fieldnames: list[str], row: dict[str, Any]) -> None:
    with csv_path.open("a", newline="", encoding="utf-8") as handle:
        csv.DictWriter(handle, fieldnames=fieldnames).writerow(row)


def _copy_env_config(config: dict[str, Any], map_size: int | None) -> tuple[dict[str, Any], dict[str, Any]]:
    env_cfg = dict(config["env"])
    if map_size is not None:
        env_cfg["size"] = map_size
    local_config = dict(config)
    local_config["env"] = env_cfg
    return env_cfg, local_config


def _build_reference_model(
    config: dict[str, Any],
    device: torch.device,
    initial_checkpoint: str | Path | None,
    reference_kl_coef: float,
) -> tuple[RecurrentPPOActorCritic, RecurrentPPOActorCritic | None]:
    model = RecurrentPPOActorCritic(config).to(device)
    reference_model: RecurrentPPOActorCritic | None = None
    if initial_checkpoint is not None:
        checkpoint = torch.load(initial_checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        if reference_kl_coef > 0.0:
            reference_model = RecurrentPPOActorCritic(config).to(device)
            reference_model.load_state_dict(checkpoint["model_state_dict"])
            reference_model.eval()
            for parameter in reference_model.parameters():
                parameter.requires_grad_(False)
    return model, reference_model


def _collect_model_actions(
    model: RecurrentPPOActorCritic,
    device: torch.device,
    obs_batch: np.ndarray,
    health_batch: np.ndarray,
    episode_starts: np.ndarray,
    hidden: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, torch.Tensor]:
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
    return (
        action_out.actions.cpu().numpy().astype(np.int64),
        action_out.log_probs.cpu().numpy().astype(np.float32),
        action_out.values.cpu().numpy().astype(np.float32),
        action_out.hidden.detach(),
    )


def _collect_mcts_actions(
    model: RecurrentPPOActorCritic,
    envs: list[MinefieldEnv],
    infos: list[dict[str, Any]],
    observations: list[np.ndarray],
    search: AlphaZeroMCTS,
    episode_starts: np.ndarray,
    hidden: torch.Tensor,
    device: torch.device,
    simulations: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, torch.Tensor, np.ndarray]:
    num_envs = len(envs)
    obs_batch = np.stack(observations, axis=0).astype(np.float32)
    health_batch = np.asarray(
        [info["health"] / env.max_health for env, info in zip(envs, infos)],
        dtype=np.float32,
    )
    with torch.no_grad():
        obs_tensor = torch.from_numpy(obs_batch).float().to(device)
        health_tensor = torch.from_numpy(health_batch).float().to(device)
        episode_start_tensor = torch.from_numpy(episode_starts).float().to(device)
        actor_out = model.forward_step(
            obs_tensor,
            health_tensor,
            episode_start=episode_start_tensor,
            hidden=hidden,
        )

    actions = np.zeros(num_envs, dtype=np.int64)
    values = actor_out.values.cpu().numpy().astype(np.float32)
    policy_targets = np.zeros((num_envs, search.model.action_dim), dtype=np.float32)
    behavior_log_probs = np.zeros(num_envs, dtype=np.float32)
    next_hidden = actor_out.hidden.detach()
    hidden_np = hidden.detach().cpu().numpy().astype(np.float32)

    for env_index, env in enumerate(envs):
        result = search.search(
            env=env,
            observation=observations[env_index],
            health=int(infos[env_index]["health"]),
            hidden_in=hidden_np[:, env_index : env_index + 1, :],
            simulations=simulations,
            training=True,
            add_root_noise=False,
        )
        actions[env_index] = result.chosen_action
        policy_targets[env_index] = result.policy.astype(np.float32)
        # PPO ratio denominator must match the distribution we actually sampled
        # from. MCTS picked chosen_action from result.behavior_policy (the
        # temperature-softened visit-count distribution in training mode), so
        # log_prob_old = log p_behavior(chosen_action). Using the actor logits
        # here would make the off-policy ratio mathematically inconsistent.
        prob = float(result.behavior_policy[result.chosen_action])
        behavior_log_probs[env_index] = float(np.log(max(prob, 1e-8)))

    return actions, behavior_log_probs, values, next_hidden, policy_targets


def _maybe_compute_reference_logits(
    reference_model: RecurrentPPOActorCritic | None,
    reference_hidden: torch.Tensor | None,
    obs_batch: np.ndarray,
    health_batch: np.ndarray,
    episode_starts: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray | None, torch.Tensor | None]:
    if reference_model is None or reference_hidden is None:
        return None, reference_hidden

    with torch.no_grad():
        obs_tensor = torch.from_numpy(obs_batch).float().to(device)
        health_tensor = torch.from_numpy(health_batch).float().to(device)
        episode_start_tensor = torch.from_numpy(episode_starts).float().to(device)
        reference_out = reference_model.forward_step(
            obs_tensor,
            health_tensor,
            episode_start_tensor,
            reference_hidden,
        )
    return reference_out.logits.cpu().numpy().astype(np.float32), reference_out.hidden.detach()


def _train_ppo_impl(
    config: dict[str, Any],
    device: str = "cpu",
    map_size: int | None = None,
    checkpoint_dir: str | Path | None = None,
    log_dir: str | Path | None = None,
    initial_checkpoint: str | Path | None = None,
    *,
    use_mcts: bool,
) -> dict[str, Any]:
    section_name = "ppo_mcts" if use_mcts else "ppo"
    ppo_cfg = config.get(section_name, config["ppo"])
    env_cfg, local_config = _copy_env_config(config, map_size)

    checkpoint_root = ensure_dir(checkpoint_dir or "minefield_rl/checkpoints")
    log_root = ensure_dir(log_dir or "minefield_rl/logs")
    csv_name = "ppo_mcts_train.csv" if use_mcts else "ppo_train.csv"
    csv_path = log_root / csv_name
    checkpoint_prefix = "ppo_mcts" if use_mcts else "ppo"
    training_stage = "ppo_mcts" if use_mcts else "ppo"
    tensorboard_subdir = "ppo_mcts" if use_mcts else "ppo"

    num_envs = int(ppo_cfg["num_envs"])
    rollout_steps = int(ppo_cfg["rollout_steps"])
    total_timesteps = int(ppo_cfg["total_timesteps"])
    sequence_length = int(ppo_cfg["sequence_length"])
    updates = max(1, total_timesteps // (num_envs * rollout_steps))
    gamma = float(ppo_cfg["gamma"])
    gae_lambda = float(ppo_cfg["gae_lambda"])
    clip_coef = float(ppo_cfg["clip_coef"])
    entropy_coef = float(ppo_cfg["entropy_coef"])
    value_coef = float(ppo_cfg["value_coef"])
    grad_clip = float(ppo_cfg["grad_clip_norm"])
    checkpoint_interval = int(ppo_cfg["checkpoint_interval_updates"])
    num_minibatches = int(ppo_cfg["num_minibatches"])
    update_epochs = int(ppo_cfg["update_epochs"])
    normalize_advantages = bool(ppo_cfg.get("normalize_advantages", True))
    target_kl = ppo_cfg.get("target_kl")
    reference_kl_coef = float(ppo_cfg.get("reference_kl_coef", 0.0))
    mcts_policy_coef = float(ppo_cfg.get("mcts_policy_coef", 0.0)) if use_mcts else 0.0
    mcts_simulations = int(ppo_cfg.get("mcts_simulations", config.get("mcts", {}).get("simulations_train", 4)))

    device_torch = torch.device(device)
    envs = [MinefieldEnv.from_config({"env": env_cfg}) for _ in range(num_envs)]
    observations: list[np.ndarray] = []
    infos: list[dict[str, Any]] = []
    for env in envs:
        observation, info = env.reset()
        observations.append(observation)
        infos.append(info)

    model, reference_model = _build_reference_model(local_config, device_torch, initial_checkpoint, reference_kl_coef)
    optimizer_lr = (
        float(ppo_cfg.get("fine_tune_learning_rate", ppo_cfg["learning_rate"]))
        if initial_checkpoint is not None
        else float(ppo_cfg["learning_rate"])
    )
    optimizer = Adam(model.parameters(), lr=optimizer_lr, eps=1e-5)
    search = AlphaZeroMCTS(model, local_config, device=device) if use_mcts else None

    rollout = RecurrentRolloutBuffer(
        rollout_steps=rollout_steps,
        num_envs=num_envs,
        obs_shape=observations[0].shape,
        gru_layers=model.gru_layers,
        hidden_dim=model.hidden_dim,
        sequence_length=sequence_length,
    )
    writer = _build_writer(log_root, tensorboard_subdir)

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
        "reference_kl",
        "mcts_policy_loss",
        "map_seed",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        csv.DictWriter(handle, fieldnames=fieldnames).writeheader()

    hidden = model.init_hidden(num_envs, device_torch)
    reference_hidden = (
        reference_model.init_hidden(num_envs, device_torch)
        if reference_model is not None
        else None
    )
    episode_starts = np.ones(num_envs, dtype=np.float32)
    rolling_successes: deque[int] = deque(maxlen=100)
    best_success_rate = -1.0
    best_checkpoint_path = checkpoint_root / f"{checkpoint_prefix}_best.pt"
    latest_checkpoint_path = checkpoint_root / f"{checkpoint_prefix}_latest.pt"
    episode_idx = 0
    env_steps = 0
    last_policy_loss = np.nan
    last_value_loss = np.nan
    last_entropy = np.nan
    last_approx_kl = np.nan
    last_reference_kl = np.nan
    last_mcts_policy_loss = np.nan

    with trange(updates, desc=f"{training_stage.upper()} training", unit="update") as progress:
        for update in range(1, updates + 1):
            rollout.reset()
            for step in range(rollout_steps):
                obs_batch = np.stack(observations, axis=0).astype(np.float32)
                health_batch = np.asarray(
                    [info["health"] / env.max_health for env, info in zip(envs, infos)],
                    dtype=np.float32,
                )
                hidden_np = hidden.detach().cpu().numpy().astype(np.float32)
                reference_logits, reference_hidden = _maybe_compute_reference_logits(
                    reference_model,
                    reference_hidden,
                    obs_batch,
                    health_batch,
                    episode_starts,
                    device_torch,
                )

                if use_mcts and search is not None:
                    actions, log_probs, values, hidden, policy_targets = _collect_mcts_actions(
                        model=model,
                        envs=envs,
                        infos=infos,
                        observations=observations,
                        search=search,
                        episode_starts=episode_starts,
                        hidden=hidden,
                        device=device_torch,
                        simulations=mcts_simulations,
                    )
                else:
                    actions, log_probs, values, hidden = _collect_model_actions(
                        model=model,
                        device=device_torch,
                        obs_batch=obs_batch,
                        health_batch=health_batch,
                        episode_starts=episode_starts,
                        hidden=hidden,
                    )
                    policy_targets = None

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
                            "reference_kl": last_reference_kl,
                            "mcts_policy_loss": last_mcts_policy_loss,
                            "map_seed": next_info["map_seed"],
                        }
                        _write_episode_row(csv_path, fieldnames, row)

                        if writer is not None:
                            writer.add_scalar("episode/reward_raw", float(next_info["episode_reward_raw"]), episode_idx)
                            writer.add_scalar("episode/length", float(next_info["steps"]), episode_idx)
                            writer.add_scalar("episode/success_rate_100", success_rate, episode_idx)

                        reset_observation, reset_info = env.reset()
                        observations[env_index] = reset_observation
                        infos[env_index] = reset_info
                        next_episode_starts[env_index] = 1.0
                        hidden[:, env_index] = 0.0
                        if reference_hidden is not None:
                            reference_hidden[:, env_index] = 0.0

                rollout.add(
                    step=step,
                    obs=obs_batch,
                    health=health_batch,
                    actions=actions,
                    log_probs=log_probs,
                    rewards=rewards,
                    dones=dones,
                    values=values,
                    episode_starts=episode_starts,
                    hiddens=hidden_np,
                    ref_logits=reference_logits,
                    policy_targets=policy_targets,
                )
                episode_starts = next_episode_starts
                env_steps += num_envs

            with torch.no_grad():
                obs_batch = np.stack(observations, axis=0).astype(np.float32)
                health_batch = np.asarray(
                    [info["health"] / env.max_health for env, info in zip(envs, infos)],
                    dtype=np.float32,
                )
                obs_tensor = torch.from_numpy(obs_batch).float().to(device_torch)
                health_tensor = torch.from_numpy(health_batch).float().to(device_torch)
                episode_start_tensor = torch.from_numpy(episode_starts).float().to(device_torch)
                bootstrap_out = model.forward_step(obs_tensor, health_tensor, episode_start_tensor, hidden)
                last_values = bootstrap_out.values.cpu().numpy().astype(np.float32)

            rollout.compute_returns_and_advantages(last_values=last_values, gamma=gamma, gae_lambda=gae_lambda)

            early_stop = False
            model.train()
            for batch in rollout.iterate_minibatches(
                num_minibatches=num_minibatches,
                update_epochs=update_epochs,
                normalize_advantages=normalize_advantages,
            ):
                obs_tensor = torch.from_numpy(batch.obs).float().to(device_torch)
                health_tensor = torch.from_numpy(batch.health).float().to(device_torch)
                actions_tensor = torch.from_numpy(batch.actions).long().to(device_torch)
                old_log_probs_tensor = torch.from_numpy(batch.old_log_probs).float().to(device_torch)
                old_values_tensor = torch.from_numpy(batch.old_values).float().to(device_torch)
                returns_tensor = torch.from_numpy(batch.returns).float().to(device_torch)
                advantages_tensor = torch.from_numpy(batch.advantages).float().to(device_torch)
                episode_start_tensor = torch.from_numpy(batch.episode_starts).float().to(device_torch)
                init_hidden_tensor = torch.from_numpy(batch.init_hidden).float().to(device_torch)
                ref_logits_tensor = (
                    torch.from_numpy(batch.ref_logits).float().to(device_torch)
                    if batch.ref_logits is not None
                    else None
                )
                policy_targets_tensor = (
                    torch.from_numpy(batch.policy_targets).float().to(device_torch)
                    if batch.policy_targets is not None
                    else None
                )

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

                entropy_term = entropy.mean()
                reference_kl = torch.zeros((), device=device_torch)
                if ref_logits_tensor is not None:
                    current_log_probs = F.log_softmax(output.logits, dim=-1)
                    reference_log_probs = F.log_softmax(ref_logits_tensor, dim=-1)
                    reference_probs = reference_log_probs.exp()
                    reference_kl = torch.sum(reference_probs * (reference_log_probs - current_log_probs), dim=-1).mean()

                mcts_policy_loss = torch.zeros((), device=device_torch)
                if policy_targets_tensor is not None and mcts_policy_coef > 0.0:
                    mcts_policy_loss = -torch.sum(
                        policy_targets_tensor * F.log_softmax(output.logits, dim=-1),
                        dim=-1,
                    ).mean()

                loss = (
                    policy_loss
                    + value_coef * value_loss
                    - entropy_coef * entropy_term
                    + reference_kl_coef * reference_kl
                    + mcts_policy_coef * mcts_policy_loss
                )

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

                last_policy_loss = float(policy_loss.item())
                last_value_loss = float(value_loss.item())
                last_entropy = float(entropy_term.item())
                last_approx_kl = float(((ratio - 1.0) - log_ratio).mean().item())
                last_reference_kl = float(reference_kl.item())
                last_mcts_policy_loss = float(mcts_policy_loss.item())

                if target_kl is not None and last_approx_kl > float(target_kl):
                    early_stop = True
                    break

            success_rate = float(np.mean(rolling_successes)) if rolling_successes else 0.0
            if writer is not None:
                writer.add_scalar("train/policy_loss", last_policy_loss, env_steps)
                writer.add_scalar("train/value_loss", last_value_loss, env_steps)
                writer.add_scalar("train/entropy", last_entropy, env_steps)
                writer.add_scalar("train/approx_kl", last_approx_kl, env_steps)
                writer.add_scalar("train/reference_kl", last_reference_kl, env_steps)
                writer.add_scalar("train/mcts_policy_loss", last_mcts_policy_loss, env_steps)
                writer.add_scalar("train/success_rate_100", success_rate, env_steps)

            full_success_window = len(rolling_successes) == rolling_successes.maxlen
            if full_success_window and success_rate >= best_success_rate:
                best_success_rate = success_rate
                save_ppo_checkpoint(
                    best_checkpoint_path,
                    model,
                    optimizer,
                    update,
                    env_steps,
                    episode_idx,
                    local_config,
                    training_stage=training_stage,
                )

            if update % checkpoint_interval == 0:
                save_ppo_checkpoint(
                    checkpoint_root / f"{checkpoint_prefix}_update_{update}.pt",
                    model,
                    optimizer,
                    update,
                    env_steps,
                    episode_idx,
                    local_config,
                    training_stage=training_stage,
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
        latest_checkpoint_path,
        model,
        optimizer,
        updates,
        env_steps,
        episode_idx,
        local_config,
        training_stage=training_stage,
    )
    if not best_checkpoint_path.exists():
        save_ppo_checkpoint(
            best_checkpoint_path,
            model,
            optimizer,
            updates,
            env_steps,
            episode_idx,
            local_config,
            training_stage=training_stage,
        )
    if writer is not None:
        writer.close()

    return {
        "episodes": episode_idx,
        "env_steps": env_steps,
        "best_success_rate_100": None if best_success_rate < 0.0 else best_success_rate,
        "checkpoint": str(latest_checkpoint_path),
        "best_checkpoint": str(best_checkpoint_path),
        "log_csv": str(csv_path),
        "training_stage": training_stage,
    }


def train_ppo(
    config: dict[str, Any],
    device: str = "cpu",
    map_size: int | None = None,
    checkpoint_dir: str | Path | None = None,
    log_dir: str | Path | None = None,
    initial_checkpoint: str | Path | None = None,
) -> dict[str, Any]:
    return _train_ppo_impl(
        config=config,
        device=device,
        map_size=map_size,
        checkpoint_dir=checkpoint_dir,
        log_dir=log_dir,
        initial_checkpoint=initial_checkpoint,
        use_mcts=False,
    )


def train_ppo_mcts(
    config: dict[str, Any],
    device: str = "cpu",
    map_size: int | None = None,
    checkpoint_dir: str | Path | None = None,
    log_dir: str | Path | None = None,
    initial_checkpoint: str | Path | None = None,
) -> dict[str, Any]:
    return _train_ppo_impl(
        config=config,
        device=device,
        map_size=map_size,
        checkpoint_dir=checkpoint_dir,
        log_dir=log_dir,
        initial_checkpoint=initial_checkpoint,
        use_mcts=True,
    )
