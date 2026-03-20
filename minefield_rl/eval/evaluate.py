from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
import torch

from minefield_rl.env.minefield_env import MinefieldEnv
from minefield_rl.models.drqn import DRQN
from minefield_rl.models.mcts import AlphaZeroMCTS
from minefield_rl.models.rppo import RecurrentPPOActorCritic
from minefield_rl.utils import ensure_dir


def load_model(
    config: dict[str, Any],
    checkpoint_path: str | Path | None,
    device: str,
    agent: str,
) -> torch.nn.Module:
    if agent == "ppo":
        model = RecurrentPPOActorCritic(config).to(device)
    else:
        model = DRQN(config).to(device)
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def evaluate_agent(
    config: dict[str, Any],
    checkpoint_path: str | Path | None,
    agent: str = "dqn",
    device: str = "cpu",
    episodes: int | None = None,
    map_size: int | None = None,
    simulations: int | None = None,
    log_dir: str | Path | None = None,
) -> dict[str, Any]:
    env_cfg = dict(config["env"])
    if map_size is not None:
        env_cfg["size"] = map_size
    env = MinefieldEnv.from_config({"env": env_cfg})

    total_episodes = int(episodes if episodes is not None else config["eval"].get("episodes", 50))
    log_root = ensure_dir(log_dir or "minefield_rl/logs")
    csv_path = log_root / f"eval_{agent}.csv"

    model = load_model(config, checkpoint_path, device, agent)
    search = AlphaZeroMCTS(model, config, device=device) if agent == "mcts" else None

    fieldnames = [
        "episode",
        "reward_raw",
        "reward_clipped",
        "length",
        "outcome",
        "health_remaining",
        "mine_hits",
        "map_seed",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        csv.DictWriter(handle, fieldnames=fieldnames).writeheader()

    successes = 0
    deaths = 0
    timeouts = 0
    reward_raw: list[float] = []
    reward_clipped: list[float] = []
    lengths: list[int] = []
    mine_hits: list[int] = []
    remaining_health: list[int] = []

    for episode_idx in range(1, total_episodes + 1):
        observation, info = env.reset()
        terminated = False
        truncated = False
        hidden_torch = model.init_hidden(1, torch.device(device)) if agent in {"dqn", "ppo"} else None
        hidden_mcts: np.ndarray | None = None
        episode_start = torch.ones(1, dtype=torch.float32, device=device)

        while not (terminated or truncated):
            if agent == "dqn":
                normalized_health = info["health"] / env.max_health
                obs_tensor = torch.from_numpy(observation).float().unsqueeze(0).to(device)
                health_tensor = torch.tensor([[normalized_health]], dtype=torch.float32, device=device)
                with torch.no_grad():
                    output = model.forward_step(obs_tensor, health_tensor, hidden_torch)
                    q_values = output.q_values[0].cpu().numpy()
                    hidden_torch = output.hidden
                action = int(np.argmax(q_values))
                episode_start = torch.zeros(1, dtype=torch.float32, device=device)
            elif agent == "ppo":
                normalized_health = info["health"] / env.max_health
                obs_tensor = torch.from_numpy(observation).float().unsqueeze(0).to(device)
                health_tensor = torch.tensor([normalized_health], dtype=torch.float32, device=device)
                with torch.no_grad():
                    action_out = model.act(
                        obs_tensor,
                        health_tensor,
                        episode_start=episode_start,
                        hidden=hidden_torch,
                        deterministic=True,
                    )
                    action = int(action_out.actions[0].cpu().item())
                    hidden_torch = action_out.hidden
                episode_start = torch.zeros(1, dtype=torch.float32, device=device)
            elif agent == "mcts" and search is not None:
                result = search.search(
                    env=env,
                    observation=observation,
                    health=info["health"],
                    hidden_in=hidden_mcts,
                    simulations=simulations or int(config["mcts"]["simulations_eval"]),
                    training=False,
                )
                action = result.chosen_action
                hidden_mcts = result.root_hidden
            else:
                raise ValueError(f"Unsupported agent type: {agent}")

            observation, _, terminated, truncated, info = env.step(action)

        reward_raw.append(float(info["episode_reward_raw"]))
        reward_clipped.append(float(info["episode_reward_clipped"]))
        lengths.append(int(info["steps"]))
        mine_hits.append(int(info["mine_hits"]))
        remaining_health.append(int(info["health"]))
        successes += int(info["outcome"] == "success")
        deaths += int(info["outcome"] == "death")
        timeouts += int(info["outcome"] == "timeout")

        with csv_path.open("a", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=fieldnames).writerow(
                {
                    "episode": episode_idx,
                    "reward_raw": info["episode_reward_raw"],
                    "reward_clipped": info["episode_reward_clipped"],
                    "length": info["steps"],
                    "outcome": info["outcome"],
                    "health_remaining": info["health"],
                    "mine_hits": info["mine_hits"],
                    "map_seed": info["map_seed"],
                }
            )

    return {
        "agent": agent,
        "episodes": total_episodes,
        "success_rate": successes / max(total_episodes, 1),
        "death_rate": deaths / max(total_episodes, 1),
        "timeout_rate": timeouts / max(total_episodes, 1),
        "avg_reward_raw": float(np.mean(reward_raw)) if reward_raw else 0.0,
        "avg_reward_clipped": float(np.mean(reward_clipped)) if reward_clipped else 0.0,
        "avg_length": float(np.mean(lengths)) if lengths else 0.0,
        "avg_mine_hits": float(np.mean(mine_hits)) if mine_hits else 0.0,
        "avg_health_remaining": float(np.mean(remaining_health)) if remaining_health else 0.0,
        "log_csv": str(csv_path),
    }
