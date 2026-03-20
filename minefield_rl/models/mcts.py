from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from minefield_rl.env.minefield_env import MinefieldEnv
from minefield_rl.models.drqn import DRQN
from minefield_rl.models.rppo import RecurrentPPOActorCritic


@dataclass(slots=True)
class SearchNode:
    snapshot: Any
    observation: np.ndarray
    health: float
    hidden: np.ndarray
    q_values: np.ndarray
    state_value: float
    terminal: bool
    outcome: str | None
    reward_from_parent: float = 0.0
    priors: np.ndarray | None = None
    children: dict[int, "SearchNode"] = field(default_factory=dict)
    visit_count: int = 0
    action_visits: np.ndarray = field(default_factory=lambda: np.zeros(8, dtype=np.float32))
    action_value_sums: np.ndarray = field(default_factory=lambda: np.zeros(8, dtype=np.float32))

    def expanded(self) -> bool:
        return self.priors is not None

    def mean_action_values(self) -> np.ndarray:
        safe_denominator = np.maximum(self.action_visits, 1.0)
        return self.action_value_sums / safe_denominator


@dataclass(slots=True)
class SearchResult:
    policy: np.ndarray
    visit_counts: np.ndarray
    chosen_action: int
    root_hidden: np.ndarray
    root_q_values: np.ndarray
    root_value: float
    dqn_action: int
    mcts_action: int
    simulations: int
    root_priors: np.ndarray


class AlphaZeroMCTS:
    def __init__(self, model: DRQN | RecurrentPPOActorCritic, config: dict[str, Any], device: str = "cpu") -> None:
        self.model = model
        self.config = config
        self.mcts_cfg = config.get("mcts", {})
        self.device = torch.device(device)
        self.c_puct = float(self.mcts_cfg.get("c_puct", 1.25))
        self.max_depth = int(self.mcts_cfg.get("max_depth", 10))
        self.gamma = float(self.mcts_cfg.get("gamma", 0.99))
        self.dirichlet_alpha = float(self.mcts_cfg.get("dirichlet_alpha", 0.3))
        self.dirichlet_epsilon = float(self.mcts_cfg.get("dirichlet_epsilon", 0.25))
        self.policy_temperature = float(self.mcts_cfg.get("policy_temperature", 1.0))

    def search(
        self,
        env: MinefieldEnv,
        observation: np.ndarray,
        health: int,
        hidden_in: np.ndarray | None = None,
        simulations: int | None = None,
        training: bool = False,
    ) -> SearchResult:
        simulation_budget = int(
            simulations
            if simulations is not None
            else self.mcts_cfg.get("simulations_train" if training else "simulations_eval", 20)
        )
        root_q, root_value, root_hidden = self._evaluate_state(
            observation,
            health / env.max_health,
            hidden_in,
        )
        root = SearchNode(
            snapshot=env.snapshot(),
            observation=np.array(observation, copy=True),
            health=float(health / env.max_health),
            hidden=root_hidden,
            q_values=root_q,
            state_value=root_value,
            terminal=False,
            outcome=None,
        )
        self._expand(root, add_dirichlet_noise=training)

        for _ in range(simulation_budget):
            self._simulate(root)

        visit_counts = np.array(root.action_visits, copy=True)
        if visit_counts.sum() <= 0:
            policy = np.array(root.priors, copy=True)
        else:
            policy = visit_counts / visit_counts.sum()

        if training:
            tempered = np.power(policy + 1e-8, 1.0 / max(self.policy_temperature, 1e-6))
            tempered /= tempered.sum()
            chosen_action = int(np.random.choice(len(tempered), p=tempered))
        else:
            chosen_action = int(np.argmax(visit_counts if visit_counts.sum() > 0 else root.priors))

        return SearchResult(
            policy=policy,
            visit_counts=visit_counts,
            chosen_action=chosen_action,
            root_hidden=root_hidden,
            root_q_values=root_q,
            root_value=root_value,
            dqn_action=int(np.argmax(root_q)),
            mcts_action=chosen_action,
            simulations=simulation_budget,
            root_priors=np.array(root.priors, copy=True),
        )

    def _simulate(self, root: SearchNode) -> None:
        node = root
        search_path: list[tuple[SearchNode, int, float]] = []
        depth = 0

        while node.expanded() and not node.terminal and depth < self.max_depth:
            action = self._select_action(node)
            if action in node.children:
                child = node.children[action]
                info_reward = child.reward_from_parent
            else:
                child, info_reward = self._expand_child(node, action)
                node.children[action] = child
            search_path.append((node, action, info_reward))
            node = child
            if not node.expanded() and not node.terminal:
                self._expand(node, add_dirichlet_noise=False)
                break
            depth += 1

        value = self._leaf_value(node)
        node.visit_count += 1
        for parent, action, reward in reversed(search_path):
            value = reward + self.gamma * value
            parent.visit_count += 1
            parent.action_visits[action] += 1.0
            parent.action_value_sums[action] += value

    def _select_action(self, node: SearchNode) -> int:
        priors = node.priors if node.priors is not None else self._policy_from_q(node.q_values)
        q_values = node.mean_action_values()
        sqrt_visits = np.sqrt(max(node.visit_count, 1))
        exploration = self.c_puct * priors * sqrt_visits / (1.0 + node.action_visits)
        scores = q_values + exploration
        return int(np.argmax(scores))

    def _expand(self, node: SearchNode, add_dirichlet_noise: bool) -> None:
        priors = self._policy_from_q(node.q_values)
        if add_dirichlet_noise:
            noise = np.random.dirichlet([self.dirichlet_alpha] * len(priors))
            priors = (1.0 - self.dirichlet_epsilon) * priors + self.dirichlet_epsilon * noise
        node.priors = priors.astype(np.float32)

    def _expand_child(self, parent: SearchNode, action: int) -> tuple[SearchNode, float]:
        sim_env = MinefieldEnv.from_config(self.config)
        sim_env.restore(parent.snapshot)
        child_obs, reward, terminated, truncated, info = sim_env.step(action)
        if terminated or truncated:
            child = SearchNode(
                snapshot=sim_env.snapshot(),
                observation=np.array(child_obs, copy=True),
                health=float(info["health"] / sim_env.max_health),
                hidden=np.array(parent.hidden, copy=True),
                q_values=np.zeros_like(parent.q_values),
                state_value=self._terminal_value(info.get("outcome")),
                terminal=True,
                outcome=info.get("outcome"),
                reward_from_parent=float(reward),
            )
        else:
            child_q, child_value, child_hidden = self._evaluate_state(
                child_obs,
                info["health"] / sim_env.max_health,
                parent.hidden,
            )
            child = SearchNode(
                snapshot=sim_env.snapshot(),
                observation=np.array(child_obs, copy=True),
                health=float(info["health"] / sim_env.max_health),
                hidden=child_hidden,
                q_values=child_q,
                state_value=child_value,
                terminal=False,
                outcome=None,
                reward_from_parent=float(reward),
            )
        return child, float(reward)

    def _leaf_value(self, node: SearchNode) -> float:
        if node.terminal:
            return float(node.state_value)
        return float(node.state_value)

    def _policy_from_q(self, q_values: np.ndarray) -> np.ndarray:
        scaled = q_values / max(self.policy_temperature, 1e-6)
        scaled -= np.max(scaled)
        exp_values = np.exp(scaled)
        return exp_values / np.clip(exp_values.sum(), 1e-8, None)

    def _terminal_value(self, outcome: str | None) -> float:
        if outcome == "success":
            return 1.0
        if outcome == "death":
            return -1.0
        if outcome == "timeout":
            return -0.2
        return 0.0

    def _evaluate_state(
        self,
        observation: np.ndarray,
        normalized_health: float,
        hidden_in: np.ndarray | None,
    ) -> tuple[np.ndarray, float, np.ndarray]:
        self.model.eval()
        with torch.no_grad():
            obs_tensor = torch.from_numpy(observation).float().unsqueeze(0).to(self.device)
            hidden_tensor = None
            if hidden_in is not None:
                hidden_tensor = torch.from_numpy(hidden_in).float().to(self.device)
            if isinstance(self.model, RecurrentPPOActorCritic):
                health_tensor = torch.tensor([normalized_health], dtype=torch.float32, device=self.device)
                episode_start = torch.zeros(1, dtype=torch.float32, device=self.device)
                output = self.model.forward_step(
                    obs_tensor,
                    health_tensor,
                    episode_start=episode_start,
                    hidden=hidden_tensor,
                )
                q_values = output.logits[0].detach().cpu().numpy().astype(np.float32)
                value = float(output.values[0].detach().cpu().item())
                hidden = output.hidden.detach().cpu().numpy().astype(np.float32)
            else:
                health_tensor = torch.tensor([[normalized_health]], dtype=torch.float32, device=self.device)
                output = self.model.forward_step(obs_tensor, health_tensor, hidden_tensor)
                q_values = output.q_values[0].detach().cpu().numpy().astype(np.float32)
                value = float(output.values[0].detach().cpu().item())
                hidden = output.hidden.detach().cpu().numpy().astype(np.float32)
        return q_values, value, hidden
