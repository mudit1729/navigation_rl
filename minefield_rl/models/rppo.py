from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical


def layer_init(layer: nn.Module, std: float = np.sqrt(2.0), bias_const: float = 0.0) -> nn.Module:
    if isinstance(layer, (nn.Linear, nn.Conv2d)):
        nn.init.orthogonal_(layer.weight, std)
        if layer.bias is not None:
            nn.init.constant_(layer.bias, bias_const)
    return layer


@dataclass(slots=True)
class RPPOOutput:
    logits: torch.Tensor
    values: torch.Tensor
    hidden: torch.Tensor
    trunk: torch.Tensor


@dataclass(slots=True)
class RPPOActionOutput:
    actions: torch.Tensor
    log_probs: torch.Tensor
    entropy: torch.Tensor
    values: torch.Tensor
    hidden: torch.Tensor
    logits: torch.Tensor


class RecurrentPPOActorCritic(nn.Module):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        model_cfg = config.get("model", config)
        channels = model_cfg.get("cnn_channels", [16, 32, 32])
        obs_channels = model_cfg.get("obs_channels", 2)
        obs_size = model_cfg.get("obs_size", 9)
        feature_dim = model_cfg.get("feature_dim", 256)
        hidden_dim = model_cfg.get("hidden_dim", 256)
        gru_layers = model_cfg.get("gru_layers", 1)
        post_health_dim = model_cfg.get("post_health_dim", 128)
        action_dim = model_cfg.get("action_dim", 8)

        conv_layers: list[nn.Module] = []
        in_channels = obs_channels
        for out_channels in channels:
            conv_layers.extend(
                [
                    layer_init(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)),
                    nn.ReLU(inplace=True),
                ]
            )
            in_channels = out_channels
        self.cnn = nn.Sequential(*conv_layers)
        self.feature_mlp = nn.Sequential(
            nn.Flatten(),
            layer_init(nn.Linear(channels[-1] * obs_size * obs_size, feature_dim)),
            nn.ReLU(inplace=True),
        )
        self.gru = nn.GRU(
            input_size=feature_dim,
            hidden_size=hidden_dim,
            num_layers=gru_layers,
            batch_first=True,
        )
        for name, param in self.gru.named_parameters():
            if "weight" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.constant_(param, 0.0)

        self.post_health = nn.Sequential(
            layer_init(nn.Linear(hidden_dim + 1, post_health_dim)),
            nn.ReLU(inplace=True),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(post_health_dim, 64)),
            nn.ReLU(inplace=True),
            layer_init(nn.Linear(64, action_dim), std=0.01),
        )
        self.critic = nn.Sequential(
            layer_init(nn.Linear(post_health_dim, 64)),
            nn.ReLU(inplace=True),
            layer_init(nn.Linear(64, 1), std=1.0),
        )

        self.hidden_dim = hidden_dim
        self.gru_layers = gru_layers
        self.action_dim = action_dim

    def init_hidden(self, batch_size: int, device: torch.device | None = None) -> torch.Tensor:
        hidden = torch.zeros(self.gru_layers, batch_size, self.hidden_dim)
        return hidden if device is None else hidden.to(device)

    def encode_observations(self, observations: torch.Tensor) -> torch.Tensor:
        if observations.dim() != 5:
            raise ValueError(f"Expected observations with shape [B, T, C, H, W], got {observations.shape}")
        batch_size, sequence_length = observations.shape[:2]
        flat = observations.reshape(batch_size * sequence_length, *observations.shape[2:])
        encoded = self.cnn(flat)
        features = self.feature_mlp(encoded)
        return features.reshape(batch_size, sequence_length, -1)

    def forward(
        self,
        observations: torch.Tensor,
        health: torch.Tensor,
        episode_starts: torch.Tensor | None = None,
        hidden: torch.Tensor | None = None,
    ) -> RPPOOutput:
        if observations.dim() == 4:
            observations = observations.unsqueeze(1)
        if health.dim() == 1:
            health = health.unsqueeze(0)
        if health.dim() == 2:
            health = health.unsqueeze(-1)

        batch_size, sequence_length = observations.shape[:2]
        if hidden is None:
            hidden = self.init_hidden(batch_size, observations.device)
        if episode_starts is None:
            episode_starts = torch.zeros(batch_size, sequence_length, device=observations.device)
        elif episode_starts.dim() == 1:
            episode_starts = episode_starts.unsqueeze(1)

        features = self.encode_observations(observations)
        outputs: list[torch.Tensor] = []
        current_hidden = hidden
        for step in range(sequence_length):
            reset_mask = (1.0 - episode_starts[:, step]).view(1, batch_size, 1)
            current_hidden = current_hidden * reset_mask
            step_out, current_hidden = self.gru(features[:, step : step + 1, :], current_hidden)
            outputs.append(step_out)

        rnn_out = torch.cat(outputs, dim=1)
        trunk_input = torch.cat([rnn_out, health], dim=-1)
        trunk = self.post_health(trunk_input)
        logits = self.actor(trunk)
        values = self.critic(trunk).squeeze(-1)
        return RPPOOutput(logits=logits, values=values, hidden=current_hidden, trunk=trunk)

    def forward_step(
        self,
        observation: torch.Tensor,
        health: torch.Tensor,
        episode_start: torch.Tensor | None = None,
        hidden: torch.Tensor | None = None,
    ) -> RPPOOutput:
        if observation.dim() == 3:
            observation = observation.unsqueeze(0)
        if health.dim() == 0:
            health = health.unsqueeze(0)
        if episode_start is None:
            episode_start = torch.zeros(observation.shape[0], device=observation.device)
        elif episode_start.dim() == 0:
            episode_start = episode_start.unsqueeze(0)

        output = self.forward(
            observation.unsqueeze(1),
            health.unsqueeze(1),
            episode_start.unsqueeze(1),
            hidden,
        )
        return RPPOOutput(
            logits=output.logits[:, 0, :],
            values=output.values[:, 0],
            hidden=output.hidden,
            trunk=output.trunk[:, 0, :],
        )

    def act(
        self,
        observation: torch.Tensor,
        health: torch.Tensor,
        episode_start: torch.Tensor | None = None,
        hidden: torch.Tensor | None = None,
        deterministic: bool = False,
    ) -> RPPOActionOutput:
        output = self.forward_step(observation, health, episode_start, hidden)
        dist = Categorical(logits=output.logits)
        actions = torch.argmax(output.logits, dim=-1) if deterministic else dist.sample()
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return RPPOActionOutput(
            actions=actions,
            log_probs=log_probs,
            entropy=entropy,
            values=output.values,
            hidden=output.hidden,
            logits=output.logits,
        )
