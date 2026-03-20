from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(slots=True)
class DRQNOutput:
    q_values: torch.Tensor
    values: torch.Tensor
    hidden: torch.Tensor
    trunk: torch.Tensor


class DRQN(nn.Module):
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
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                ]
            )
            in_channels = out_channels
        self.cnn = nn.Sequential(*conv_layers)
        self.feature_mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels[-1] * obs_size * obs_size, feature_dim),
            nn.ReLU(inplace=True),
        )
        self.gru = nn.GRU(
            input_size=feature_dim,
            hidden_size=hidden_dim,
            num_layers=gru_layers,
            batch_first=True,
        )
        self.post_health = nn.Sequential(
            nn.Linear(hidden_dim + 1, post_health_dim),
            nn.ReLU(inplace=True),
        )

        self.value_stream = nn.Sequential(
            nn.Linear(post_health_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(post_health_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, action_dim),
        )
        self.value_head = nn.Sequential(
            nn.Linear(post_health_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Tanh(),
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
        hidden: torch.Tensor | None = None,
    ) -> DRQNOutput:
        if observations.dim() == 4:
            observations = observations.unsqueeze(1)
        if health.dim() == 1:
            health = health.unsqueeze(0).unsqueeze(-1)
        elif health.dim() == 2:
            health = health.unsqueeze(-1)

        batch_size = observations.shape[0]
        if hidden is None:
            hidden = self.init_hidden(batch_size, observations.device)

        features = self.encode_observations(observations)
        rnn_out, next_hidden = self.gru(features, hidden)
        fused = torch.cat([rnn_out, health], dim=-1)
        trunk = self.post_health(fused)

        value = self.value_stream(trunk)
        advantage = self.advantage_stream(trunk)
        q_values = value + advantage - advantage.mean(dim=-1, keepdim=True)
        values = self.value_head(trunk).squeeze(-1)
        return DRQNOutput(
            q_values=q_values,
            values=values,
            hidden=next_hidden,
            trunk=trunk,
        )

    def forward_step(
        self,
        observation: torch.Tensor,
        health: torch.Tensor,
        hidden: torch.Tensor | None = None,
    ) -> DRQNOutput:
        if observation.dim() == 3:
            observation = observation.unsqueeze(0)
        if health.dim() == 0:
            health = health.unsqueeze(0).unsqueeze(0)
        elif health.dim() == 1:
            health = health.unsqueeze(-1)

        output = self.forward(observation.unsqueeze(1), health.unsqueeze(1), hidden)
        return DRQNOutput(
            q_values=output.q_values[:, 0, :],
            values=output.values[:, 0],
            hidden=output.hidden,
            trunk=output.trunk[:, 0, :],
        )

    @staticmethod
    def policy_from_q(q_values: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        return F.softmax(q_values / temperature, dim=-1)
