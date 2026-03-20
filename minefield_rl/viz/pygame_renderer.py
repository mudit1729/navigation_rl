from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from minefield_rl.constants import ACTION_NAMES, CellType
from minefield_rl.env.minefield_env import MinefieldEnv
from minefield_rl.models.drqn import DRQN
from minefield_rl.models.mcts import AlphaZeroMCTS
from minefield_rl.models.rppo import RecurrentPPOActorCritic
from minefield_rl.viz.charts import draw_bar_chart, draw_line_chart
from minefield_rl.viz.heatmap import normalize_heatmap, visit_counts_to_cell_heatmap

try:
    import pygame
except Exception as exc:  # pragma: no cover
    raise ImportError("pygame is required for the playable renderer") from exc


class PygameRenderer:
    BG = (17, 22, 30)
    PANEL = (25, 32, 44)
    GRID_BG = (21, 26, 35)
    TEXT = (236, 240, 244)
    SUBTLE = (167, 176, 190)
    EMPTY = (208, 214, 220)
    WALL = (75, 84, 98)
    MINE = (196, 74, 74)
    START = (249, 187, 88)
    EXIT = (77, 180, 116)
    AGENT = (73, 132, 255)
    FOG = (70, 74, 82)
    VISIBLE_OVERLAY = (91, 148, 255, 70)

    def __init__(
        self,
        config: dict[str, Any],
        agent: str = "human",
        checkpoint_path: str | None = None,
        device: str = "cpu",
        map_size: int | None = None,
    ) -> None:
        self.config = config
        self.device = device
        self.checkpoint_path = checkpoint_path
        env_cfg = dict(config["env"])
        if map_size is not None:
            env_cfg["size"] = map_size
        self.env = MinefieldEnv.from_config({"env": env_cfg})
        self.mode = "human" if agent == "human" else agent
        self.scenario_label = str(config.get("env", {}).get("scenario_label", f"{env_cfg['size']}x{env_cfg['size']} Custom"))
        self.show_full_map = False
        self.reward_history: list[float] = []
        self.current_q_values = np.zeros(len(ACTION_NAMES), dtype=np.float32)
        self.current_visit_counts = np.zeros(len(ACTION_NAMES), dtype=np.float32)
        self.current_simulations = 0
        self.current_dqn_action: int | None = None
        self.current_mcts_action: int | None = None
        self.episode = 0
        self.rolling_results: list[int] = []
        self.episode_end_time: float | None = None
        self.bar_chart_title = "Q-Values"
        self.checkpoint_payload: dict[str, Any] | None = None
        if self.checkpoint_path is not None:
            self.checkpoint_payload = torch.load(self.checkpoint_path, map_location=device)

        self.model: DRQN | RecurrentPPOActorCritic | None = None
        self.search: AlphaZeroMCTS | None = None
        self._load_model_for_mode(self.mode)
        self.hidden_torch = self.model.init_hidden(1, torch.device(device)) if self.mode in {"dqn", "ppo"} and self.model else None
        self.hidden_mcts: np.ndarray | None = None

        pygame.init()
        pygame.display.set_caption("Minefield Navigator")
        viz_cfg = config["viz"]
        self.window = pygame.display.set_mode((viz_cfg["window_width"], viz_cfg["window_height"]))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("arial", 18)
        self.small_font = pygame.font.SysFont("arial", 14)
        self.big_font = pygame.font.SysFont("arial", 28, bold=True)

        self.speed_modes = list(viz_cfg["speed_modes"].items())
        self.speed_index = 0 if self.mode == "human" else len(self.speed_modes) - 1
        self.last_auto_step = 0.0
        self.auto_reset_delay_ms = int(viz_cfg["auto_reset_delay_ms"])
        self.reward_history_limit = int(viz_cfg["reward_history"])

        self.observation, self.info = self.env.reset()
        self.episode += 1
        self.buttons = self._build_buttons()

    def _build_buttons(self) -> list[dict[str, Any]]:
        y = self.window.get_height() - 58
        labels = [("Human", "human"), ("DQN", "dqn"), ("PPO", "ppo"), ("MCTS", "mcts")]
        buttons: list[dict[str, Any]] = []
        for index, (label, mode) in enumerate(labels):
            buttons.append(
                {
                    "label": label,
                    "mode": mode,
                    "rect": pygame.Rect(30 + index * 102, y, 92, 34),
                }
            )
        for index, (label, delay) in enumerate(self.speed_modes):
            buttons.append(
                {
                    "label": label,
                    "speed": index,
                    "rect": pygame.Rect(500 + index * 110, y, 96, 34),
                }
            )
        return buttons

    def run(self) -> None:
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    running = self._handle_key(event.key)
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    self._handle_click(event.pos)

            if self.mode in {"dqn", "ppo", "mcts"}:
                self._auto_step_if_needed()
            elif self.episode_end_time is not None and self._should_auto_reset():
                self._reset_episode()

            self._draw()
            self.clock.tick(int(self.config["viz"]["fps"]))

        pygame.quit()

    def _handle_key(self, key: int) -> bool:
        if key == pygame.K_ESCAPE:
            return False
        if key == pygame.K_r:
            self._reset_episode()
            return True
        if key == pygame.K_m:
            self.show_full_map = not self.show_full_map
            return True
        if key == pygame.K_s:
            self._save_screenshot()
            return True
        if key == pygame.K_1:
            self._set_mode("human")
            return True
        if key == pygame.K_2:
            self._set_mode("dqn")
            return True
        if key == pygame.K_3:
            self._set_mode("ppo")
            return True
        if key == pygame.K_4:
            self._set_mode("mcts")
            return True
        if key == pygame.K_5:
            self.speed_index = 0
            return True
        if key == pygame.K_6:
            self.speed_index = min(1, len(self.speed_modes) - 1)
            return True
        if key == pygame.K_7:
            self.speed_index = len(self.speed_modes) - 1
            return True

        if self.mode == "human":
            action = self._key_to_action(key)
            if action is not None:
                self._step_env(action)
        return True

    def _handle_click(self, position: tuple[int, int]) -> None:
        for button in self.buttons:
            if button["rect"].collidepoint(position):
                if "mode" in button:
                    self._set_mode(button["mode"])
                elif "speed" in button:
                    self.speed_index = int(button["speed"])
                break

    def _load_model_for_mode(self, mode: str) -> None:
        if mode in {"dqn", "mcts"}:
            if not isinstance(self.model, DRQN):
                self.model = DRQN(self.config).to(self.device)
                if self._checkpoint_is_compatible({"dqn", "mcts"}):
                    self.model.load_state_dict(self.checkpoint_payload["model_state_dict"])
                self.model.eval()
            self.bar_chart_title = "Q-Values"
            if mode == "mcts":
                self.search = AlphaZeroMCTS(self.model, self.config, device=self.device)
            else:
                self.search = None
        elif mode == "ppo":
            if not isinstance(self.model, RecurrentPPOActorCritic):
                self.model = RecurrentPPOActorCritic(self.config).to(self.device)
                if self._checkpoint_is_compatible({"ppo"}):
                    self.model.load_state_dict(self.checkpoint_payload["model_state_dict"])
                self.model.eval()
            self.search = None
            self.bar_chart_title = "Action Probabilities"
        else:
            self.search = None

    def _checkpoint_is_compatible(self, expected_agents: set[str]) -> bool:
        if self.checkpoint_payload is None:
            return False
        checkpoint_agent = self.checkpoint_payload.get("agent")
        if checkpoint_agent is None and {"dqn", "mcts"} & expected_agents:
            return True
        return checkpoint_agent in expected_agents

    def _set_mode(self, mode: str) -> None:
        self.mode = mode
        self._load_model_for_mode(mode)
        self.hidden_torch = self.model.init_hidden(1, torch.device(self.device)) if mode in {"dqn", "ppo"} and self.model else None
        self.hidden_mcts = None
        self.speed_index = 0 if mode == "human" else len(self.speed_modes) - 1
        self.current_q_values = np.zeros(len(ACTION_NAMES), dtype=np.float32)
        self.current_visit_counts = np.zeros(len(ACTION_NAMES), dtype=np.float32)

    def _reset_episode(self) -> None:
        self.observation, self.info = self.env.reset()
        self.episode += 1
        self.current_q_values = np.zeros(len(ACTION_NAMES), dtype=np.float32)
        self.current_visit_counts = np.zeros(len(ACTION_NAMES), dtype=np.float32)
        self.current_simulations = 0
        self.current_dqn_action = None
        self.current_mcts_action = None
        self.hidden_torch = self.model.init_hidden(1, torch.device(self.device)) if self.mode in {"dqn", "ppo"} and self.model else None
        self.hidden_mcts = None
        self.episode_end_time = None

    def _should_auto_reset(self) -> bool:
        if self.episode_end_time is None:
            return False
        return (time.time() - self.episode_end_time) * 1000.0 >= self.auto_reset_delay_ms

    def _auto_step_if_needed(self) -> None:
        delay_ms = int(self.speed_modes[self.speed_index][1])
        now = time.time()
        if delay_ms > 0 and (now - self.last_auto_step) * 1000.0 < delay_ms:
            return
        self.last_auto_step = now

        if self.episode_end_time is not None:
            if self._should_auto_reset():
                self._reset_episode()
            return

        if self.mode == "dqn" and self.model is not None and self.hidden_torch is not None:
            normalized_health = self.info["health"] / self.env.max_health
            obs_tensor = torch.from_numpy(self.observation).float().unsqueeze(0).to(self.device)
            health_tensor = torch.tensor([[normalized_health]], dtype=torch.float32, device=self.device)
            with torch.no_grad():
                output = self.model.forward_step(obs_tensor, health_tensor, self.hidden_torch)
                self.hidden_torch = output.hidden
                self.current_q_values = output.q_values[0].cpu().numpy()
            action = int(np.argmax(self.current_q_values))
            self.current_dqn_action = action
            self.current_mcts_action = None
            self.current_visit_counts.fill(0.0)
            self.current_simulations = 0
            self._step_env(action)

        if self.mode == "ppo" and isinstance(self.model, RecurrentPPOActorCritic) and self.hidden_torch is not None:
            normalized_health = self.info["health"] / self.env.max_health
            obs_tensor = torch.from_numpy(self.observation).float().unsqueeze(0).to(self.device)
            health_tensor = torch.tensor([normalized_health], dtype=torch.float32, device=self.device)
            episode_start = torch.tensor([1.0 if self.info["steps"] == 0 else 0.0], dtype=torch.float32, device=self.device)
            with torch.no_grad():
                action_out = self.model.act(
                    obs_tensor,
                    health_tensor,
                    episode_start=episode_start,
                    hidden=self.hidden_torch,
                    deterministic=True,
                )
                self.hidden_torch = action_out.hidden
                self.current_q_values = torch.softmax(action_out.logits[0], dim=-1).cpu().numpy()
            action = int(action_out.actions[0].cpu().item())
            self.current_dqn_action = action
            self.current_mcts_action = None
            self.current_visit_counts.fill(0.0)
            self.current_simulations = 0
            self._step_env(action)

        if self.mode == "mcts" and self.search is not None:
            result = self.search.search(
                env=self.env,
                observation=self.observation,
                health=self.info["health"],
                hidden_in=self.hidden_mcts,
                simulations=int(self.config["mcts"]["simulations_eval"]),
                training=False,
            )
            self.hidden_mcts = result.root_hidden
            self.current_q_values = result.root_q_values
            self.current_visit_counts = result.visit_counts
            self.current_simulations = result.simulations
            self.current_dqn_action = result.dqn_action
            self.current_mcts_action = result.mcts_action
            self._step_env(result.chosen_action)

    def _step_env(self, action: int) -> None:
        self.observation, _, terminated, truncated, self.info = self.env.step(action)
        if terminated or truncated:
            self.reward_history.append(float(self.info["episode_reward_raw"]))
            self.reward_history = self.reward_history[-self.reward_history_limit :]
            self.rolling_results.append(1 if self.info["outcome"] == "success" else 0)
            self.rolling_results = self.rolling_results[-100:]
            self.episode_end_time = time.time()

    def _save_screenshot(self) -> None:
        output = Path("minefield_rl/logs") / f"screenshot_ep_{self.episode}.png"
        output.parent.mkdir(parents=True, exist_ok=True)
        pygame.image.save(self.window, str(output))

    def _key_to_action(self, key: int) -> int | None:
        mapping = {
            pygame.K_UP: 0,
            pygame.K_w: 0,
            pygame.K_e: 1,
            pygame.K_PAGEUP: 1,
            pygame.K_RIGHT: 2,
            pygame.K_d: 2,
            pygame.K_c: 3,
            pygame.K_PAGEDOWN: 3,
            pygame.K_DOWN: 4,
            pygame.K_s: 4,
            pygame.K_z: 5,
            pygame.K_END: 5,
            pygame.K_LEFT: 6,
            pygame.K_a: 6,
            pygame.K_q: 7,
            pygame.K_HOME: 7,
        }
        return mapping.get(key)

    def _draw(self) -> None:
        self.window.fill(self.BG)
        grid_rect = pygame.Rect(24, 24, 720, 720)
        info_rect = pygame.Rect(768, 24, 400, 720)
        pygame.draw.rect(self.window, self.GRID_BG, grid_rect, border_radius=14)
        pygame.draw.rect(self.window, self.PANEL, info_rect, border_radius=14)

        self._draw_grid(grid_rect)
        self._draw_info_panel(info_rect)
        self._draw_buttons()

        if self.episode_end_time is not None:
            self._draw_end_overlay(grid_rect)

        pygame.display.flip()

    def _draw_grid(self, rect: pygame.Rect) -> None:
        title_surface = self.big_font.render("Visible Map", True, self.TEXT)
        self.window.blit(title_surface, (rect.x + 18, rect.y + 14))
        map_rect = pygame.Rect(rect.x + 16, rect.y + 56, rect.width - 32, rect.height - 72)
        self._draw_map(
            map_rect,
            reveal_all=self.show_full_map,
            show_mcts_heat=True,
            show_visible_overlay=False,
            show_trajectory=self.episode_end_time is not None,
        )

    def _draw_map(
        self,
        rect: pygame.Rect,
        *,
        reveal_all: bool,
        show_mcts_heat: bool,
        show_visible_overlay: bool,
        show_trajectory: bool,
    ) -> None:
        full_state = self.env.full_state()
        grid = full_state["grid"]
        visible_mask = full_state["visible_mask"]
        size = self.env.size
        cell_size = max(1, min(rect.width // size, rect.height // size))
        x_offset = rect.x + (rect.width - cell_size * size) // 2
        y_offset = rect.y + (rect.height - cell_size * size) // 2

        trajectory_heat = normalize_heatmap(full_state["trajectory_counts"])
        mcts_heat = visit_counts_to_cell_heatmap(self.env.agent_pos, self.current_visit_counts, grid.shape)
        for row in range(size):
            for col in range(size):
                cell_rect = pygame.Rect(x_offset + col * cell_size, y_offset + row * cell_size, cell_size - 1, cell_size - 1)
                visible = reveal_all or bool(visible_mask[row, col])
                if not visible:
                    color = self.FOG
                else:
                    cell_type = CellType(grid[row, col])
                    color = self.EMPTY
                    if cell_type == CellType.WALL:
                        color = self.WALL
                    elif cell_type == CellType.MINE:
                        color = self.MINE
                    elif cell_type == CellType.START:
                        color = self.START
                    elif cell_type == CellType.EXIT:
                        color = self.EXIT
                pygame.draw.rect(self.window, color, cell_rect)

                if show_mcts_heat and self.mode == "mcts" and mcts_heat[row, col] > 0 and visible:
                    overlay = pygame.Surface((cell_rect.width, cell_rect.height), pygame.SRCALPHA)
                    overlay.fill((255, 200, 80, int(150 * mcts_heat[row, col])))
                    self.window.blit(overlay, cell_rect.topleft)

                if show_visible_overlay and visible_mask[row, col]:
                    overlay = pygame.Surface((cell_rect.width, cell_rect.height), pygame.SRCALPHA)
                    overlay.fill(self.VISIBLE_OVERLAY)
                    self.window.blit(overlay, cell_rect.topleft)

                if show_trajectory and trajectory_heat[row, col] > 0:
                    overlay = pygame.Surface((cell_rect.width, cell_rect.height), pygame.SRCALPHA)
                    overlay.fill((73, 132, 255, int(110 * trajectory_heat[row, col])))
                    self.window.blit(overlay, cell_rect.topleft)

        agent_row, agent_col = self.env.agent_pos
        center = (
            x_offset + agent_col * cell_size + cell_size // 2,
            y_offset + agent_row * cell_size + cell_size // 2,
        )
        pygame.draw.circle(self.window, self.AGENT, center, max(2, min(6, cell_size // 2 + 1)))

    def _draw_info_panel(self, rect: pygame.Rect) -> None:
        self.window.blit(self.big_font.render("Minefield Navigator", True, self.TEXT), (rect.x + 20, rect.y + 16))
        preview_rect = pygame.Rect(rect.x + 20, rect.y + 58, rect.width - 40, 268)
        pygame.draw.rect(self.window, (30, 38, 52), preview_rect, border_radius=10)
        pygame.draw.rect(self.window, (80, 90, 105), preview_rect, width=1, border_radius=10)
        self.window.blit(self.font.render("Full Map", True, self.TEXT), (preview_rect.x + 10, preview_rect.y + 8))
        self._draw_map(
            pygame.Rect(preview_rect.x + 10, preview_rect.y + 32, preview_rect.width - 20, preview_rect.height - 42),
            reveal_all=True,
            show_mcts_heat=False,
            show_visible_overlay=True,
            show_trajectory=self.episode_end_time is not None,
        )

        success_rate = float(np.mean(self.rolling_results)) if self.rolling_results else 0.0
        lines = [
            f"Mode: {self.mode.upper()}",
            f"Scenario: {self.scenario_label}",
            f"Map: {self.env.size}x{self.env.size}",
            f"Health: {' '.join(['HP'] * self.info['health'])}",
            f"Steps: {self.info['steps']}/{self.info['max_steps']}",
            f"Episode: {self.episode}",
            f"Map Seed: {self.info['map_seed']}",
            f"Success Rate: {success_rate * 100.0:5.1f}%",
            f"Outcome: {self.info['outcome'] or 'running'}",
            f"Mine Hits: {self.info['mine_hits']}",
        ]
        if self.mode == "mcts":
            lines.extend(
                [
                    f"Simulations: {self.current_simulations}",
                    f"DQN Action: {ACTION_NAMES[self.current_dqn_action] if self.current_dqn_action is not None else '-'}",
                    f"MCTS Action: {ACTION_NAMES[self.current_mcts_action] if self.current_mcts_action is not None else '-'}",
                ]
            )

        stats_top = preview_rect.bottom + 16
        split_index = (len(lines) + 1) // 2
        left_lines = lines[:split_index]
        right_lines = lines[split_index:]
        left_x = rect.x + 20
        right_x = rect.centerx + 8
        for index, line in enumerate(left_lines):
            self.window.blit(self.small_font.render(line, True, self.TEXT), (left_x, stats_top + index * 22))
        for index, line in enumerate(right_lines):
            self.window.blit(self.small_font.render(line, True, self.TEXT), (right_x, stats_top + index * 22))

        reward_chart_rect = pygame.Rect(rect.x + 20, rect.bottom - 188, rect.width - 40, 76)
        draw_line_chart(
            self.window,
            reward_chart_rect,
            self.reward_history[-self.reward_history_limit :],
            color=(73, 132, 255),
            bg_color=(30, 38, 52),
            label="Episode Reward (raw)",
            font=self.small_font,
        )

        bar_chart_rect = pygame.Rect(rect.x + 20, rect.bottom - 98, rect.width - 40, 76)
        highlight = self.current_dqn_action if self.mode == "dqn" else self.current_mcts_action
        draw_bar_chart(
            self.window,
            bar_chart_rect,
            ACTION_NAMES,
            self.current_q_values.tolist(),
            highlight_idx=highlight,
            font=self.small_font,
            title=self.bar_chart_title,
        )

    def _draw_buttons(self) -> None:
        for button in self.buttons:
            active = False
            if "mode" in button:
                active = self.mode == button["mode"]
            elif "speed" in button:
                active = self.speed_index == button["speed"]
            color = (73, 132, 255) if active else (40, 48, 62)
            pygame.draw.rect(self.window, color, button["rect"], border_radius=8)
            pygame.draw.rect(self.window, (110, 120, 138), button["rect"], width=1, border_radius=8)
            label_surface = self.font.render(button["label"], True, self.TEXT)
            self.window.blit(
                label_surface,
                (
                    button["rect"].centerx - label_surface.get_width() // 2,
                    button["rect"].centery - label_surface.get_height() // 2,
                ),
            )

    def _draw_end_overlay(self, grid_rect: pygame.Rect) -> None:
        overlay = pygame.Surface((grid_rect.width, grid_rect.height), pygame.SRCALPHA)
        overlay.fill((12, 14, 20, 150))
        self.window.blit(overlay, grid_rect.topleft)
        title = "SUCCESS" if self.info["outcome"] == "success" else self.info["outcome"].upper()
        lines = [
            title,
            f"Reward: {self.info['episode_reward_raw']:.2f}",
            f"Steps: {self.info['steps']}",
            f"Mine Hits: {self.info['mine_hits']}",
            "Press R to reset",
        ]
        center_x = grid_rect.centerx
        y = grid_rect.y + 260
        for index, line in enumerate(lines):
            font = self.big_font if index == 0 else self.font
            surface = font.render(line, True, self.TEXT)
            self.window.blit(surface, (center_x - surface.get_width() // 2, y))
            y += 40
