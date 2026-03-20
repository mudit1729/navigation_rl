from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from minefield_rl.eval.render_rollout import render_rollout
from minefield_rl.main import prepare_config
from minefield_rl.training.train_grpo import train_grpo
from minefield_rl.training.train_imitation import train_imitation
from minefield_rl.utils import ensure_dir


def run_curriculum(
    config_path: str | Path = "minefield_rl/configs/config.yaml",
    device: str = "cpu",
    root_dir: str | Path | None = None,
) -> dict[str, Any]:
    base_config = prepare_config(config_path, seed_override=None, scenario_name=None)
    curriculum_cfg = base_config.get("curriculum", {})
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = ensure_dir(root_dir or Path("minefield_rl") / "logs" / f"curriculum_{timestamp}")

    current_checkpoint: str | Path | None = None
    stage_results: dict[str, Any] = {}

    for scenario_name in ("easy", "medium", "hard"):
        stage_cfg = curriculum_cfg.get(scenario_name, {})
        stage_root = ensure_dir(root / scenario_name)
        config = prepare_config(config_path, seed_override=None, scenario_name=scenario_name)

        if bool(stage_cfg.get("run_imitation", False)):
            imitation_root = ensure_dir(stage_root / "imitation")
            imitation_result = train_imitation(
                config=config,
                device=device,
                checkpoint_dir=imitation_root / "checkpoints",
                log_dir=imitation_root / "logs",
            )
            imitation_best = imitation_root / "checkpoints" / "ppo_il_best.pt"
            current_checkpoint = imitation_best if imitation_best.exists() else imitation_result["checkpoint"]
            stage_results[f"{scenario_name}_imitation"] = imitation_result

        grpo_root = ensure_dir(stage_root / "grpo")
        grpo_result = train_grpo(
            config=config,
            device=device,
            checkpoint_dir=grpo_root / "checkpoints",
            log_dir=grpo_root / "logs",
            initial_checkpoint=current_checkpoint,
            total_updates=int(stage_cfg.get("grpo_updates", base_config["grpo"]["total_updates"])),
            mcts_simulations=int(stage_cfg.get("mcts_simulations", base_config["grpo"]["mcts_simulations"])),
            max_episode_steps=int(stage_cfg["max_episode_steps"]) if stage_cfg.get("max_episode_steps") is not None else None,
        )
        current_checkpoint = grpo_result["best_checkpoint"]
        stage_results[f"{scenario_name}_grpo"] = grpo_result

    render_results: dict[str, Any] = {}
    for scenario_name in ("easy", "medium", "hard"):
        config = prepare_config(config_path, seed_override=None, scenario_name=scenario_name)
        stage_cfg = curriculum_cfg.get(scenario_name, {})
        render_results[scenario_name] = render_rollout(
            config=config,
            checkpoint_path=current_checkpoint,
            output_prefix=root / f"{scenario_name}_final_play",
            agent="ppo",
            device=device,
            max_attempts=int(stage_cfg.get("render_attempts", 3)),
            sample_every=int(stage_cfg.get("render_sample_every", 8)),
        )

    return {
        "root_dir": str(root),
        "final_checkpoint": str(current_checkpoint),
        "stages": stage_results,
        "renders": render_results,
    }
