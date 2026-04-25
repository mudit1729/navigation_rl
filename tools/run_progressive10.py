"""Progressive local-context navigation curriculum on a fixed 10x10 grid.

Five stages, each adding one nuance:
  L1 Open         — goal-seeking under fog of war
  L2 Mined Open   — mine avoidance with low wall clutter
  L3 Maze         — corridor navigation, dead ends
  L4 Mined Maze   — both at once
  L5 Generalist   — mix of L1..L4, plus GRPO polish

Each stage:
  * IL warm-starts from previous stage's PPO best (None for L1)
  * PPO warm-starts from this stage's IL best
  * Renders IL + PPO rollout GIFs
  * Per-profile eval against L1..L4 to detect forgetting

Final stage adds GRPO bootstrapped from L5 PPO best.
"""
from __future__ import annotations

import json
import sys
import traceback
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from minefield_rl.eval.evaluate import evaluate_agent
from minefield_rl.eval.render_rollout import render_rollout
from minefield_rl.training.train_grpo import train_grpo
from minefield_rl.training.train_imitation import train_imitation
from minefield_rl.training.train_ppo import train_ppo
from minefield_rl.utils import deep_update, ensure_dir, set_global_seeds


PROFILES = {
    "L1_open":       {"name": "L1_open",       "wall_density": 0.10, "mine_density": 0.05, "max_steps": 120, "generation_max_attempts": 32,  "revisit_window": 8},
    "L2_minedopen":  {"name": "L2_minedopen",  "wall_density": 0.10, "mine_density": 0.20, "max_steps": 150, "generation_max_attempts": 64,  "revisit_window": 8},
    "L3_maze":       {"name": "L3_maze",       "wall_density": 0.35, "mine_density": 0.05, "max_steps": 200, "generation_max_attempts": 256, "revisit_window": 8},
    "L4_minedmaze":  {"name": "L4_minedmaze",  "wall_density": 0.30, "mine_density": 0.20, "max_steps": 200, "generation_max_attempts": 256, "revisit_window": 8},
}

STAGE_ORDER = ["L1_open", "L2_minedopen", "L3_maze", "L4_minedmaze"]

STAGE_BUDGETS = {
    "L1_open":      {"demos": 600,  "ppo_steps": 50_000},
    "L2_minedopen": {"demos": 800,  "ppo_steps": 80_000},
    "L3_maze":      {"demos": 1000, "ppo_steps": 120_000},
    "L4_minedmaze": {"demos": 1200, "ppo_steps": 150_000},
}

L5_BUDGET = {"demos": 1500, "ppo_steps": 100_000, "grpo_updates": 24}

EVAL_EPISODES = 50


def _stage_env_overrides(profile: dict) -> dict:
    return {
        "size": 10,
        "view_radius": 4,
        "wall_density": profile["wall_density"],
        "mine_density": profile["mine_density"],
        "max_steps": profile["max_steps"],
        "generation_max_attempts": profile["generation_max_attempts"],
        "revisit_window": profile["revisit_window"],
        "scenario_name": f"progressive10_{profile['name']}",
        "scenario_label": f"10x10 {profile['name']}",
        "episode_profiles": [profile],
    }


def _build_stage_config(base_config: dict, profile: dict, demos: int, ppo_steps: int) -> dict:
    cfg = deepcopy(base_config)
    cfg = deep_update(cfg, {
        "env": _stage_env_overrides(profile),
        "imitation": {"demo_episodes": demos, "epochs": 8},
        "ppo": {"total_timesteps": ppo_steps, "checkpoint_interval_updates": 25},
    })
    return cfg


def _build_mixed_config(base_config: dict, demos: int, ppo_steps: int, grpo_updates: int) -> dict:
    cfg = deepcopy(base_config)
    profiles = [PROFILES[name] for name in STAGE_ORDER]
    cfg = deep_update(cfg, {
        "env": {
            "size": 10,
            "view_radius": 4,
            "max_steps": 200,
            "generation_max_attempts": 256,
            "revisit_window": 8,
            "scenario_name": "progressive10_L5_mixed",
            "scenario_label": "10x10 L5 Mixed Generalist",
            "episode_profiles": profiles,
        },
        "imitation": {"demo_episodes": demos, "epochs": 8},
        "ppo": {"total_timesteps": ppo_steps, "checkpoint_interval_updates": 25},
        "grpo": {"total_updates": grpo_updates},
    })
    return cfg


def _per_profile_eval(stage_root: Path, base_config: dict, ckpt_path: Path) -> dict:
    out = {}
    for name in STAGE_ORDER:
        profile = PROFILES[name]
        eval_cfg = deepcopy(base_config)
        eval_cfg = deep_update(eval_cfg, {"env": _stage_env_overrides(profile)})
        log_dir = ensure_dir(stage_root / "per_profile_eval" / name)
        out[name] = evaluate_agent(
            config=eval_cfg,
            checkpoint_path=str(ckpt_path),
            agent="ppo",
            device="cpu",
            episodes=EVAL_EPISODES,
            log_dir=log_dir,
        )
    return out


def _render_pair(stage_cfg: dict, stage_root: Path, il_best: Path, ppo_best: Path) -> dict:
    out = {}
    for tag, ckpt in [("il", il_best), ("ppo", ppo_best)]:
        if not ckpt.exists():
            out[tag] = {"missing": str(ckpt)}
            continue
        try:
            r = render_rollout(
                config=stage_cfg,
                checkpoint_path=str(ckpt),
                output_prefix=stage_root / f"render_{tag}_best",
                agent="ppo",
                device="cpu",
                max_attempts=4,
                sample_every=4,
            )
            out[tag] = {"outcome": r.get("outcome"), "steps": r.get("steps")}
        except Exception as exc:
            out[tag] = {"error": repr(exc)}
    return out


def main() -> None:
    config_path = REPO_ROOT / "minefield_rl" / "configs" / "config.yaml"
    with config_path.open() as fh:
        base_config = yaml.safe_load(fh)
    set_global_seeds(int(base_config["env"]["seed"]))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = ensure_dir(REPO_ROOT / "minefield_rl" / "logs" / f"progressive10_run_{timestamp}")
    summary_path = root / "summary.json"
    results: dict = {}

    def write_summary(stage: str | None, status: str, error: str | None = None) -> None:
        payload = {
            "root_dir": str(root.resolve()),
            "status": status,
            "current_stage": stage,
            "stages": STAGE_ORDER + ["L5_mixed"],
            "results": results,
        }
        if error:
            payload["error"] = error
        summary_path.write_text(json.dumps(payload, indent=2, default=str))

    write_summary(None, "starting")

    try:
        prev_ppo_best: Path | None = None

        for stage_name in STAGE_ORDER:
            profile = PROFILES[stage_name]
            budget = STAGE_BUDGETS[stage_name]
            stage_cfg = _build_stage_config(base_config, profile, budget["demos"], budget["ppo_steps"])

            stage_root = ensure_dir(root / stage_name)
            stage_results: dict = {}
            results[stage_name] = stage_results

            # IL
            il_stage = f"{stage_name}_imitation"
            write_summary(il_stage, "running")
            il_dir = ensure_dir(stage_root / "imitation")
            stage_results["imitation"] = train_imitation(
                config=stage_cfg,
                device="cpu",
                checkpoint_dir=il_dir / "checkpoints",
                log_dir=il_dir / "logs",
                initial_checkpoint=str(prev_ppo_best) if prev_ppo_best else None,
            )
            il_best = il_dir / "checkpoints" / "ppo_il_best.pt"
            il_ckpt = il_best if il_best.exists() else Path(stage_results["imitation"]["checkpoint"])
            write_summary(il_stage, "running")

            # PPO
            ppo_stage = f"{stage_name}_ppo"
            write_summary(ppo_stage, "running")
            ppo_dir = ensure_dir(stage_root / "ppo")
            stage_results["ppo"] = train_ppo(
                config=stage_cfg,
                device="cpu",
                checkpoint_dir=ppo_dir / "checkpoints",
                log_dir=ppo_dir / "logs",
                initial_checkpoint=il_ckpt,
            )
            ppo_best = ppo_dir / "checkpoints" / "ppo_best.pt"
            ppo_ckpt = ppo_best if ppo_best.exists() else Path(stage_results["ppo"]["checkpoint"])
            write_summary(ppo_stage, "running")

            # Render rollouts for this stage
            stage_results["render"] = _render_pair(stage_cfg, stage_root, il_ckpt, ppo_ckpt)
            write_summary(ppo_stage, "running")

            # Per-profile eval (forgetting check)
            stage_results["per_profile_eval"] = _per_profile_eval(stage_root, base_config, ppo_ckpt)
            write_summary(ppo_stage, "running")

            prev_ppo_best = ppo_ckpt

        # ---- L5 mixed generalist + GRPO ----
        stage_name = "L5_mixed"
        stage_root = ensure_dir(root / stage_name)
        stage_results = {}
        results[stage_name] = stage_results
        l5_cfg = _build_mixed_config(base_config, L5_BUDGET["demos"], L5_BUDGET["ppo_steps"], L5_BUDGET["grpo_updates"])

        # IL on the mix, warm-starting from L4 PPO
        write_summary(f"{stage_name}_imitation", "running")
        il_dir = ensure_dir(stage_root / "imitation")
        stage_results["imitation"] = train_imitation(
            config=l5_cfg,
            device="cpu",
            checkpoint_dir=il_dir / "checkpoints",
            log_dir=il_dir / "logs",
            initial_checkpoint=str(prev_ppo_best) if prev_ppo_best else None,
        )
        il_best = il_dir / "checkpoints" / "ppo_il_best.pt"
        il_ckpt = il_best if il_best.exists() else Path(stage_results["imitation"]["checkpoint"])
        write_summary(f"{stage_name}_imitation", "running")

        # PPO on the mix
        write_summary(f"{stage_name}_ppo", "running")
        ppo_dir = ensure_dir(stage_root / "ppo")
        stage_results["ppo"] = train_ppo(
            config=l5_cfg,
            device="cpu",
            checkpoint_dir=ppo_dir / "checkpoints",
            log_dir=ppo_dir / "logs",
            initial_checkpoint=il_ckpt,
        )
        ppo_best = ppo_dir / "checkpoints" / "ppo_best.pt"
        ppo_ckpt = ppo_best if ppo_best.exists() else Path(stage_results["ppo"]["checkpoint"])
        write_summary(f"{stage_name}_ppo", "running")

        # GRPO on the mix
        write_summary(f"{stage_name}_grpo", "running")
        grpo_dir = ensure_dir(stage_root / "grpo")
        stage_results["grpo"] = train_grpo(
            config=l5_cfg,
            device="cpu",
            checkpoint_dir=grpo_dir / "checkpoints",
            log_dir=grpo_dir / "logs",
            initial_checkpoint=ppo_ckpt,
        )
        grpo_best = grpo_dir / "checkpoints" / "ppo_grpo_best.pt"
        grpo_ckpt = grpo_best if grpo_best.exists() else Path(stage_results["grpo"]["checkpoint"])
        write_summary(f"{stage_name}_grpo", "running")

        # Render IL/PPO/GRPO for the generalist stage
        render_out = _render_pair(l5_cfg, stage_root, il_ckpt, ppo_ckpt)
        try:
            r = render_rollout(
                config=l5_cfg,
                checkpoint_path=str(grpo_ckpt),
                output_prefix=stage_root / "render_grpo_best",
                agent="ppo",
                device="cpu",
                max_attempts=4,
                sample_every=4,
            )
            render_out["grpo"] = {"outcome": r.get("outcome"), "steps": r.get("steps")}
        except Exception as exc:
            render_out["grpo"] = {"error": repr(exc)}
        stage_results["render"] = render_out
        write_summary(f"{stage_name}_grpo", "running")

        # Final per-profile eval on the GRPO best
        stage_results["per_profile_eval"] = _per_profile_eval(stage_root, base_config, grpo_ckpt)
        write_summary(None, "running")

    except Exception as exc:
        write_summary(None, "failed", error=repr(exc))
        traceback.print_exc()
        raise

    write_summary(None, "completed")
    print(json.dumps({"root_dir": str(root.resolve()), "results": results}, indent=2, default=str))


if __name__ == "__main__":
    main()
