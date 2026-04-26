"""Re-run only the GRPO-affected stages with the corrected MCTS / PPO+MCTS code.

Two recently-landed fixes change behavior on any code path that uses MCTS:
  - models/mcts.py: SearchResult.behavior_policy is the actual sampling
    distribution; PPO old_log_probs use it now (previously the actor logits
    were used, making the off-policy ratio mathematically inconsistent).
  - env/minefield_env.py: copy() / restore() / snapshot now carry dispersion
    and blocked_fraction_target so MCTS-forked envs match the active profile.

GRPO is the only stage in this codebase that exercises MCTS, so re-running
GRPO (warm-started from the same PPO best as before) gives a clean head-to-
head with the previously-shipped numbers.

Stages:
  1. L9 generalist (10x10 clustered)         — extension run
  2. L13 dispersed mixed (10x10)             — dispersed curriculum
  3. L13 dispersed mixed (20x20)             — 20x20 dispersed extension
  4. Mine-is-death fine-tune (20x20, hp=1)   — headline stage

For each, we:
  - load the PPO best checkpoint, recover its training config
  - call train_grpo() with the same config, warm-started from that ckpt
  - run per-profile eval (50 ep) on the trained GRPO best for dispersed
    stages and the mine-death stage
  - record old vs new GRPO best_success_rate_100 in summary.json
"""
from __future__ import annotations

import csv
import json
import sys
import traceback
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from minefield_rl.eval.evaluate import evaluate_agent
from minefield_rl.training.train_grpo import train_grpo
from minefield_rl.utils import deep_update, ensure_dir, set_global_seeds


STAGES: list[dict] = [
    {
        "name": "L9_clustered_10x10",
        "ppo_best": REPO_ROOT / "minefield_rl/logs/progressive10_ext_run_20260425_130124/L9_extmixed/ppo/checkpoints/ppo_best.pt",
        "eval_profiles": [],   # generalist, evaluated by GRPO own log only
    },
    {
        "name": "L13_dispersed_10x10",
        "ppo_best": REPO_ROOT / "minefield_rl/logs/progressive10_disp_run_20260425_133443/L13_dispmixed/ppo/checkpoints/ppo_best.pt",
        "eval_profiles": [
            {"name": "L10_dispwalls",   "wall_density": 0.40, "mine_density": 0.20, "max_steps": 200,  "generation_max_attempts": 256, "revisit_window": 12, "dispersion": "dispersed"},
            {"name": "L11_dispmines",   "wall_density": 0.30, "mine_density": 0.30, "max_steps": 200,  "generation_max_attempts": 256, "revisit_window": 12, "dispersion": "dispersed"},
            {"name": "L12_dispextreme", "wall_density": 0.40, "mine_density": 0.30, "max_steps": 240,  "generation_max_attempts": 512, "revisit_window": 12, "dispersion": "dispersed"},
            {"name": "L13_dispopen",    "wall_density": 0.20, "mine_density": 0.20, "max_steps": 160,  "generation_max_attempts": 128, "revisit_window": 12, "dispersion": "dispersed"},
        ],
        "eval_grid": 10, "eval_max_health": 3,
    },
    {
        "name": "L13_dispersed_20x20",
        "ppo_best": REPO_ROOT / "minefield_rl/logs/progressive20_disp_run_20260425_154449/L13_dispmixed/ppo/checkpoints/ppo_best.pt",
        "eval_profiles": [
            {"name": "L10_dispwalls",   "wall_density": 0.40, "mine_density": 0.20, "max_steps": 800,  "generation_max_attempts": 1024, "revisit_window": 12, "dispersion": "dispersed"},
            {"name": "L11_dispmines",   "wall_density": 0.30, "mine_density": 0.30, "max_steps": 800,  "generation_max_attempts": 1024, "revisit_window": 12, "dispersion": "dispersed"},
            {"name": "L12_dispextreme", "wall_density": 0.40, "mine_density": 0.30, "max_steps": 1000, "generation_max_attempts": 2048, "revisit_window": 12, "dispersion": "dispersed"},
            {"name": "L13_dispopen",    "wall_density": 0.20, "mine_density": 0.20, "max_steps": 600,  "generation_max_attempts": 256,  "revisit_window": 12, "dispersion": "dispersed"},
        ],
        "eval_grid": 20, "eval_max_health": 3,
    },
    {
        "name": "minedeath_fine_tune",
        "ppo_best": REPO_ROOT / "minefield_rl/logs/finetune_minedeath_run_20260425_180820/ppo/checkpoints/ppo_best.pt",
        "eval_profiles": [
            {"name": "L10_dispwalls",   "wall_density": 0.40, "mine_density": 0.20, "max_steps": 800,  "generation_max_attempts": 1024, "revisit_window": 12, "dispersion": "dispersed"},
            {"name": "L11_dispmines",   "wall_density": 0.30, "mine_density": 0.30, "max_steps": 800,  "generation_max_attempts": 1024, "revisit_window": 12, "dispersion": "dispersed"},
            {"name": "L12_dispextreme", "wall_density": 0.40, "mine_density": 0.30, "max_steps": 1000, "generation_max_attempts": 2048, "revisit_window": 12, "dispersion": "dispersed"},
            {"name": "L13_dispopen",    "wall_density": 0.20, "mine_density": 0.20, "max_steps": 600,  "generation_max_attempts": 256,  "revisit_window": 12, "dispersion": "dispersed"},
        ],
        "eval_grid": 20, "eval_max_health": 1,
    },
]

EVAL_EPISODES = 50


def _config_from_checkpoint(ckpt_path: Path) -> dict:
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = blob.get("config")
    if not isinstance(cfg, dict):
        raise RuntimeError(f"checkpoint {ckpt_path} has no 'config' dict")
    return cfg


def _old_grpo_best(ppo_best_path: Path) -> dict:
    """Read the previously-shipped GRPO log next to the PPO best, return
    {best_success_rate_100, last_success_rate_100, num_updates}."""
    grpo_csv = ppo_best_path.parent.parent.parent / "grpo" / "logs" / "grpo_train.csv"
    if not grpo_csv.exists():
        return {"missing": str(grpo_csv)}
    rows = list(csv.DictReader(grpo_csv.open()))
    if not rows:
        return {"empty": True}
    sr = [float(r["success_rate_100"]) for r in rows if r.get("success_rate_100")]
    return {
        "num_updates": len(rows),
        "last_success_rate_100": sr[-1] if sr else None,
        "best_success_rate_100": max(sr) if sr else None,
    }


def _eval_per_profile(base_config: dict, ckpt_path: Path, log_root: Path,
                      profiles: list[dict], grid: int, max_health: int) -> dict:
    out = {}
    for profile in profiles:
        cfg = deepcopy(base_config)
        cfg = deep_update(cfg, {
            "env": {
                "size": grid,
                "view_radius": 4,
                "max_health": max_health,
                "wall_density": profile["wall_density"],
                "mine_density": profile["mine_density"],
                "max_steps": profile["max_steps"],
                "generation_max_attempts": profile["generation_max_attempts"],
                "revisit_window": profile["revisit_window"],
                "dispersion": profile["dispersion"],
                "scenario_name": f"grpo_rerun_eval_{profile['name']}",
                "scenario_label": f"{grid}x{grid} {profile['name']} hp={max_health}",
                "episode_profiles": [profile],
            },
        })
        log_dir = ensure_dir(log_root / profile["name"])
        res = evaluate_agent(
            config=cfg,
            checkpoint_path=str(ckpt_path),
            agent="ppo",
            device="cpu",
            episodes=EVAL_EPISODES,
            log_dir=log_dir,
        )
        out[profile["name"]] = {
            "success_rate": res.get("success_rate"),
            "death_rate":   res.get("death_rate"),
            "timeout_rate": res.get("timeout_rate"),
            "avg_length":   res.get("avg_length"),
            "avg_mine_hits": res.get("avg_mine_hits"),
            "avg_health_remaining": res.get("avg_health_remaining"),
        }
    return out


def main() -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = ensure_dir(REPO_ROOT / "minefield_rl/logs" / f"grpo_rerun_{timestamp}")
    summary_path = root / "summary.json"
    summary: dict = {
        "root_dir": str(root.resolve()),
        "timestamp": timestamp,
        "stages": [s["name"] for s in STAGES],
        "results": {},
    }

    def write_summary() -> None:
        summary_path.write_text(json.dumps(summary, indent=2, default=str))

    write_summary()

    for stage in STAGES:
        name = stage["name"]
        ppo_best: Path = stage["ppo_best"]
        if not ppo_best.exists():
            summary["results"][name] = {"error": f"missing ppo_best: {ppo_best}"}
            write_summary()
            continue

        try:
            cfg = _config_from_checkpoint(ppo_best)
            set_global_seeds(int(cfg.get("env", {}).get("seed", 0)))

            # Set up per-stage log dir
            stage_root = ensure_dir(root / name)
            ckpt_dir = ensure_dir(stage_root / "grpo" / "checkpoints")
            log_dir = ensure_dir(stage_root / "grpo" / "logs")

            old_grpo = _old_grpo_best(ppo_best)
            print(f"\n=== {name} ===")
            print(f"  warm-start: {ppo_best}")
            print(f"  old GRPO  : {old_grpo}")

            grpo_result = train_grpo(
                config=cfg,
                device="cpu",
                checkpoint_dir=ckpt_dir,
                log_dir=log_dir,
                initial_checkpoint=ppo_best,
            )

            grpo_best = ckpt_dir / "ppo_grpo_best.pt"
            grpo_ckpt = grpo_best if grpo_best.exists() else Path(grpo_result["checkpoint"])

            stage_summary = {
                "warm_start": str(ppo_best),
                "old_grpo": old_grpo,
                "new_grpo": {
                    "checkpoint": str(grpo_ckpt),
                    "best_success_rate_100": grpo_result.get("best_success_rate_100"),
                    "updates": grpo_result.get("updates"),
                },
            }

            if stage["eval_profiles"]:
                eval_dir = ensure_dir(stage_root / "per_profile_eval_grpo")
                stage_summary["per_profile_eval_grpo_new"] = _eval_per_profile(
                    cfg, grpo_ckpt, eval_dir,
                    stage["eval_profiles"], stage["eval_grid"], stage["eval_max_health"],
                )

            summary["results"][name] = stage_summary
            print(f"  new GRPO  : {stage_summary['new_grpo']}")
            write_summary()

        except Exception as exc:
            traceback.print_exc()
            summary["results"][name] = {"error": repr(exc)}
            write_summary()

    summary["status"] = "completed"
    write_summary()
    print(f"\nSummary at: {summary_path}")


if __name__ == "__main__":
    main()
