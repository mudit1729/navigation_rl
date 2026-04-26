"""Fine-tune the 20x20 dispersed L13 PPO best with mine = instant death.

Sets max_health=1 so any mine hit terminates the episode. The expert planner
already refuses to step on mines when health <= 1, so demos are strictly
mine-free paths. Pipeline:

  IL fine-tune  (warm from L13 PPO best, 20x20 dispersed mix, max_health=1)
  PPO fine-tune (warm from IL best)
  GRPO         (warm from PPO best)
  per-profile eval against L10/L11/L12/L13_dispopen at max_health=1
  render PPO best + GRPO best at 1 fps
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


PRIOR_RUN_ROOT = REPO_ROOT / "minefield_rl" / "logs" / "progressive20_disp_run_20260425_154449"
PRIOR_PPO_BEST = PRIOR_RUN_ROOT / "L13_dispmixed" / "ppo" / "checkpoints" / "ppo_best.pt"

GRID_SIZE = 20
MAX_HEALTH = 1  # mine = instant death

PROFILES = {
    "L10_dispwalls":   {"name": "L10_dispwalls",   "wall_density": 0.40, "mine_density": 0.20, "max_steps": 800,  "generation_max_attempts": 1024, "revisit_window": 12, "dispersion": "dispersed"},
    "L11_dispmines":   {"name": "L11_dispmines",   "wall_density": 0.30, "mine_density": 0.30, "max_steps": 800,  "generation_max_attempts": 1024, "revisit_window": 12, "dispersion": "dispersed"},
    "L12_dispextreme": {"name": "L12_dispextreme", "wall_density": 0.40, "mine_density": 0.30, "max_steps": 1000, "generation_max_attempts": 2048, "revisit_window": 12, "dispersion": "dispersed"},
    "L13_dispopen":    {"name": "L13_dispopen",    "wall_density": 0.20, "mine_density": 0.20, "max_steps": 600,  "generation_max_attempts": 256,  "revisit_window": 12, "dispersion": "dispersed"},
}

EVAL_PROFILES = ["L10_dispwalls", "L11_dispmines", "L12_dispextreme", "L13_dispopen"]
MIX_PROFILES = ["L13_dispopen", "L10_dispwalls", "L11_dispmines", "L12_dispextreme"]

DEMOS = 1500
PPO_STEPS = 250_000
GRPO_UPDATES = 24
EVAL_EPISODES = 50


def _stage_env_overrides(profile: dict, profiles_list: list[dict] | None = None) -> dict:
    return {
        "size": GRID_SIZE,
        "view_radius": 4,
        "max_health": MAX_HEALTH,
        "wall_density": profile["wall_density"],
        "mine_density": profile["mine_density"],
        "max_steps": profile["max_steps"],
        "generation_max_attempts": profile["generation_max_attempts"],
        "revisit_window": profile["revisit_window"],
        "dispersion": profile["dispersion"],
        "scenario_name": f"finetune_minedeath_{profile['name']}",
        "scenario_label": f"{GRID_SIZE}x{GRID_SIZE} mine=death {profile['name']}",
        "episode_profiles": profiles_list if profiles_list is not None else [profile],
    }


def _build_mix_config(base_config: dict) -> dict:
    cfg = deepcopy(base_config)
    profiles = [PROFILES[name] for name in MIX_PROFILES]
    return deep_update(cfg, {
        "env": {
            "size": GRID_SIZE,
            "view_radius": 4,
            "max_health": MAX_HEALTH,
            "max_steps": 1000,
            "generation_max_attempts": 2048,
            "revisit_window": 12,
            "dispersion": "dispersed",
            "scenario_name": "finetune_minedeath_mix",
            "scenario_label": f"{GRID_SIZE}x{GRID_SIZE} mine=death mixed",
            "episode_profiles": profiles,
        },
        "imitation": {"demo_episodes": DEMOS, "epochs": 8},
        "ppo": {"total_timesteps": PPO_STEPS, "checkpoint_interval_updates": 25},
        "grpo": {"total_updates": GRPO_UPDATES},
    })


def _per_profile_eval(stage_root: Path, base_config: dict, ckpt_path: Path) -> dict:
    out = {}
    for name in EVAL_PROFILES:
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


def _render_one(cfg: dict, ckpt: Path, output_prefix: Path, sample_every: int = 1) -> dict:
    if not ckpt.exists():
        return {"missing": str(ckpt)}
    try:
        r = render_rollout(
            config=cfg,
            checkpoint_path=str(ckpt),
            output_prefix=output_prefix,
            agent="ppo",
            device="cpu",
            max_attempts=8,
            sample_every=sample_every,
            hold_frames=2,
        )
        return {"outcome": r.get("outcome"), "steps": r.get("steps")}
    except Exception as exc:
        return {"error": repr(exc)}


def main() -> None:
    if not PRIOR_PPO_BEST.exists():
        raise FileNotFoundError(f"prior L13 PPO checkpoint missing: {PRIOR_PPO_BEST}")

    config_path = REPO_ROOT / "minefield_rl" / "configs" / "config.yaml"
    with config_path.open() as fh:
        base_config = yaml.safe_load(fh)
    set_global_seeds(int(base_config["env"]["seed"]))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = ensure_dir(REPO_ROOT / "minefield_rl" / "logs" / f"finetune_minedeath_run_{timestamp}")
    summary_path = root / "summary.json"
    results: dict = {
        "prior_run": str(PRIOR_RUN_ROOT),
        "prior_checkpoint": str(PRIOR_PPO_BEST),
        "grid_size": GRID_SIZE,
        "max_health": MAX_HEALTH,
    }

    def write_summary(stage: str | None, status: str, error: str | None = None) -> None:
        payload = {
            "root_dir": str(root.resolve()),
            "status": status,
            "current_stage": stage,
            "stages": ["imitation", "ppo", "grpo"],
            "results": results,
        }
        if error:
            payload["error"] = error
        summary_path.write_text(json.dumps(payload, indent=2, default=str))

    write_summary(None, "starting")

    try:
        cfg = _build_mix_config(base_config)

        write_summary("imitation", "running")
        il_dir = ensure_dir(root / "imitation")
        results["imitation"] = train_imitation(
            config=cfg,
            device="cpu",
            checkpoint_dir=il_dir / "checkpoints",
            log_dir=il_dir / "logs",
            initial_checkpoint=str(PRIOR_PPO_BEST),
        )
        il_best = il_dir / "checkpoints" / "ppo_il_best.pt"
        il_ckpt = il_best if il_best.exists() else Path(results["imitation"]["checkpoint"])
        write_summary("imitation", "running")

        write_summary("ppo", "running")
        ppo_dir = ensure_dir(root / "ppo")
        results["ppo"] = train_ppo(
            config=cfg,
            device="cpu",
            checkpoint_dir=ppo_dir / "checkpoints",
            log_dir=ppo_dir / "logs",
            initial_checkpoint=il_ckpt,
        )
        ppo_best = ppo_dir / "checkpoints" / "ppo_best.pt"
        ppo_ckpt = ppo_best if ppo_best.exists() else Path(results["ppo"]["checkpoint"])
        write_summary("ppo", "running")

        write_summary("grpo", "running")
        grpo_dir = ensure_dir(root / "grpo")
        results["grpo"] = train_grpo(
            config=cfg,
            device="cpu",
            checkpoint_dir=grpo_dir / "checkpoints",
            log_dir=grpo_dir / "logs",
            initial_checkpoint=ppo_ckpt,
        )
        grpo_best = grpo_dir / "checkpoints" / "ppo_grpo_best.pt"
        grpo_ckpt = grpo_best if grpo_best.exists() else Path(results["grpo"]["checkpoint"])
        write_summary("grpo", "running")

        # Render PPO best and GRPO best on the mixed profile
        results["render"] = {
            "ppo": _render_one(cfg, ppo_ckpt, root / "render_ppo_best"),
            "grpo": _render_one(cfg, grpo_ckpt, root / "render_grpo_best"),
        }
        # Also render one rollout per profile for the slow-step montage
        per_profile_renders = {}
        for name in EVAL_PROFILES:
            profile = PROFILES[name]
            stage_cfg = deepcopy(base_config)
            stage_cfg = deep_update(stage_cfg, {"env": _stage_env_overrides(profile)})
            per_profile_renders[name] = _render_one(
                stage_cfg, grpo_ckpt,
                output_prefix=root / "per_profile_renders" / f"render_grpo_{name}",
                sample_every=1,
            )
        results["per_profile_renders"] = per_profile_renders
        write_summary("grpo", "running")

        results["per_profile_eval_grpo"] = _per_profile_eval(root, base_config, grpo_ckpt)
        results["per_profile_eval_ppo"] = _per_profile_eval(root, base_config, ppo_ckpt)
        write_summary(None, "running")

    except Exception as exc:
        write_summary(None, "failed", error=repr(exc))
        traceback.print_exc()
        raise

    write_summary(None, "completed")
    print(json.dumps({"root_dir": str(root.resolve()), "results": results}, indent=2, default=str))


if __name__ == "__main__":
    main()
