"""Dispersed-mode extension to the progressive10 curriculum.

Bootstraps from the L9 GRPO best of progressive10_ext_run_20260425_130124.
All previous stages used ``dispersion=clustered`` (CA-smoothed walls,
flood-fill mines). This extension flips the generator to ``dispersion=dispersed``
(walls and mines sampled uniformly at random, no clustering), which produces
maps with lots of small isolated obstacles instead of corridor walls and
mine fields. The agent has to decide cell-by-cell from its 9x9 view.

Stages:
  L10 Dispersed Walls    — 0.40 walls, 0.20 mines, dispersed
  L11 Dispersed Mines    — 0.30 walls, 0.30 mines, dispersed
  L12 Dispersed Extreme  — 0.40 walls, 0.30 mines, dispersed
  L13 Dispersed Mixed (PPO+GRPO) — uniform mix of L10/L11/L12 + a low-density
       dispersed-open profile to keep the easy-case skill alive

Each stage: IL (warm-start from prev PPO best) -> PPO (warm from IL best) ->
render -> per-profile eval against L1, L4 (clustered baselines), and the new
dispersed profiles. The clustered evals catch catastrophic forgetting when
the policy specializes to dispersed layouts.
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


PRIOR_RUN_ROOT = REPO_ROOT / "minefield_rl" / "logs" / "progressive10_ext_run_20260425_130124"
PRIOR_GRPO_BEST = PRIOR_RUN_ROOT / "L9_extmixed" / "grpo" / "checkpoints" / "ppo_grpo_best.pt"


PROFILES = {
    "L1_open":            {"name": "L1_open",            "wall_density": 0.10, "mine_density": 0.05, "max_steps": 120, "generation_max_attempts": 32,   "revisit_window": 8, "dispersion": "clustered"},
    "L4_minedmaze":       {"name": "L4_minedmaze",       "wall_density": 0.30, "mine_density": 0.20, "max_steps": 200, "generation_max_attempts": 256,  "revisit_window": 8, "dispersion": "clustered"},
    "L10_dispwalls":      {"name": "L10_dispwalls",      "wall_density": 0.40, "mine_density": 0.20, "max_steps": 220, "generation_max_attempts": 256,  "revisit_window": 8, "dispersion": "dispersed"},
    "L11_dispmines":      {"name": "L11_dispmines",      "wall_density": 0.30, "mine_density": 0.30, "max_steps": 220, "generation_max_attempts": 256,  "revisit_window": 8, "dispersion": "dispersed"},
    "L12_dispextreme":    {"name": "L12_dispextreme",    "wall_density": 0.40, "mine_density": 0.30, "max_steps": 240, "generation_max_attempts": 512,  "revisit_window": 8, "dispersion": "dispersed"},
    "L13_dispopen":       {"name": "L13_dispopen",       "wall_density": 0.20, "mine_density": 0.20, "max_steps": 180, "generation_max_attempts": 128,  "revisit_window": 8, "dispersion": "dispersed"},
}

NEW_STAGE_ORDER = ["L10_dispwalls", "L11_dispmines", "L12_dispextreme"]
EVAL_PROFILES = ["L1_open", "L4_minedmaze", "L10_dispwalls", "L11_dispmines", "L12_dispextreme"]
MIX_PROFILES_FOR_L13 = ["L13_dispopen", "L10_dispwalls", "L11_dispmines", "L12_dispextreme"]

STAGE_BUDGETS = {
    "L10_dispwalls":   {"demos": 1500, "ppo_steps": 200_000},
    "L11_dispmines":   {"demos": 1500, "ppo_steps": 200_000},
    "L12_dispextreme": {"demos": 1800, "ppo_steps": 250_000},
}

L13_BUDGET = {"demos": 1800, "ppo_steps": 200_000, "grpo_updates": 30}

EVAL_EPISODES = 50


def _stage_env_overrides(profile: dict, profiles_list: list[dict] | None = None) -> dict:
    return {
        "size": 10,
        "view_radius": 4,
        "wall_density": profile["wall_density"],
        "mine_density": profile["mine_density"],
        "max_steps": profile["max_steps"],
        "generation_max_attempts": profile["generation_max_attempts"],
        "revisit_window": profile["revisit_window"],
        "dispersion": profile["dispersion"],
        "scenario_name": f"progressive10_{profile['name']}",
        "scenario_label": f"10x10 {profile['name']}",
        "episode_profiles": profiles_list if profiles_list is not None else [profile],
    }


def _build_stage_config(base_config: dict, profile: dict, demos: int, ppo_steps: int) -> dict:
    cfg = deepcopy(base_config)
    return deep_update(cfg, {
        "env": _stage_env_overrides(profile),
        "imitation": {"demo_episodes": demos, "epochs": 8},
        "ppo": {"total_timesteps": ppo_steps, "checkpoint_interval_updates": 25},
    })


def _build_l13_config(base_config: dict, demos: int, ppo_steps: int, grpo_updates: int) -> dict:
    cfg = deepcopy(base_config)
    profiles = [PROFILES[name] for name in MIX_PROFILES_FOR_L13]
    return deep_update(cfg, {
        "env": {
            "size": 10,
            "view_radius": 4,
            "max_steps": 240,
            "generation_max_attempts": 512,
            "revisit_window": 8,
            "dispersion": "dispersed",
            "scenario_name": "progressive10_L13_dispmixed",
            "scenario_label": "10x10 L13 Dispersed Mixed",
            "episode_profiles": profiles,
        },
        "imitation": {"demo_episodes": demos, "epochs": 8},
        "ppo": {"total_timesteps": ppo_steps, "checkpoint_interval_updates": 25},
        "grpo": {"total_updates": grpo_updates},
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
    if not PRIOR_GRPO_BEST.exists():
        raise FileNotFoundError(f"prior L9 GRPO checkpoint missing: {PRIOR_GRPO_BEST}")

    config_path = REPO_ROOT / "minefield_rl" / "configs" / "config.yaml"
    with config_path.open() as fh:
        base_config = yaml.safe_load(fh)
    set_global_seeds(int(base_config["env"]["seed"]))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = ensure_dir(REPO_ROOT / "minefield_rl" / "logs" / f"progressive10_disp_run_{timestamp}")
    summary_path = root / "summary.json"
    results: dict = {"prior_run": str(PRIOR_RUN_ROOT)}

    def write_summary(stage: str | None, status: str, error: str | None = None) -> None:
        payload = {
            "root_dir": str(root.resolve()),
            "status": status,
            "current_stage": stage,
            "stages": NEW_STAGE_ORDER + ["L13_dispmixed"],
            "results": results,
        }
        if error:
            payload["error"] = error
        summary_path.write_text(json.dumps(payload, indent=2, default=str))

    write_summary(None, "starting")

    try:
        prev_ppo_best: Path = PRIOR_GRPO_BEST

        for stage_name in NEW_STAGE_ORDER:
            profile = PROFILES[stage_name]
            budget = STAGE_BUDGETS[stage_name]
            stage_cfg = _build_stage_config(base_config, profile, budget["demos"], budget["ppo_steps"])

            stage_root = ensure_dir(root / stage_name)
            stage_results: dict = {}
            results[stage_name] = stage_results

            il_stage = f"{stage_name}_imitation"
            write_summary(il_stage, "running")
            il_dir = ensure_dir(stage_root / "imitation")
            stage_results["imitation"] = train_imitation(
                config=stage_cfg,
                device="cpu",
                checkpoint_dir=il_dir / "checkpoints",
                log_dir=il_dir / "logs",
                initial_checkpoint=str(prev_ppo_best),
            )
            il_best = il_dir / "checkpoints" / "ppo_il_best.pt"
            il_ckpt = il_best if il_best.exists() else Path(stage_results["imitation"]["checkpoint"])
            write_summary(il_stage, "running")

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

            stage_results["render"] = _render_pair(stage_cfg, stage_root, il_ckpt, ppo_ckpt)
            write_summary(ppo_stage, "running")

            stage_results["per_profile_eval"] = _per_profile_eval(stage_root, base_config, ppo_ckpt)
            write_summary(ppo_stage, "running")

            prev_ppo_best = ppo_ckpt

        # ---- L13 dispersed mixed + GRPO ----
        stage_name = "L13_dispmixed"
        stage_root = ensure_dir(root / stage_name)
        stage_results = {}
        results[stage_name] = stage_results
        l13_cfg = _build_l13_config(base_config, L13_BUDGET["demos"], L13_BUDGET["ppo_steps"], L13_BUDGET["grpo_updates"])

        write_summary(f"{stage_name}_imitation", "running")
        il_dir = ensure_dir(stage_root / "imitation")
        stage_results["imitation"] = train_imitation(
            config=l13_cfg,
            device="cpu",
            checkpoint_dir=il_dir / "checkpoints",
            log_dir=il_dir / "logs",
            initial_checkpoint=str(prev_ppo_best),
        )
        il_best = il_dir / "checkpoints" / "ppo_il_best.pt"
        il_ckpt = il_best if il_best.exists() else Path(stage_results["imitation"]["checkpoint"])
        write_summary(f"{stage_name}_imitation", "running")

        write_summary(f"{stage_name}_ppo", "running")
        ppo_dir = ensure_dir(stage_root / "ppo")
        stage_results["ppo"] = train_ppo(
            config=l13_cfg,
            device="cpu",
            checkpoint_dir=ppo_dir / "checkpoints",
            log_dir=ppo_dir / "logs",
            initial_checkpoint=il_ckpt,
        )
        ppo_best = ppo_dir / "checkpoints" / "ppo_best.pt"
        ppo_ckpt = ppo_best if ppo_best.exists() else Path(stage_results["ppo"]["checkpoint"])
        write_summary(f"{stage_name}_ppo", "running")

        write_summary(f"{stage_name}_grpo", "running")
        grpo_dir = ensure_dir(stage_root / "grpo")
        stage_results["grpo"] = train_grpo(
            config=l13_cfg,
            device="cpu",
            checkpoint_dir=grpo_dir / "checkpoints",
            log_dir=grpo_dir / "logs",
            initial_checkpoint=ppo_ckpt,
        )
        grpo_best = grpo_dir / "checkpoints" / "ppo_grpo_best.pt"
        grpo_ckpt = grpo_best if grpo_best.exists() else Path(stage_results["grpo"]["checkpoint"])
        write_summary(f"{stage_name}_grpo", "running")

        render_out = _render_pair(l13_cfg, stage_root, il_ckpt, ppo_ckpt)
        try:
            r = render_rollout(
                config=l13_cfg,
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
