"""Evaluate the best curriculum checkpoints at larger grid sizes.

The model is fully convolutional over the 9x9 local view (view_radius=4),
so the same weights run unchanged on any grid size. This script evaluates
three checkpoints (L9 GRPO clustered, L12 PPO dispersed, L13 GRPO dispersed
mixed) across grids 10/20/30/50 on two profile families (clustered minedmaze
0.30w/0.20m and dispersed extreme 0.40w/0.30m).
"""
from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from minefield_rl.eval.evaluate import evaluate_agent
from minefield_rl.utils import deep_update, ensure_dir


CHECKPOINTS = {
    "L9_grpo_clustered":  REPO_ROOT / "minefield_rl/logs/progressive10_ext_run_20260425_130124/L9_extmixed/grpo/checkpoints/ppo_grpo_best.pt",
    "L12_ppo_dispersed":  REPO_ROOT / "minefield_rl/logs/progressive10_disp_run_20260425_133443/L12_dispextreme/ppo/checkpoints/ppo_best.pt",
    "L13_grpo_dispmixed": REPO_ROOT / "minefield_rl/logs/progressive10_disp_run_20260425_133443/L13_dispmixed/grpo/checkpoints/ppo_grpo_best.pt",
}

PROFILES = {
    "clustered_minedmaze": {"wall_density": 0.30, "mine_density": 0.20, "dispersion": "clustered"},
    "dispersed_extreme":   {"wall_density": 0.40, "mine_density": 0.30, "dispersion": "dispersed"},
}

SIZES = [10, 20, 30]
EPISODES = 50


def main() -> None:
    with (REPO_ROOT / "minefield_rl/configs/config.yaml").open() as fh:
        base = yaml.safe_load(fh)

    out_root = ensure_dir(REPO_ROOT / "minefield_rl/logs/scaleup_eval")
    summary: dict = {}

    for ckpt_name, ckpt_path in CHECKPOINTS.items():
        if not ckpt_path.exists():
            print(f"SKIP {ckpt_name}: checkpoint missing at {ckpt_path}")
            continue
        summary[ckpt_name] = {}
        for prof_name, prof in PROFILES.items():
            summary[ckpt_name][prof_name] = {}
            for size in SIZES:
                profile = {
                    "name": f"{prof_name}_n{size}",
                    "wall_density": prof["wall_density"],
                    "mine_density": prof["mine_density"],
                    "dispersion": prof["dispersion"],
                    "max_steps": int(5 * size * size),
                    "generation_max_attempts": 1024,
                    "revisit_window": 8,
                }
                cfg = deepcopy(base)
                cfg = deep_update(cfg, {
                    "env": {
                        "size": size,
                        "view_radius": 4,
                        "wall_density": prof["wall_density"],
                        "mine_density": prof["mine_density"],
                        "dispersion": prof["dispersion"],
                        "max_steps": profile["max_steps"],
                        "generation_max_attempts": 1024,
                        "revisit_window": 8,
                        "scenario_name": f"scaleup_{prof_name}_{size}",
                        "scenario_label": f"{prof_name} {size}x{size}",
                        "episode_profiles": [profile],
                    },
                })
                log_dir = ensure_dir(out_root / ckpt_name / prof_name / f"size_{size}")
                res = evaluate_agent(
                    config=cfg,
                    checkpoint_path=str(ckpt_path),
                    agent="ppo",
                    device="cpu",
                    episodes=EPISODES,
                    log_dir=log_dir,
                )
                summary[ckpt_name][prof_name][f"size_{size}"] = {
                    "success_rate": res.get("success_rate"),
                    "death_rate": res.get("death_rate"),
                    "timeout_rate": res.get("timeout_rate"),
                    "avg_length": res.get("avg_length"),
                    "avg_mine_hits": res.get("avg_mine_hits"),
                }
                sr = res.get("success_rate", 0.0) or 0.0
                dr = res.get("death_rate", 0.0) or 0.0
                tr = res.get("timeout_rate", 0.0) or 0.0
                avg = res.get("avg_length", 0.0) or 0.0
                print(f"{ckpt_name:24s} | {prof_name:22s} | n={size:3d} | succ={sr:.2f} death={dr:.2f} timeout={tr:.2f} avg_len={avg:.1f}")

    (out_root / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nWrote summary to {out_root / 'summary.json'}")


if __name__ == "__main__":
    main()
