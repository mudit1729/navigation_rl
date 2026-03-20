from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from minefield_rl.eval.evaluate import evaluate_agent
from minefield_rl.utils import ensure_dir


def compare_agents(
    config: dict[str, Any],
    dqn_checkpoint: str | Path | None,
    mcts_checkpoint: str | Path | None,
    device: str = "cpu",
    episodes: int | None = None,
    map_size: int | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    output_root = ensure_dir(output_dir or "minefield_rl/logs")
    dqn_summary = evaluate_agent(
        config=config,
        checkpoint_path=dqn_checkpoint,
        agent="dqn",
        device=device,
        episodes=episodes,
        map_size=map_size,
        log_dir=output_root,
    )
    mcts_summary = evaluate_agent(
        config=config,
        checkpoint_path=mcts_checkpoint,
        agent="mcts",
        device=device,
        episodes=episodes,
        map_size=map_size,
        log_dir=output_root,
    )

    metrics = [
        ("Success Rate", dqn_summary["success_rate"], mcts_summary["success_rate"]),
        ("Avg Reward", dqn_summary["avg_reward_raw"], mcts_summary["avg_reward_raw"]),
        ("Avg Length", dqn_summary["avg_length"], mcts_summary["avg_length"]),
        ("Avg Mine Hits", dqn_summary["avg_mine_hits"], mcts_summary["avg_mine_hits"]),
    ]

    figure, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 4))
    if len(metrics) == 1:
        axes = [axes]
    for axis, (title, dqn_value, mcts_value) in zip(axes, metrics):
        axis.bar(["DQN", "DQN+MCTS"], [dqn_value, mcts_value], color=["#5C80BC", "#E76F51"])
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()

    output_path = Path(output_root) / "dqn_vs_mcts_comparison.png"
    figure.savefig(output_path, dpi=180)
    plt.close(figure)

    return {
        "dqn": dqn_summary,
        "mcts": mcts_summary,
        "figure": str(output_path),
    }
