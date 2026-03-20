from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Minefield Navigator RL")
    parser.add_argument("--mode", choices=["train", "eval", "play", "compare", "render", "curriculum"], required=True)
    parser.add_argument("--agent", choices=["human", "dqn", "ppo", "mcts"], default="dqn")
    parser.add_argument("--imitation", action="store_true")
    parser.add_argument("--grpo", action="store_true")
    parser.add_argument("--scenario", default=None)
    parser.add_argument("--config", default="minefield_rl/configs/config.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--mcts-checkpoint", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--map_size", type=int, default=None)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-prefix", default=None)
    return parser


def prepare_config(
    config_path: str | Path,
    seed_override: int | None,
    scenario_name: str | None = None,
) -> dict[str, Any]:
    from minefield_rl.utils import deep_update, load_config, set_global_seeds

    config = load_config(config_path)
    if scenario_name is not None:
        scenarios = config.get("scenarios", {})
        if scenario_name not in scenarios:
            available = ", ".join(sorted(scenarios)) or "<none>"
            raise ValueError(f"Unknown scenario '{scenario_name}'. Available scenarios: {available}")
        config = deep_update(config, {"env": scenarios[scenario_name]})
    if seed_override is not None:
        config = deep_update(config, {"env": {"seed": seed_override}})
    set_global_seeds(int(config["env"]["seed"]))
    return config


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config = prepare_config(args.config, args.seed, args.scenario)

    if args.mode == "train":
        from minefield_rl.training.train_grpo import train_grpo
        from minefield_rl.training.train_drqn import train_drqn
        from minefield_rl.training.train_imitation import train_imitation
        from minefield_rl.training.train_mcts import train_mcts
        from minefield_rl.training.train_ppo import train_ppo

        if args.agent == "dqn":
            result = train_drqn(
                config=config,
                device=args.device,
                map_size=args.map_size,
            )
        elif args.agent == "ppo":
            if args.imitation:
                result = train_imitation(
                    config=config,
                    device=args.device,
                    map_size=args.map_size,
                )
            elif args.grpo:
                result = train_grpo(
                    config=config,
                    device=args.device,
                    map_size=args.map_size,
                    initial_checkpoint=args.checkpoint,
                )
            else:
                result = train_ppo(
                    config=config,
                    device=args.device,
                    map_size=args.map_size,
                    initial_checkpoint=args.checkpoint,
                )
        elif args.agent == "mcts":
            result = train_mcts(
                config=config,
                device=args.device,
                map_size=args.map_size,
                initial_checkpoint=args.checkpoint,
                episodes=args.episodes,
            )
        else:
            raise ValueError("Training supports only --agent dqn, --agent ppo, or --agent mcts")
        print(json.dumps(result, indent=2))
        return

    if args.mode == "eval":
        from minefield_rl.eval.evaluate import evaluate_agent

        if args.agent not in {"dqn", "ppo", "mcts"}:
            raise ValueError("Evaluation supports only --agent dqn, --agent ppo, or --agent mcts")
        result = evaluate_agent(
            config=config,
            checkpoint_path=args.checkpoint,
            agent=args.agent,
            device=args.device,
            episodes=args.episodes,
            map_size=args.map_size,
        )
        print(json.dumps(result, indent=2))
        return

    if args.mode == "compare":
        from minefield_rl.eval.compare import compare_agents

        result = compare_agents(
            config=config,
            dqn_checkpoint=args.checkpoint,
            mcts_checkpoint=args.mcts_checkpoint or args.checkpoint,
            device=args.device,
            episodes=args.episodes,
            map_size=args.map_size,
        )
        print(json.dumps(result, indent=2))
        return

    if args.mode == "render":
        from minefield_rl.eval.render_rollout import render_rollout

        if args.checkpoint is None:
            raise ValueError("--checkpoint is required for render mode")
        prefix = args.output_prefix or "minefield_rl/logs/rendered_rollout"
        result = render_rollout(
            config=config,
            checkpoint_path=args.checkpoint,
            output_prefix=prefix,
            agent=args.agent,
            device=args.device,
        )
        print(json.dumps(result, indent=2))
        return

    if args.mode == "curriculum":
        from minefield_rl.training.run_curriculum import run_curriculum

        result = run_curriculum(
            config_path=args.config,
            device=args.device,
            root_dir=args.output_prefix,
        )
        print(json.dumps(result, indent=2))
        return

    from minefield_rl.viz.pygame_renderer import PygameRenderer

    renderer = PygameRenderer(
        config=config,
        agent=args.agent,
        checkpoint_path=args.checkpoint,
        device=args.device,
        map_size=args.map_size,
    )
    renderer.run()


if __name__ == "__main__":
    main()
