# Minefield Navigator RL

Procedural partially-observable navigation: a recurrent agent sees only a 9×9 fog-of-war window, has limited health, must reason around walls and mines, and is trained with a staged pipeline (imitation → recurrent PPO → MCTS-guided GRPO) over a multi-stage curriculum.

The environment is Gymnasium-compatible and fully procedural per episode. Every generated map is BFS-validated so there is always at least one valid path from start to exit.

## Headline result: mine = instant death, 20×20 dispersed

![L13 GRPO 20×20 mine=death, every step](assets/l13_grpo_minedeath_20x20_every_step.gif)

The same recurrent policy fine-tuned with `max_health = 1`, so a single mine hit terminates the episode. IL warm-starts from the L13 PPO best (20×20 dispersed mixed); PPO and GRPO then push the agent to navigate every dispersed profile without ever stepping on a mine. One frame per environment step at 1 fps across all four 20×20 dispersed profiles.

| Profile | Walls / Mines | Success | Death | Avg health left |
| --- | --- | --- | --- | --- |
| L10 dispwalls | 0.40 / 0.20 | **0.68** | 0.24 | 0.76 |
| L11 dispmines | 0.30 / 0.30 | **0.66** | 0.22 | 0.78 |
| L12 dispextreme | 0.40 / 0.30 | **0.90** | 0.06 | 0.94 |
| L13 dispopen | 0.20 / 0.20 | **0.48** | 0.46 | 0.54 |

(GRPO best, 50 evaluation episodes per profile, all under mine = instant death.)

Reproduce with `python tools/run_finetune_minedeath.py` — warm-starts from the 20×20 L13 PPO best.

## Architecture

The policy is a small recurrent actor-critic over the local 9×9 view plus a health scalar. The CNN+GRU stack is **grid-size-agnostic** — the same weights run unchanged on 10×10, 20×20, 30×30, etc., because only the egocentric window enters the network.

```text
                       Observation (per env step)
              ┌──────────────────────────────────────────┐
              │   obs : (2, 9, 9)        health : (1,)   │
              │   ch0 = walls/free       in [0, 1]       │
              │   ch1 = mines (visible)                  │
              └──────────────────────────────────────────┘
                              │
                              ▼
              ┌──────────────────────────────────────────┐
              │ CNN encoder   (egocentric, no padding-pad)│
              │   Conv2d(2 → 16, 3×3, pad=1) + ReLU       │
              │   Conv2d(16→ 32, 3×3, pad=1) + ReLU       │
              │   Conv2d(32→ 32, 3×3, pad=1) + ReLU       │
              │   Flatten → Linear(32·9·9 → 256) + ReLU   │
              └──────────────────────────────────────────┘
                              │  features (256)
                              ▼
              ┌──────────────────────────────────────────┐
              │ GRU memory   hidden_size = 256, 1 layer  │
              │   reset on episode_starts mask           │
              └──────────────────────────────────────────┘
                              │  rnn_out (256)
                              ▼              ┌──────────┐
                concat ◀──────────────────────│ health(1)│
                              │              └──────────┘
                              ▼
              ┌──────────────────────────────────────────┐
              │ Post-health trunk                        │
              │   Linear(257 → 128) + ReLU               │
              └──────────────────────────────────────────┘
                    │                              │
                    ▼                              ▼
        ┌────────────────────┐         ┌────────────────────┐
        │ Actor head         │         │ Critic head        │
        │ Linear(128→64)+ReLU│         │ Linear(128→64)+ReLU│
        │ Linear(64 → 8)     │         │ Linear(64 → 1)     │
        │ → action logits    │         │ → state value V(s) │
        └────────────────────┘         └────────────────────┘
                    │
                    ▼
              8-way move (N, NE, E, SE, S, SW, W, NW)
                  no diagonal corner-cutting
```

Reward shaping packs progress, mine penalty, living cost, invalid-move penalty, revisit penalty, loop penalty, and terminal goal/timeout/death rewards. At `max_health = 1` the planner refuses to step on any mine, so demonstrations are strictly mine-free.

## Training strategy

Three trainers stack on top of each other. Each one consumes the previous stage's checkpoint as `initial_checkpoint`, so prior skill carries forward instead of being overwritten. The curriculum then walks the policy through progressively harder distributions on progressively larger grids, ending with a mine-is-death fine-tune.

```text
   ┌─────────────────────────────────────────────────────────────────┐
   │                       Training pipeline (per stage)             │
   │                                                                 │
   │  ┌──────────────────┐   ┌──────────────────┐   ┌──────────────┐ │
   │  │ 1. Imitation     │   │ 2. Recurrent PPO │   │ 3. GRPO+MCTS │ │
   │  │ Learning (BC)    │──▶│ on-policy in env │──▶│ search-guided│ │
   │  │                  │   │                  │   │ policy update│ │
   │  │ A* expert with   │   │ GAE, clipped     │   │ KL-anchored  │ │
   │  │ mine-cost plans  │   │ ratio, value MSE │   │ to PPO trunk │ │
   │  │ → (s,a,return)   │   │ + entropy bonus  │   │ MCTS targets │ │
   │  │ → BC + value MSE │   │                  │   │ over groups  │ │
   │  └──────────────────┘   └──────────────────┘   └──────────────┘ │
   │          │                       │                     │       │
   │          └────warm-start ────────┼─────warm-start ─────┘       │
   │           ckpt model_state_dict only (architecture invariant)  │
   └─────────────────────────────────────────────────────────────────┘

                       Curriculum (each stage = full pipeline above)

   10×10 clustered (CA-smoothed walls + flood-fill mines)
   ┌────┬────┬────┬────┬────┬────┬────┬────┬────┐
   │ L1 │ L2 │ L3 │ L4 │ L5 │ L6 │ L7 │ L8 │ L9 │
   │open│mine│maze│mine│dens│dens│ext │ext │GEN │
   │    │d-op│    │maze│wall│mine│remel│remm│ +GRPO
   └─┬──┴────┴────┴────┴────┴────┴────┴────┴──┬─┘
     │                                         │
     │  warm-start L9 GRPO  →  scale grid/mode │
     ▼                                         ▼
   10×10 dispersed (uniform-random walls/mines, no clustering)
   ┌────┬────┬────┬────┐
   │ L10│ L11│ L12│ L13│      same 9×9-view weights, no retraining
   │d-w │d-m │d-x │mix │      of the CNN/GRU shape needed
   └────┴────┴────┴────┘
                 │
                 ▼  warm-start L13 GRPO  → grid 10 → 20
   20×20 dispersed
   ┌────┬────┬────┬────┐
   │ L10│ L11│ L12│ L13│
   └────┴────┴────┴────┘
                 │
                 ▼  warm-start L13 PPO  → max_health 3 → 1
   20×20 dispersed, MINE = INSTANT DEATH        ◀── headline result
   ┌──────────────────────────────────────────┐
   │ IL  →  PPO  →  GRPO  on mixed dispersed │
   │ per-profile eval + slow-step rendering  │
   └──────────────────────────────────────────┘
```

Why each piece earns its keep:

- **Imitation** boots the policy to a competent prior in minutes; PPO from scratch on the dispersed extreme profile would spend most of its budget on random-walk dying.
- **PPO** lets the agent discover behaviors the expert can't teach: detours that exploit the GRU memory of recently-seen mines, anti-loop maneuvers in dense walls, and value calibration under partial observability.
- **GRPO + MCTS** uses short search-improved rollouts as a teacher signal in regions where one local mistake is fatal — exactly the regime mine = instant death lives in.
- **Curriculum + warm-starts** — every distribution shift (clustered → dispersed, 10×10 → 20×20, health 3 → 1) reuses prior weights instead of starting over. The 9×9 view + small CNN is the structural reason this works: the input shape never changes.

## Environment

- Observation: `(2, 9, 9)` egocentric local window plus a scalar health channel
- Actions: 8-directional movement, with no diagonal corner cutting
- Health: configurable; default 3-hit, headline result at 1-hit (mine = death)
- Rewards: progress shaping, mine penalties, living cost, invalid-move penalties, revisit penalties, loop penalties, terminal rewards
- Map generation: fully procedural and BFS-validated every episode; supports both `clustered` (cellular-automaton walls + flood-fill mines) and `dispersed` (uniform-random) modes via `episode_profiles`

## Project layout

```text
minefield_rl/
  env/         # Environment, map generation, fast snapshots
  models/      # Recurrent PPO actor-critic, MCTS, DRQN
  planning/    # A* expert with health-aware mine cost
  training/    # imitation, PPO, GRPO trainers
  eval/        # batch evaluation and rollout rendering
  viz/         # pygame UI, charts, overlays
  configs/     # base config.yaml

tools/         # orchestrators (curriculum runners, fine-tunes, montage builder)
assets/        # demo media tracked in git
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
```

## Quickstart

```bash
# Evaluate the headline checkpoint
minefield-rl --mode eval   --agent ppo --checkpoint <path>/ppo_grpo_best.pt --scenario medium --episodes 50

# Watch a rollout
minefield-rl --mode play   --agent ppo --checkpoint <path>/ppo_grpo_best.pt --scenario medium

# Render to GIF + MP4
minefield-rl --mode render --agent ppo --checkpoint <path>/ppo_grpo_best.pt --scenario medium \
             --output-prefix minefield_rl/logs/rollout
```

## Reproduce the headline result

```bash
# 1. Progressive curriculum, 10×10 clustered: L1 → L9 (PPO) → L9 GRPO
python tools/run_progressive10.py
python tools/run_progressive10_extension.py

# 2. Dispersed mode at 10×10 then 20×20 (warm-starts from L9 GRPO)
python tools/run_progressive10_dispersed.py
python tools/run_progressive20_dispersed.py

# 3. Mine = instant death fine-tune (warm-starts from 20×20 L13 PPO)
python tools/run_finetune_minedeath.py

# 4. Build the slow-step montage from the per-profile renders
python tools/build_minedeath_montage.py
```

## Verification

```bash
python -m compileall minefield_rl
.venv/bin/python -m minefield_rl --help
```

## License

MIT
