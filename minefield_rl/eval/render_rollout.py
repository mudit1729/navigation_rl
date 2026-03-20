from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

from minefield_rl.utils import ensure_dir


def render_rollout(
    config: dict[str, Any],
    checkpoint_path: str | Path,
    output_prefix: str | Path,
    agent: str = "ppo",
    device: str = "cpu",
    max_attempts: int = 3,
    sample_every: int = 8,
    hold_frames: int = 16,
    prefer_success: bool = True,
) -> dict[str, Any]:
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

    import pygame

    from minefield_rl.viz.pygame_renderer import PygameRenderer

    output_prefix = Path(output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    frames_root = ensure_dir(output_prefix.parent / f"{output_prefix.name}_frames")

    renderer = PygameRenderer(
        config=config,
        agent=agent,
        checkpoint_path=str(checkpoint_path),
        device=device,
    )
    renderer.speed_index = len(renderer.speed_modes) - 1
    renderer.show_full_map = False

    best_meta: dict[str, Any] | None = None
    best_rank = (-1, -1)
    live_path = output_prefix.parent / f"{output_prefix.name}_live.png"
    terminal_path = output_prefix.parent / f"{output_prefix.name}_terminal.png"
    live_saved = False

    for attempt in range(1, max_attempts + 1):
        if attempt > 1:
            renderer._reset_episode()
        episode_dir = frames_root / f"episode_{attempt:02d}"
        if episode_dir.exists():
            shutil.rmtree(episode_dir)
        episode_dir.mkdir(parents=True)

        frame_index = 0
        renderer._draw()
        pygame.image.save(renderer.window, str(episode_dir / f"frame_{frame_index:04d}.png"))
        frame_index += 1

        while renderer.episode_end_time is None:
            renderer._auto_step_if_needed()
            if renderer.info["steps"] % sample_every == 0 or renderer.episode_end_time is not None:
                renderer._draw()
                pygame.image.save(renderer.window, str(episode_dir / f"frame_{frame_index:04d}.png"))
                frame_index += 1
            if (not live_saved) and renderer.info["outcome"] is None and renderer.info["steps"] >= max(20, sample_every * 4):
                renderer._draw()
                pygame.image.save(renderer.window, str(live_path))
                live_saved = True

        for _ in range(hold_frames):
            renderer._draw()
            pygame.image.save(renderer.window, str(episode_dir / f"frame_{frame_index:04d}.png"))
            frame_index += 1

        renderer.show_full_map = True
        renderer._draw()
        pygame.image.save(renderer.window, str(terminal_path))
        renderer.show_full_map = False

        outcome = str(renderer.info["outcome"])
        rank_primary = 3 if outcome == "success" else 2 if outcome == "death" else 1
        rank_secondary = int(renderer.info["steps"]) if outcome == "success" else -int(renderer.info["steps"])
        if not prefer_success and outcome == "timeout":
            rank_secondary = int(renderer.info["steps"])
        rank = (rank_primary, rank_secondary)
        meta = {
            "attempt": attempt,
            "episode": renderer.episode,
            "map_seed": int(renderer.info["map_seed"]),
            "steps": int(renderer.info["steps"]),
            "health": int(renderer.info["health"]),
            "outcome": outcome,
            "reward_raw": float(renderer.info["episode_reward_raw"]),
            "mine_hits": int(renderer.info["mine_hits"]),
            "frame_count": frame_index,
            "frames_dir": str(episode_dir.resolve()),
            "live_path": str(live_path.resolve()),
            "terminal_path": str(terminal_path.resolve()),
        }
        if rank > best_rank:
            best_rank = rank
            best_meta = meta
        if prefer_success and outcome == "success":
            break

    pygame.quit()
    if best_meta is None:
        raise RuntimeError("Failed to render any rollout")

    mp4_path = output_prefix.with_suffix(".mp4")
    gif_path = output_prefix.with_suffix(".gif")
    frame_pattern = str(Path(best_meta["frames_dir"]) / "frame_%04d.png")
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is None:
        raise RuntimeError("ffmpeg is required to render rollout videos")

    subprocess.run(
        [
            ffmpeg_path,
            "-y",
            "-framerate",
            "8",
            "-i",
            frame_pattern,
            "-vf",
            "format=yuv420p",
            str(mp4_path),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    subprocess.run(
        [
            ffmpeg_path,
            "-y",
            "-i",
            str(mp4_path),
            "-vf",
            "fps=8,scale=900:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse",
            str(gif_path),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    best_meta["mp4_path"] = str(mp4_path.resolve())
    best_meta["gif_path"] = str(gif_path.resolve())
    return best_meta
