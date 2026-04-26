"""Render the GRPO best on L12_dispextreme: keep retrying seeds and pick the
longest successful rollout (so the headline GIF actually shows the policy
navigating dense walls + mines, not a 5-step adjacent-exit success).
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import yaml
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from minefield_rl.utils import deep_update, ensure_dir
import tools.run_finetune_minedeath as ft

RUN_ROOT = REPO_ROOT / "minefield_rl/logs/finetune_minedeath_run_20260425_180820"
GRPO_BEST = RUN_ROOT / "grpo/checkpoints/ppo_grpo_best.pt"

OUT_PREFIX = REPO_ROOT / "minefield_rl/logs/l12_grpo_minedeath_render"
OUT_GIF = REPO_ROOT / "assets/l12_grpo_minedeath_20x20_every_step.gif"
OUT_MP4 = REPO_ROOT / "assets/l12_grpo_minedeath_20x20_every_step.mp4"

NUM_ATTEMPTS = 12        # try this many seeds, pick longest success
MIN_STEPS = 12           # require at least this many steps to be "interesting"
HOLD_FRAMES = 3          # extra hold on terminal

HEADER = "Mine = instant death | GRPO best | L12 dispextreme (0.40w / 0.30m) | every step, 1 fps"


def _font(size: int) -> ImageFont.ImageFont:
    for cand in ("/System/Library/Fonts/SFNS.ttf", "/System/Library/Fonts/Helvetica.ttc",
                 "/Library/Fonts/Arial.ttf"):
        if Path(cand).exists():
            try:
                return ImageFont.truetype(cand, size)
            except Exception:
                pass
    return ImageFont.load_default()


def _wrap(img: Image.Image, header: str, footer: str) -> Image.Image:
    BAN = 36
    canvas = Image.new("RGB", (img.width, img.height + 2 * BAN), (16, 18, 22))
    canvas.paste(img, (0, BAN))
    draw = ImageDraw.Draw(canvas)
    f_top = _font(14)
    f_bot = _font(16)
    bbox = draw.textbbox((0, 0), header, font=f_top)
    draw.text(((canvas.width - (bbox[2] - bbox[0])) / 2, (BAN - (bbox[3] - bbox[1])) / 2 - 2),
              header, fill=(200, 210, 220), font=f_top)
    bbox = draw.textbbox((0, 0), footer, font=f_bot)
    y0 = BAN + img.height
    draw.text(((canvas.width - (bbox[2] - bbox[0])) / 2, y0 + (BAN - (bbox[3] - bbox[1])) / 2 - 2),
              footer, fill=(255, 255, 255), font=f_bot)
    return canvas


def _scale(img: Image.Image, w: int) -> Image.Image:
    if img.width == w:
        return img
    h = round(img.height * w / img.width)
    return img.resize((w, h), Image.LANCZOS)


def main() -> None:
    if not GRPO_BEST.exists():
        sys.exit(f"missing GRPO ckpt: {GRPO_BEST}")

    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    import pygame
    from minefield_rl.viz.pygame_renderer import PygameRenderer

    with (REPO_ROOT / "minefield_rl/configs/config.yaml").open() as fh:
        base_cfg = yaml.safe_load(fh)

    profile = ft.PROFILES["L12_dispextreme"]
    cfg = deepcopy(base_cfg)
    cfg = deep_update(cfg, {"env": ft._stage_env_overrides(profile)})

    frames_root = ensure_dir(OUT_PREFIX.parent / f"{OUT_PREFIX.name}_frames")
    if frames_root.exists():
        shutil.rmtree(frames_root)
    frames_root.mkdir(parents=True)

    renderer = PygameRenderer(
        config=cfg, agent="ppo", checkpoint_path=str(GRPO_BEST), device="cpu",
    )
    renderer.speed_index = len(renderer.speed_modes) - 1
    renderer.show_full_map = False

    best: dict | None = None
    for attempt in range(1, NUM_ATTEMPTS + 1):
        if attempt > 1:
            renderer._reset_episode()
        ep_dir = frames_root / f"ep_{attempt:02d}"
        ep_dir.mkdir()
        idx = 0
        renderer._draw()
        pygame.image.save(renderer.window, str(ep_dir / f"f_{idx:04d}.png"))
        idx += 1
        while renderer.episode_end_time is None:
            renderer._auto_step_if_needed()
            renderer._draw()
            pygame.image.save(renderer.window, str(ep_dir / f"f_{idx:04d}.png"))
            idx += 1
        for _ in range(HOLD_FRAMES):
            renderer._draw()
            pygame.image.save(renderer.window, str(ep_dir / f"f_{idx:04d}.png"))
            idx += 1
        outcome = str(renderer.info["outcome"])
        steps = int(renderer.info["steps"])
        print(f"attempt {attempt:2d}: outcome={outcome:8s} steps={steps:3d} "
              f"map_seed={int(renderer.info['map_seed'])}")
        if outcome == "success" and steps >= MIN_STEPS:
            if best is None or steps > best["steps"]:
                best = {"dir": ep_dir, "steps": steps, "frames": idx,
                        "seed": int(renderer.info["map_seed"])}
                # keep going to maybe find a longer one, but don't go forever
                if attempt >= 6 and best["steps"] >= 20:
                    break

    pygame.quit()

    if best is None:
        # fallback: any success
        for attempt in range(1, NUM_ATTEMPTS + 1):
            ep_dir = frames_root / f"ep_{attempt:02d}"
            if not ep_dir.exists():
                continue
            # take the longest one we have
            n = len(list(ep_dir.glob("f_*.png")))
            if best is None or n > best["frames"]:
                best = {"dir": ep_dir, "steps": n - HOLD_FRAMES - 1, "frames": n,
                        "seed": -1}
        if best is None:
            sys.exit("no rollouts produced")

    print(f"\nselected: steps={best['steps']} frames={best['frames']} seed={best['seed']}")

    # Build polished GIF: each PNG -> scaled + headered + footered, then ffmpeg at 1 fps.
    polished_dir = OUT_PREFIX.parent / f"{OUT_PREFIX.name}_polished"
    if polished_dir.exists():
        shutil.rmtree(polished_dir)
    polished_dir.mkdir(parents=True)

    pngs = sorted(best["dir"].glob("f_*.png"))
    footer = f"map_seed {best['seed']}  |  {best['steps']} steps  |  success"
    for i, p in enumerate(pngs):
        im = Image.open(p).convert("RGB")
        im = _scale(im, 720)
        wrapped = _wrap(im, HEADER, footer)
        wrapped.save(polished_dir / f"p_{i:04d}.png")

    OUT_GIF.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg"
    pattern = str(polished_dir / "p_%04d.png")

    palette = polished_dir / "palette.png"
    subprocess.run([ffmpeg, "-y", "-framerate", "1", "-i", pattern,
                    "-vf", "palettegen=stats_mode=full", str(palette)],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run([ffmpeg, "-y", "-framerate", "1", "-i", pattern, "-i", str(palette),
                    "-lavfi", "fps=1[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=4",
                    str(OUT_GIF)],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run([ffmpeg, "-y", "-framerate", "1", "-i", pattern,
                    "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", "30",
                    str(OUT_MP4)],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    shutil.rmtree(polished_dir)
    shutil.rmtree(frames_root)

    print(f"\nGIF: {OUT_GIF} ({OUT_GIF.stat().st_size/1e6:.2f} MB)")
    print(f"MP4: {OUT_MP4} ({OUT_MP4.stat().st_size/1e6:.2f} MB)")


if __name__ == "__main__":
    main()
