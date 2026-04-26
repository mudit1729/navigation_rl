"""Concatenate the 4 per-profile GRPO-best (mine=death) renders into a single
slow-step montage GIF + MP4. One frame per env step, 1 fps, with a labeled
banner card for each profile. Mirrors the style of l13_ppo_20x20_every_step.gif.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parent.parent
RUN_ROOT = REPO_ROOT / "minefield_rl/logs/finetune_minedeath_run_20260425_180820"
RENDERS = RUN_ROOT / "per_profile_renders"
OUT_GIF = REPO_ROOT / "assets/l13_grpo_minedeath_20x20_every_step.gif"
OUT_MP4 = REPO_ROOT / "assets/l13_grpo_minedeath_20x20_every_step.mp4"

# Order + labels (matches the earlier dispersed montage)
PROFILES = [
    ("L10_dispwalls",   "L10 dispwalls (0.40w / 0.20m)"),
    ("L11_dispmines",   "L11 dispmines (0.30w / 0.30m)"),
    ("L12_dispextreme", "L12 dispextreme (0.40w / 0.30m)"),
    ("L13_dispopen",    "L13 dispopen (0.20w / 0.20m)"),
]

TARGET_W = 600  # final gif width
BANNER_H = 36
HEADER = "Mine = instant death | GRPO best | 1 fps, every env step"
BANNER_HOLD_FRAMES = 2  # show profile card N frames before its rollout
TERMINAL_HOLD_FRAMES = 2  # extra hold on last frame of each rollout


def _font(size: int) -> ImageFont.ImageFont:
    for cand in ("/System/Library/Fonts/SFNS.ttf", "/System/Library/Fonts/Helvetica.ttc",
                 "/Library/Fonts/Arial.ttf"):
        if Path(cand).exists():
            try:
                return ImageFont.truetype(cand, size)
            except Exception:
                pass
    return ImageFont.load_default()


def _frames_of(gif_path: Path) -> list[Image.Image]:
    im = Image.open(gif_path)
    out = []
    try:
        while True:
            out.append(im.copy().convert("RGB"))
            im.seek(im.tell() + 1)
    except EOFError:
        pass
    return out


def _scale(img: Image.Image, w: int) -> Image.Image:
    if img.width == w:
        return img
    h = round(img.height * w / img.width)
    return img.resize((w, h), Image.LANCZOS)


def _wrap(img: Image.Image, label: str) -> Image.Image:
    canvas_w = img.width
    canvas_h = img.height + BANNER_H + BANNER_H  # header + footer
    canvas = Image.new("RGB", (canvas_w, canvas_h), (16, 18, 22))
    canvas.paste(img, (0, BANNER_H))
    draw = ImageDraw.Draw(canvas)
    f_top = _font(14)
    f_bot = _font(16)
    # top header (constant across the whole montage)
    bbox = draw.textbbox((0, 0), HEADER, font=f_top)
    draw.text(((canvas_w - (bbox[2] - bbox[0])) / 2, (BANNER_H - (bbox[3] - bbox[1])) / 2 - 2),
              HEADER, fill=(200, 210, 220), font=f_top)
    # bottom: profile name
    bbox = draw.textbbox((0, 0), label, font=f_bot)
    y0 = BANNER_H + img.height
    draw.text(((canvas_w - (bbox[2] - bbox[0])) / 2, y0 + (BANNER_H - (bbox[3] - bbox[1])) / 2 - 2),
              label, fill=(255, 255, 255), font=f_bot)
    return canvas


def _card(label: str, ref: Image.Image) -> Image.Image:
    canvas = Image.new("RGB", ref.size, (16, 18, 22))
    draw = ImageDraw.Draw(canvas)
    f = _font(28)
    bbox = draw.textbbox((0, 0), label, font=f)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    draw.text(((canvas.width - tw) / 2, (canvas.height - th) / 2 - 6),
              label, fill=(255, 255, 255), font=f)
    return canvas


def main() -> None:
    if not RUN_ROOT.exists():
        sys.exit(f"missing run root: {RUN_ROOT}")

    montage: list[Image.Image] = []
    for profile, label in PROFILES:
        gif = RENDERS / f"render_grpo_{profile}.gif"
        if not gif.exists():
            print(f"WARN missing {gif}", file=sys.stderr)
            continue
        frames = _frames_of(gif)
        scaled = [_scale(f, TARGET_W) for f in frames]
        wrapped = [_wrap(f, label) for f in scaled]
        # banner card before rollout
        card = _card(label, wrapped[0])
        montage.extend([card] * BANNER_HOLD_FRAMES)
        montage.extend(wrapped)
        # hold on terminal
        montage.extend([wrapped[-1]] * TERMINAL_HOLD_FRAMES)
        print(f"{profile}: {len(frames)} step frames")

    if not montage:
        sys.exit("no frames to write")

    OUT_GIF.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg"

    # Dump frames to disk so ffmpeg can build both GIF and MP4 from a clean source.
    tmp_dir = OUT_GIF.parent / ".minedeath_montage_frames"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True)
    for i, frame in enumerate(montage):
        frame.save(tmp_dir / f"f{i:04d}.png")
    pattern = str(tmp_dir / "f%04d.png")

    # GIF: palette pipeline at 1 fps
    palette = tmp_dir / "palette.png"
    subprocess.run([
        ffmpeg, "-y", "-framerate", "1", "-i", pattern,
        "-vf", "palettegen=stats_mode=full",
        str(palette),
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run([
        ffmpeg, "-y", "-framerate", "1", "-i", pattern, "-i", str(palette),
        "-lavfi", "fps=1[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=4",
        str(OUT_GIF),
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # MP4: 1 fps source held to 30 fps output for player compatibility
    subprocess.run([
        ffmpeg, "-y", "-framerate", "1", "-i", pattern,
        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", "30",
        str(OUT_MP4),
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    shutil.rmtree(tmp_dir)

    print(f"\nGIF: {OUT_GIF} ({OUT_GIF.stat().st_size/1e6:.2f} MB)")
    print(f"MP4: {OUT_MP4} ({OUT_MP4.stat().st_size/1e6:.2f} MB)")


if __name__ == "__main__":
    main()
