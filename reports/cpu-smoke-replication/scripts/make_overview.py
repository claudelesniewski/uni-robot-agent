"""Overview assets for a set of rollouts: 2x2 grid video, filmstrip PNG, per-run annotated videos."""
import glob
import json
import re
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import os
S = Path(os.environ["WORK"])
RUNS = [("opus_bpt1", "button-press-topdown (easy)"), ("opus_drawer-close", "drawer-close (easy)"),
        ("opus_push", "push (medium)"), ("opus_pick-place", "pick-place (hard)")]
OUT = S / "deliverables" / "opus"
OUT.mkdir(parents=True, exist_ok=True)
try:
    FONT = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
    SMALL = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
except OSError:
    FONT = SMALL = ImageFont.load_default()


def label(img, lines, font=FONT, h=None):
    im = Image.fromarray(img).convert("RGB")
    d = ImageDraw.Draw(im)
    h = h or 6 + 17 * len(lines)
    d.rectangle((0, 0, im.width, h), fill=(0, 0, 0))
    for k, t in enumerate(lines):
        d.text((5, 3 + 17 * k), t, fill=(255, 255, 255), font=font if k == 0 else SMALL)
    return np.asarray(im)


def run_info(name):
    summ = json.load(open(S / "runs" / name / "agent" / "summary.json"))
    native = json.load(open(S / "runs" / name / "env" / "native" / "result.json"))
    frames = [json.loads(l) for l in open(S / "runs" / name / "env" / "native" / "frames.jsonl")]
    video = [f for f in imageio.get_reader(str(S / "runs" / name / "env" / "native" / "recording.mp4"), "ffmpeg")]
    return summ, native, frames, video


infos = {n: run_info(n) for n, _ in RUNS}

# 1) 2x2 grid video of native recordings (10 fps, last frame held, labelled with task/step/outcome).
T = max(len(v[3]) for v in infos.values()) + 15
grid_frames = []
for t in range(T):
    tiles = []
    for name, title in RUNS:
        summ, native, frames, video = infos[name]
        k = min(t, len(video) - 1)
        step = frames[min(k, len(frames) - 1)]["step"]
        done = k == len(video) - 1
        status = ("SUCCESS" if native["success"] else "FAIL") if done else "running"
        tiles.append(label(video[k], [title, f"step {step}/{native['max_steps']}  {status}"]))
    grid_frames.append(np.concatenate([np.concatenate(tiles[:2], 1), np.concatenate(tiles[2:], 1)], 0))
imageio.mimsave(str(OUT / "opus_grid.mp4"), grid_frames, fps=10, macro_block_size=1)

# 2) Filmstrip: rows = tasks, columns = 5 evenly spaced moments, plus a stats caption per row.
rows = []
for name, title in RUNS:
    summ, native, frames, video = infos[name]
    idx = np.linspace(0, len(video) - 1, 5).round().astype(int)
    tiles = [label(video[i], [f"step {frames[min(i, len(frames) - 1)]['step']}"]) for i in idx]
    strip = np.concatenate(tiles, 1)
    cap = np.zeros((26, strip.shape[1], 3), np.uint8)
    cap = label(cap, [f"{title}:  {'SUCCESS' if native['success'] else 'FAIL'} | {summ['turns']} turns, "
                      f"{summ['tool_calls']} tool calls, {summ['wall_s']:.0f}s wall, ${summ['cost_usd']:.2f}, "
                      f"{native['steps_used']} sim steps"], h=26)
    rows.append(np.concatenate([cap, strip], 0))
Image.fromarray(np.concatenate(rows, 0)).save(OUT / "opus_filmstrip.png")

# 3) Per-run annotated agent-view videos (what Opus saw: corner + topview after each tool call).
for name, title in RUNS:
    log = [json.loads(l) for l in open(S / "runs" / name / "agent" / "transcript.jsonl")]
    calls = [l for l in log if "tool" in l]
    out = []
    for f in sorted(glob.glob(str(S / "runs" / name / "agent" / "frames" / "*.png"))):
        m = re.match(r"frame_\d+_(?:t(\d+)_)?(.+)\.png", Path(f).name)
        turn, tool = m.group(1), m.group(2)
        stop = next((c.get("stop_reason") for c in calls if turn is not None and c["turn"] == int(turn) and c["tool"] == tool), None)
        line2 = f"turn {turn}: {tool}" + (f"  -> {stop}" if stop else "") if turn else "initial observation"
        img = np.asarray(Image.open(f).convert("RGB"))
        out += [label(img, [f"Opus agent view - {title}", line2])] * 2
    imageio.mimsave(str(OUT / f"{name}_agent_view.mp4"), out, fps=2, macro_block_size=1)
print(sorted(p.name for p in OUT.iterdir()))
