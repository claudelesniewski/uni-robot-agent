"""Overview video + per-episode thumbnails for the Opus RoboUse batch.

Successes (green) and failures (red) in separate grids; every clip is resampled to the same
number of frames so all episodes play start-to-finish together, then hold their final frame.
"""
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

S = Path(os.environ["WORK"])
OUT = S / "deliverables" / "opus_batch"
T, HOLD, FPS = 60, 25, 10
TILE = 90
GREEN, RED, GREY = (40, 180, 70), (215, 50, 50), (90, 90, 90)


def font(size, bold=False):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    try:
        return ImageFont.truetype(f"/usr/share/fonts/truetype/dejavu/{name}", size)
    except OSError:
        return ImageFont.load_default()


def short(task):
    name = task.split("/")[-1]
    for p in ("metaworld-", "robouse-", "robosuite-", "gymrobotics-fetch-", "kitchen-", "libero-", "arc-", "hard-", "safety-"):
        name = name.replace(p, "")
    return name.replace("-vision", " (v)").replace("-panda", "")


def load(args):
    task, ok, size = args
    p = S / "runs" / "batch" / f"opus__{task.replace('/', '__')}" / "env" / "native" / "recording.mp4"
    color = GREEN if ok else RED
    if p.exists():
        frames = [f for f in imageio.get_reader(str(p), "ffmpeg")]
        idx = np.linspace(0, len(frames) - 1, T).round().astype(int)
        clip = [frames[i] for i in idx]
    else:
        blank = np.full((320, 320, 3), 40, np.uint8)
        im = Image.fromarray(blank); ImageDraw.Draw(im).text((90, 150), "no motion", fill=(200, 200, 200), font=font(28))
        clip = [np.asarray(im)] * T
    out = []
    for f in clip:
        im = Image.fromarray(f).convert("RGB").resize((size - 6, size - 6))
        canvas = Image.new("RGB", (size, size), color)
        canvas.paste(im, (3, 3))
        d = ImageDraw.Draw(canvas)
        label = short(task)
        fs = max(9, size // 11)
        d.rectangle((3, size - 3 - fs - 3, size - 3, size - 3), fill=(0, 0, 0))
        d.text((5, size - 3 - fs - 2), label[: int(size / (fs * 0.55))], fill=(255, 255, 255), font=font(fs))
        out.append(np.asarray(canvas))
    return task, out


def main():
    rows = json.load(open(S / "runs" / "opus_batch_rows.json"))
    rows = [r for r in rows if not r["infra"]]
    rows.sort(key=lambda r: (r["group"], r["task"]))
    succ = [r for r in rows if r["ok"]]
    fail = [r for r in rows if not r["ok"]]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "thumbs").mkdir(exist_ok=True)

    with ProcessPoolExecutor(4) as ex:
        tiles = dict(ex.map(load, [(r["task"], r["ok"], TILE) for r in rows], chunksize=4))
        thumbs = dict(ex.map(load, [(r["task"], r["ok"], 200) for r in rows], chunksize=4))

    # per-episode thumbnails (200 px) and a contact sheet of final frames
    for r in rows:
        name = r["task"].replace("/", "__")
        clip = thumbs[r["task"]]
        imageio.mimsave(str(OUT / "thumbs" / f"{'ok' if r['ok'] else 'fail'}__{name}.mp4"),
                        clip + [clip[-1]] * HOLD, fps=FPS, macro_block_size=1)

    def grid(sel, cols, t):
        rows_n = -(-len(sel) // cols)
        g = np.full((rows_n * TILE, cols * TILE, 3), 18, np.uint8)
        for i, r in enumerate(sel):
            y, x = divmod(i, cols)
            g[y * TILE:(y + 1) * TILE, x * TILE:(x + 1) * TILE] = tiles[r["task"]][min(t, T - 1)]
        return g

    W, H, top = 1920, 1080, 70
    sc, fc = 15, 5
    frames = []
    for t in range(T + HOLD):
        canvas = Image.new("RGB", (W, H), (18, 18, 18))
        d = ImageDraw.Draw(canvas)
        d.text((20, 12), "Opus + uni-robot-agent harness on CPU RoboUse tasks (RGB + calibration, one episode each)",
               fill=(235, 235, 235), font=font(24, True))
        d.text((20, 44), f"{len(succ)}/{len(rows)} succeeded  ·  tabletop, MetaWorld, kitchen, robosuite, Fetch  ·  "
                         f"each clip time-normalized to the full episode", fill=(170, 170, 170), font=font(16))
        gs = grid(succ, sc, t)
        canvas.paste(Image.fromarray(gs), (20, top + 26))
        d.text((20, top + 2), f"SUCCESS ({len(succ)})", fill=GREEN, font=font(20, True))
        fx = 20 + sc * TILE + 30
        gf = grid(fail, fc, t)
        canvas.paste(Image.fromarray(gf), (fx, top + 26))
        d.text((fx, top + 2), f"FAILED ({len(fail)})", fill=RED, font=font(20, True))
        prog = min(t, T - 1) / (T - 1)
        d.rectangle((20, H - 14, 20 + int((W - 40) * prog), H - 8), fill=(120, 120, 120))
        frames.append(np.asarray(canvas))
    imageio.mimsave(str(OUT / "opus_batch_overview.mp4"), frames, fps=FPS, macro_block_size=1, quality=7)
    Image.fromarray(frames[-1]).save(OUT / "opus_batch_overview_final.png")
    print(len(succ), len(fail), "->", OUT)


if __name__ == "__main__":
    main()
