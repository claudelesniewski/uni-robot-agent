"""Robot-use agent with a Claude planner via `claude -p` (stream-json), driving uni-robot-agent tools.

- One persistent `claude` process per episode: built-in tools disabled, the harness's assembled
  robot prompt replaces Claude Code's system prompt, the conversation persists across turns.
- Each turn Claude replies with a ```json {"calls": [{"name": ..., "arguments": {...}}]}``` block;
  calls run in order through Session.tools (same executor/feedback/recording as the CLI).
- Tool feedback (compact JSON) and returned camera images go back as the next user message.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import re
import subprocess
import time
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image

from uni_robot_agent import Session, SessionConfig

ROBOT_TOOLS = ("view_env_state", "capture_scene", "locate", "move", "grasp", "release", "hold",
               "list_skills", "read_skill", "finish")
PROTOCOL = """
# How to call tools in this session
You cannot call tools natively here. To act, end every reply with exactly one fenced block:
```json
{"calls": [{"name": "<tool name>", "arguments": {...}}]}
```
Calls in one block run sequentially in the given order; you then receive each call's feedback
(JSON) and any returned camera images. Put several calls in one block only when later calls do
not depend on earlier results. Keep reasoning short. Call `finish` only after verifying success
in current observations. Available tools and their JSON-schema arguments:
"""


PY_PREAMBLE = """import math, json
import numpy as np
from scipy.spatial.transform import Rotation
from uni_robot_agent.tools.gripper.pose import rpy_to_quaternion, quaternion_to_rpy, quaternion_distance
"""
PY_SCHEMA = {
    "name": "python",
    "description": (
        "Run a short Python 3 snippet for computation only (geometry, coordinate transforms, "
        "quaternion/RPY conversion, triangulation or projection math). Preloaded: math, json, numpy as np, "
        "scipy Rotation, and the move tool's own conventions: rpy_to_quaternion([roll,pitch,yaw]) -> wxyz, "
        "quaternion_to_rpy(wxyz) -> [roll,pitch,yaw] (fixed-axis XYZ, R=Rz@Ry@Rx), quaternion_distance(q1,q2) -> rad. "
        "Use print() for results; stdout/stderr are returned (truncated). No robot, simulator, file, process or "
        "network access: robot actions and observations only go through the robot tools. 20 s timeout."),
    "input_schema": {"type": "object", "properties": {"code": {"type": "string", "minLength": 1}},
                     "required": ["code"], "additionalProperties": False},
}


def run_python(code, python_exe, workdir):
    import os, resource
    def limits():
        resource.setrlimit(resource.RLIMIT_AS, (2 << 30, 2 << 30))
        resource.setrlimit(resource.RLIMIT_CPU, (20, 20))
    dead = "http://127.0.0.1:9"  # soft guard: route urllib/requests-style traffic to a closed port
    env = {"PATH": "/usr/bin:/bin", "HOME": str(workdir), "OMP_NUM_THREADS": "1", "MPLBACKEND": "Agg",
           "HTTP_PROXY": dead, "HTTPS_PROXY": dead, "http_proxy": dead, "https_proxy": dead, "NO_PROXY": ""}
    try:
        r = subprocess.run([python_exe, "-I", "-c", PY_PREAMBLE + code], cwd=workdir, env=env, capture_output=True,
                           text=True, timeout=20, preexec_fn=limits)
        out = {"call_ok": r.returncode == 0, "returncode": r.returncode,
               "stdout": r.stdout[-4000:], "stderr": r.stderr[-2000:]}
    except subprocess.TimeoutExpired:
        out = {"call_ok": False, "error": "timeout after 20 s"}
    return json.dumps(out)


def compact(d, limit=3500):
    d = {k: v for k, v in d.items() if k != "traceback"}
    cal = d.pop("camera_calibration", None)
    if cal:  # keep calibration numbers (needed for geometry) but drop file paths
        d["camera_calibration"] = {c: {k: v for k, v in x.items() if k in ("image_size", "intrinsic_K", "T_world_from_optical")}
                                   for c, x in cal.items()}
    text = json.dumps(d, default=str, separators=(",", ":"))
    return text if len(text) <= limit else text[:limit] + "...(truncated)"


def parse_calls(text):
    blocks = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    start = text.find('{"calls"')
    if start >= 0:  # bare JSON (missing fences): decode the first complete object
        try:
            obj, _ = json.JSONDecoder().raw_decode(text[start:])
            blocks.append(json.dumps(obj))
        except json.JSONDecodeError:
            pass
    for b in reversed(blocks):
        try:
            calls = json.loads(b)["calls"]
            if isinstance(calls, list) and all(isinstance(c, dict) and "name" in c for c in calls):
                return calls
        except (json.JSONDecodeError, KeyError, TypeError):
            continue
    return None


class ClaudePlanner:
    def __init__(self, model, system_prompt, cwd):
        self.proc = subprocess.Popen(
            ["claude", "-p", "--input-format", "stream-json", "--output-format", "stream-json", "--verbose",
             "--model", model, "--tools", "", "--system-prompt", system_prompt, "--no-session-persistence",
             "--strict-mcp-config", "--disable-slash-commands"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, cwd=cwd)
        self.cost = 0.0

        import queue, threading
        self.lines = queue.Queue()
        threading.Thread(target=lambda: [self.lines.put(l) for l in self.proc.stdout] + [self.lines.put(None)],
                         daemon=True).start()

    def ask(self, content, timeout_s=180):
        t0 = time.time()
        self.proc.stdin.write(json.dumps({"type": "user", "message": {"role": "user", "content": content}}) + "\n")
        self.proc.stdin.flush()
        import queue
        while True:
            remaining = timeout_s - (time.time() - t0)
            if remaining <= 0:  # keepalive/stream lines must not extend the deadline
                raise TimeoutError(f"planner gave no result within {timeout_s}s")
            try:
                line = self.lines.get(timeout=remaining)
            except queue.Empty:
                raise TimeoutError(f"planner gave no result within {timeout_s}s")
            if line is None:
                raise RuntimeError("claude process ended without a result")
            ev = json.loads(line)
            if ev.get("type") == "result":
                self.cost = ev.get("total_cost_usd") or self.cost  # cumulative per session
                return ev.get("result") or "", ev, time.time() - t0

    def close(self):
        try:
            self.proc.stdin.close()
            self.proc.wait(timeout=10)
        except Exception:
            self.proc.kill()


def image_blocks(images):
    out = []
    for im in images:
        out.append({"type": "text", "text": f"[camera image: {im.name}]"})
        out.append({"type": "image", "source": {"type": "base64", "media_type": im.mime_type,
                                                 "data": base64.b64encode(im.data).decode()}})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env-url", default="http://127.0.0.1:18780")
    ap.add_argument("--model", default="haiku")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-turns", type=int, default=30)
    ap.add_argument("--wall-s", type=float, default=600)
    ap.add_argument("--max-images", type=int, default=4)
    ap.add_argument("--code", action="store_true", help="expose a computation-only python tool")
    args = ap.parse_args()
    out = Path(args.out); (out / "frames").mkdir(parents=True, exist_ok=True)
    log = open(out / "transcript.jsonl", "w")
    frames = []

    def record(images, label):
        if images:
            tiles = [np.asarray(Image.open(io.BytesIO(i.data)).convert("RGB").resize((320, 320))) for i in images[:2]]
            if len(tiles) == 1:
                tiles.append(np.zeros_like(tiles[0]))
            f = np.concatenate(tiles, axis=1)
            Image.fromarray(f).save(out / "frames" / f"frame_{len(frames):03d}_{label}.png")
            frames.append(f)

    session = Session(SessionConfig(integration="robouse", env_url=args.env_url, no_sam3=True,
                                    output_dir=str(out / "harness")))
    session.start()
    schemas = [{"name": s["name"], "description": s["description"], "input_schema": s.get("input_schema") or s.get("parameters")}
               for s in session.tools.list() if s["name"] in ROBOT_TOOLS]
    if args.code:
        schemas.append(PY_SCHEMA)
    py_dir = out / "python_cwd"; py_dir.mkdir(exist_ok=True)
    import sys as _sys
    system = session._assembly.build_prompt(session._args.instruction).flatten() + "\n" + PROTOCOL + json.dumps(schemas, indent=1)
    (out / "system_prompt.md").write_text(system)
    cwd = out / "claude_cwd"; cwd.mkdir(exist_ok=True)
    planner = ClaudePlanner(args.model, system, str(cwd))

    obs = session.tools.call("view_env_state")
    record(obs.images, "initial")
    content = [{"type": "text", "text": "Task:\n" + session._args.instruction + "\n\nview_env_state feedback:\n" + compact(obs.to_dict())}] \
        + image_blocks(obs.images)
    t_start, finished, ended, n_calls, turn, infra_error = time.time(), False, False, 0, -1, None
    try:
        for turn in range(args.max_turns):
            if time.time() - t_start > args.wall_s:
                print("wall-clock budget reached"); break
            try:
                text, ev, dt = planner.ask(content)
            except TimeoutError as e:
                print(f"planner_timeout: {e}", flush=True); infra_error = str(e); break
            calls = parse_calls(text)
            u = ev.get("usage", {})
            print(f"turn {turn}: {dt:.0f}s ${planner.cost:.3f} in={u.get('input_tokens')}+cache{u.get('cache_read_input_tokens')} "
                  f"out={u.get('output_tokens')} calls={[c['name'] for c in calls] if calls else None} | {text[:200]!r}", flush=True)
            log.write(json.dumps({"turn": turn, "latency_s": dt, "cost_total": planner.cost, "text": text}) + "\n"); log.flush()
            if not calls:
                content = "No valid ```json {\"calls\": [...]}``` block found. Reply again ending with one."
                continue
            parts, new_images = [], []
            for c in calls:
                name, arguments = c["name"], c.get("arguments") or {}
                if name == "python" and args.code:
                    n_calls += 1
                    result = run_python(str(arguments.get("code", "")), _sys.executable, py_dir)
                    print(f"   -> python({len(str(arguments.get('code','')))} chars) -> {result[:160]}", flush=True)
                    parts.append({"type": "text", "text": "python result:\n" + result})
                    log.write(json.dumps({"turn": turn, "tool": "python", "arguments": arguments, "result": result}) + "\n"); log.flush()
                    continue
                if name not in ROBOT_TOOLS:
                    parts.append({"type": "text", "text": f"{name}: unknown tool"}); continue
                res = session.tools.call(name, arguments)
                n_calls += 1
                fb = res.extra.get("feedback") if isinstance(res.extra.get("feedback"), dict) else {}
                print(f"   -> {name}({json.dumps(arguments)[:160]}) ok={res.success} stop={fb.get('stop_reason')} err={(res.error or '')[:150]}", flush=True)
                parts.append({"type": "text", "text": f"{name} feedback:\n" + compact(res.to_dict())})
                log.write(json.dumps({"turn": turn, "tool": name, "arguments": arguments, "ok": res.success,
                                      "stop_reason": fb.get("stop_reason")}) + "\n"); log.flush()
                record(res.images, f"t{turn}_{name}")
                new_images += res.images
                if name == "finish" and res.success:
                    finished = True
                    break
                if fb.get("stop_reason") == "native_end" or (res.extra.get("state") or {}).get("ended"):
                    ended = True
                    break
            if finished or ended:
                if ended:
                    print("episode ended natively; stopping", flush=True)
                break
            content = parts + image_blocks(new_images[-args.max_images:])
    finally:
        planner.close()
        result = session.finish()
        summary = {"model": args.model, "code_tool": args.code, "agent_finished": finished, "task_success": result.task_success, "infra_error": infra_error,
                   "turns": turn + 1, "tool_calls": n_calls, "wall_s": round(time.time() - t_start, 1),
                   "cost_usd": round(planner.cost, 4), "session_dir": str(result.session_dir)}
        print(json.dumps(summary), flush=True)
        json.dump(summary, open(out / "summary.json", "w"), indent=1)
        if frames:
            imageio.mimsave(out / "rollout.mp4", frames, fps=2, macro_block_size=1)


if __name__ == "__main__":
    main()
