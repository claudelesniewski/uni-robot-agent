"""Minimal robot-use agent: a small local VLM drives uni-robot-agent tools via the SDK.

Pared-down stand-in for the paper's Codex/Claude-runtime agent:
- planner: OpenAI chat-completions endpoint (llama-server serving Qwen3-VL-2B on CPU)
- prompt: the harness's own assembled system prompt and tool schemas (subset)
- tools: executed through Session.tools (same executor, feedback and recording as the CLI)
- context: only the newest observation images are kept (sliding window)
Every frame returned by a tool is saved; frames are stitched into an MP4 at the end.
"""
from __future__ import annotations

import argparse
import base64
import json
import time
from pathlib import Path

import httpx
import imageio.v2 as imageio
import numpy as np
from PIL import Image
import io

from uni_robot_agent import Session, SessionConfig

TOOLS = ("view_env_state", "capture_scene", "locate", "move", "grasp", "release", "hold", "finish")


def oai_tools(schemas, allowed=TOOLS):
    out = []
    for s in schemas:
        if s["name"] in allowed:
            out.append({"type": "function", "function": {
                "name": s["name"], "description": s["description"][:1500],
                "parameters": s.get("input_schema") or s.get("parameters")}})
    return out


def compact(d, limit=2500):
    d = {k: v for k, v in d.items() if k not in ("traceback", "camera_calibration")}
    text = json.dumps(d, default=str, separators=(",", ":"))
    return text if len(text) <= limit else text[:limit] + "...(truncated)"


def image_parts(images):
    parts = []
    for im in images:
        parts.append({"type": "text", "text": f"[camera: {im.name}]"})
        parts.append({"type": "image_url", "image_url": {
            "url": f"data:{im.mime_type};base64,{base64.b64encode(im.data).decode()}"}})
    return parts


class Recorder:
    def __init__(self, out_dir: Path):
        self.out_dir = out_dir
        self.frames = []
        out_dir.mkdir(parents=True, exist_ok=True)

    def add(self, images, label):
        if not images:
            return
        tiles = [np.asarray(Image.open(io.BytesIO(im.data)).convert("RGB").resize((320, 320))) for im in images[:2]]
        frame = np.concatenate(tiles, axis=1)
        idx = len(self.frames)
        Image.fromarray(frame).save(self.out_dir / f"frame_{idx:03d}_{label}.png")
        self.frames.append(frame)

    def video(self, path: Path, fps=2):
        if self.frames:
            imageio.mimsave(path, self.frames, fps=fps, macro_block_size=1)


def chat(client, url, model, messages, tools):
    t0 = time.time()
    r = client.post(f"{url}/chat/completions", json={
        "model": model, "messages": messages, "tools": tools, "tool_choice": "auto",
        "temperature": 0.2, "max_tokens": 512}, timeout=900)
    r.raise_for_status()
    body = r.json()
    return body["choices"][0]["message"], body.get("usage", {}), time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env-url", default="http://127.0.0.1:18780")
    ap.add_argument("--llm-url", default="http://127.0.0.1:8080/v1")
    ap.add_argument("--model", default="qwen3-vl-2b")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-turns", type=int, default=12)
    ap.add_argument("--wall-s", type=float, default=600)
    ap.add_argument("--tools", default=",".join(TOOLS))
    args = ap.parse_args()
    out = Path(args.out)
    rec = Recorder(out / "frames")
    log = open(out / "transcript.jsonl", "w")

    session = Session(SessionConfig(integration="robouse", env_url=args.env_url, no_sam3=True,
                                    output_dir=str(out / "harness")))
    session.start()
    a = session._assembly  # assembled prompt, as passed to built-in agents
    system = a.build_prompt(session._args.instruction).flatten()
    tools = oai_tools(session.tools.list(), tuple(args.tools.split(",")))

    obs = session.tools.call("view_env_state")
    rec.add(obs.images, "initial")
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": [{"type": "text", "text": "Task:\n" + session._args.instruction
                                       + "\n\nCurrent state:\n" + compact(obs.to_dict())}] + image_parts(obs.images)},
    ]
    t_start, finished = time.time(), False
    client = httpx.Client()
    for turn in range(args.max_turns):
        if time.time() - t_start > args.wall_s:
            print("wall-clock budget reached"); break
        msg, usage, dt = chat(client, args.llm_url, args.model, messages, tools)
        calls = msg.get("tool_calls") or []
        print(f"turn {turn}: {dt:.0f}s, prompt={usage.get('prompt_tokens')} out={usage.get('completion_tokens')}, "
              f"calls={[c['function']['name'] for c in calls]} text={(msg.get('content') or '')[:160]!r}")
        log.write(json.dumps({"turn": turn, "latency_s": dt, "usage": usage, "message": msg}) + "\n"); log.flush()
        messages.append({"role": "assistant", "content": msg.get("content") or "", **({"tool_calls": calls} if calls else {})})
        if not calls:
            messages.append({"role": "user", "content": "Continue by calling a tool. Call finish only when verified."})
            continue
        new_images = []
        for c in calls:
            name = c["function"]["name"]
            try:
                arguments = json.loads(c["function"].get("arguments") or "{}")
            except json.JSONDecodeError as e:
                result_text, images = json.dumps({"call_ok": False, "error": f"invalid JSON arguments: {e}"}), []
            else:
                res = session.tools.call(name, arguments)
                result_text, images = compact(res.to_dict()), res.images
                print(f"   -> {name}({json.dumps(arguments)[:200]}) ok={res.success} "
                      f"stop={res.extra.get('feedback', {}).get('stop_reason') if isinstance(res.extra.get('feedback'), dict) else None} "
                      f"err={(res.error or '')[:150]}")
                rec.add(images, f"t{turn}_{name}")
                if name == "finish" and res.success:
                    finished = True
            log.write(json.dumps({"turn": turn, "tool": name, "result": result_text}) + "\n"); log.flush()
            messages.append({"role": "tool", "tool_call_id": c.get("id", name), "content": result_text})
            new_images += images
        # Sliding window: drop older images, attach only the newest ones.
        for m in messages[1:]:
            if m["role"] == "user" and isinstance(m["content"], list):
                m["content"] = [p if p["type"] != "image_url" else {"type": "text", "text": "[image omitted]"} for p in m["content"]]
        if new_images:
            messages.append({"role": "user", "content": [{"type": "text", "text": "Latest images:"}] + image_parts(new_images[-2:])})
        if finished:
            break
    result = session.finish()
    summary = {"agent_finished": finished, "task_success": result.task_success,
               "turns": turn + 1, "wall_s": round(time.time() - t_start, 1), "session_dir": str(result.session_dir)}
    print(json.dumps(summary))
    json.dump(summary, open(out / "summary.json", "w"), indent=1)
    rec.video(out / "rollout.mp4")


if __name__ == "__main__":
    main()
