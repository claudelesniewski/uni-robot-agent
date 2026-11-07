"""Run a prioritized queue of RoboUse episodes with a Claude planner, N in parallel, with a cost cap.

Resumable: an episode with agent/summary.json is skipped. Writes batch/results.jsonl and prints progress.
"""
import argparse
import json
import queue
import subprocess
import threading
import time
from pathlib import Path

import os
S = Path(os.environ["WORK"])
RUNNER = Path(__file__).resolve().parent / "run_episode_p.sh"
BATCH = S / "runs" / "batch"
SCRIPT = {"tabletop": "tabletop", "metaworld": "metaworld", "kitchen": "kitchen", "robosuite": "robosuite",
          "gymrobotics": "fetch"}
DONE_BEFORE = {"metaworld/metaworld-button-press-topdown", "metaworld/metaworld-drawer-close",
               "metaworld/metaworld-push", "metaworld/metaworld-pick-place"}


def priority(task, backend):
    group = task.split("/")[0]
    name = task.split("/")[-1]
    if group == "robo-use-families":
        return 0
    if name.startswith("hard-uncal") or name in {"robouse-hanoi-3", "robouse-precise-landing"}:
        return 1
    if backend in ("kitchen", "robosuite", "gymrobotics"):
        return 2
    if backend == "tabletop":
        return {"hard": 3, "libero-style": 4, "arc-style": 5, "vision": 6, "safety": 7}.get(group, 8)
    return 9  # metaworld


def needs_run(out):
    """Run if never finished, or if the last attempt failed for infrastructure reasons."""
    try:
        summ = json.loads((out / "agent" / "summary.json").read_text())
    except Exception:
        return True
    return bool(summ.get("infra_error")) or (summ.get("cost_usd") in (0, 0.0) and summ.get("turns", 0) > 1)


def read_result(out):
    r = {}
    try:
        r.update(json.loads((out / "agent" / "summary.json").read_text()))
    except Exception:
        pass
    try:
        n = json.loads((out / "env" / "native" / "result.json").read_text())
        r.update(native_success=n.get("success"), progress=n.get("metrics", {}).get("progress"),
                 steps=n.get("steps_used"), max_steps=n.get("max_steps"))
    except Exception:
        pass
    if (out / "startup_error.txt").exists():
        r["startup_error"] = True
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="opus")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--budget-usd", type=float, default=100.0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("agent_args", nargs="*", default=["--max-turns", "40", "--wall-s", "900", "--code"])
    args = ap.parse_args()
    inv = json.loads((S / "probe" / "inventory.json").read_text())["admitted"]
    tasks = [(priority(t["task"], t["backend"]), t["task"], t["backend"]) for t in inv
             if t["backend"] in SCRIPT and t["task"] not in DONE_BEFORE]
    tasks.sort()
    if args.limit:
        tasks = tasks[:args.limit]
    BATCH.mkdir(parents=True, exist_ok=True)
    q = queue.Queue()
    for item in tasks:
        q.put(item)
    lock, spent, done, halted = threading.Lock(), [0.0], [0], [False]
    results = open(BATCH / "results.jsonl", "a")
    # Spend of earlier valid attempts is added as each skipped run is logged below.
    t0 = time.time()
    print(f"queued {len(tasks)} episodes, {args.workers} workers, budget ${args.budget_usd}", flush=True)

    def worker(port):
        while True:
            try:
                prio, task, backend = q.get_nowait()
            except queue.Empty:
                return
            name = f"{args.model}__{task.replace('/', '__')}"
            out = BATCH / name
            if needs_run(out):
                with lock:
                    if halted[0]:
                        return
                    if spent[0] >= args.budget_usd:
                        print(f"budget reached (${spent[0]:.2f}); not starting {task}", flush=True)
                        return
                subprocess.run(["bash", str(RUNNER), SCRIPT[backend], task, args.model, name, str(port), *args.agent_args],
                               timeout=1800)
            r = read_result(out)
            if str(r.get("infra_error") or "").startswith("planner_unavailable"):
                with lock:
                    if not halted[0]:
                        print(f"HALT: planner unavailable ({r['infra_error'][:120]}); no new episodes will start", flush=True)
                    halted[0] = True
            row = {"task": task, "backend": backend, "priority": prio, "model": args.model, **r}
            row.pop("session_dir", None)
            with lock:
                spent[0] += r.get("cost_usd") or 0.0
                done[0] += 1
                results.write(json.dumps(row) + "\n"); results.flush()
                ok = "✓" if r.get("native_success") else ("ERR" if r.get("startup_error") or r.get("infra_error") else "✗")
                print(f"[{done[0]}/{len(tasks)} {time.time()-t0:5.0f}s ${spent[0]:.2f}] {ok} {task} "
                      f"turns={r.get('turns')} steps={r.get('steps')} cost=${r.get('cost_usd', 0) or 0:.3f} "
                      f"wall={r.get('wall_s')}", flush=True)

    threads = [threading.Thread(target=worker, args=(18800 + i,)) for i in range(args.workers)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    print(f"finished: {done[0]} episodes, ${spent[0]:.2f}, {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
