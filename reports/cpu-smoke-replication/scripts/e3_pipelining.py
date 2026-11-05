"""E3: serial vs pipelined execution (paper Sec 2.3 / 6.2, Figs. 13-14).

Uses the real ParallelToolExecutor + ResourceScheduler with a toy exclusive "move"
tool (sleep = physical execution) and a scripted agent whose "model requests" are sleeps.
Times are scaled down ~100x (seconds instead of minutes). Each synthetic task has a
different reasoning/execution balance and a fraction of steps whose next decision
depends on the previous action's result (those must wait, as in the paper).
"""
import threading
import time

import numpy as np

from uni_robot_agent.execution.parallel import ParallelToolExecutor
from uni_robot_agent.execution.scheduler import ToolExecutionPolicy
from uni_robot_agent.tools.toolkit import ToolDef, ToolOutput, Toolkit

LOCK = threading.Lock()


def make_executor(mode, log):
    def move(duration: float, tag: str):
        t0 = time.monotonic()
        time.sleep(duration)
        with LOCK:
            log["action"].append((t0, time.monotonic()))
        return ToolOutput(extra={"tag": tag})

    tk = Toolkit()
    tk.register_tools([ToolDef(name="move", handler=move, spec={
        "name": "move", "description": "toy move", "input_schema": {"type": "object"}})])
    policy = {"move": ToolExecutionPolicy(resource_key="left_arm", return_mode="background",
                                         concurrency="exclusive", mutates_world=True)}
    return ParallelToolExecutor(tk, policy, tool_return_mode=mode)


def episode(task, pipelined, seed):
    rng = np.random.default_rng(seed)
    log = {"action": [], "model": []}
    ex = make_executor("background" if pipelined else "immediate", log)
    t_start = time.monotonic()
    pending = []
    for i in range(task["steps"]):
        r = rng.uniform(*task["reason"])
        e = rng.uniform(*task["exec"])
        dependent = rng.random() < task["dep"]
        if pipelined and dependent and pending:
            # Decision needs the outcome of submitted work: wait (batched wait).
            for job in pending:
                ex.job_result(job, timeout=None)
            pending.clear()
        t0 = time.monotonic()
        time.sleep(r)  # model request
        log["model"].append((t0, time.monotonic()))
        if pipelined:
            res = ex.submit("move", {"duration": e, "tag": str(i)})
            pending.append(res.job_id)
        else:
            ex.call("move", {"duration": e, "tag": str(i)})
    for job in pending:
        ex.job_result(job, timeout=None)
    total = time.monotonic() - t_start
    ex.close()
    act = sum(b - a for a, b in log["action"])
    overlap = sum(max(0.0, min(b, d) - max(a, c)) for a, b in log["action"] for c, d in log["model"])
    return total, overlap / act


TASKS = {  # (reasoning seconds range, execution seconds range, P(dependent decision))
    "reason-heavy":      dict(steps=12, reason=(0.30, 0.60), exec=(0.05, 0.15), dep=0.3),
    "balanced":          dict(steps=12, reason=(0.15, 0.30), exec=(0.15, 0.30), dep=0.3),
    "exec-heavy":        dict(steps=12, reason=(0.05, 0.15), exec=(0.30, 0.50), dep=0.3),
    "mostly-dependent":  dict(steps=12, reason=(0.15, 0.30), exec=(0.15, 0.30), dep=0.9),
    "open-loop batch":   dict(steps=12, reason=(0.15, 0.30), exec=(0.15, 0.30), dep=0.0),
}


def main():
    print(f"{'task':18s} {'serial_s':>8s} {'pipe_s':>7s} {'reduction%':>10s} {'overlap_serial%':>15s} {'overlap_pipe%':>13s}")
    for i, (name, task) in enumerate(TASKS.items()):
        ts, os_ = episode(task, False, seed=i)
        tp, op = episode(task, True, seed=i)
        print(f"{name:18s} {ts:8.2f} {tp:7.2f} {100*(ts-tp)/ts:10.1f} {100*os_:15.1f} {100*op:13.1f}")


if __name__ == "__main__":
    main()
