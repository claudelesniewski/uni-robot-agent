"""E1: monitored motion execution (paper Sec 2.5 / Appendix D, Algorithm 1).

Drives the real MotionExecutor (uni_robot_agent.tools.gripper.motion) against a
CPU kinematic fake backend and checks waypoint bounds (Eq. 2-3) and each stop reason.
"""
import math
import numpy as np

from uni_robot_agent.services.environment.contracts import (
    ArmDescription, ArmState, EnvironmentDescription, EnvironmentState,
    GoalRegionReply, Pose, StepReceipt,
)
from uni_robot_agent.tools.gripper.contracts import MoveRequest, MoveTarget
from uni_robot_agent.tools.gripper.motion import MotionExecutor
from uni_robot_agent.tools.gripper.pose import (
    pose_waypoint, quaternion_distance, rpy_to_quaternion,
)

RADIUS = 0.1


class FakeArmBackend:
    """One arm whose measured pose follows the commanded waypoint with a lag."""

    def __init__(self, mode="nominal", gain=0.9, goal_search=False):
        self.mode, self.gain = mode, gain
        self.description = EnvironmentDescription(
            arms=[ArmDescription(name="left", control_frame="base", grasp_site_description="tcp",
                                 rotation_radius_m=RADIUS)],
            control_frequency_hz=10, max_episode_steps=1000,
            metadata={"goal_region_search": goal_search},
        )
        self.pose = np.array([0.3, 0.0, 0.3, 1, 0, 0, 0], float)
        self.version = self.steps = 0
        self.rejections = 0
        self.waypoint_moves = []

    def _state(self):
        return EnvironmentState(
            arms={"left": ArmState(control_pose=Pose(position=tuple(self.pose[:3]), quaternion=tuple(self.pose[3:])),
                                   gripper=1.0, gripper_command=1.0)},
            state_version=self.version, native_steps=self.steps)

    def read_state(self):
        return self._state()

    def step_pose(self, command):
        cmd = command.arms["left"].pose
        target = np.r_[cmd.position, cmd.quaternion]
        step_t = np.linalg.norm(target[:3] - self.pose[:3])
        step_r = quaternion_distance(target[3:], self.pose[3:])
        self.waypoint_moves.append((step_t, step_r))
        # IK rejects targets with x > 0.55 (outside a fake workspace).
        if self.mode == "ik_reject" and target[0] > 0.55:
            self.rejections += 1
            return StepReceipt(accepted=False, state=self._state(),
                               diagnostics={"reason": "ik_failed", "arm": "left"})
        new = self.pose.copy()
        new[:3] += self.gain * (target[:3] - self.pose[:3])
        new[3:] = target[3:]
        if self.mode == "blocked":   # table/obstacle at z = 0.2
            new[2] = max(new[2], 0.2)
        if self.mode == "diverge" and self.steps >= 3:  # controller runs away
            new[:3] = self.pose[:3] + np.array([0, 0.03, 0.0]) * (self.steps - 1)
        if self.mode == "jump" and self.steps == 2:     # collision kick
            new[:3] += np.array([0.3, 0, 0])
        self.pose = new
        self.steps += 1
        self.version += 1
        return StepReceipt(accepted=True, state=self._state())

    def solve_goal_region(self, req):
        # Return the nearest feasible pose inside 80% of the tolerance ball.
        goal = np.array(req.goal.position)
        alt = goal.copy()
        alt[0] = min(goal[0], 0.55)
        ok = np.linalg.norm(alt - goal) <= 0.8 * req.position_tolerance_m
        return GoalRegionReply(found=bool(ok), state_version=self.version,
                               pose=Pose(position=tuple(alt), quaternion=req.goal.quaternion) if ok else None)


def run(mode, target_xyz, rpy=(0, 0, 0), profile="standard", **kw):
    backend = FakeArmBackend(mode, **kw)
    fb = MotionExecutor(backend).move(MoveRequest(
        targets=[MoveTarget(arm="left", pose=(*target_xyz, *rpy))], profile=profile))
    arm = fb["arms"]["left"]
    return backend, fb, arm


def main():
    # Eq. 2/3 check: rotation step cap and shared translation budget.
    delta = 0.04
    theta_max = min(0.6, 2 * math.asin(min(1, delta / (4 * RADIUS))))
    cur = np.array([0, 0, 0, 1, 0, 0, 0], float)
    wp = pose_waypoint(cur, [1, 0, 0], rpy_to_quaternion([0, 0, 1.5]), max_step_m=delta, rotation_radius_m=RADIUS)
    dth = quaternion_distance(wp[3:], cur[3:])
    dp = np.linalg.norm(wp[:3])
    print(f"[Eq2] theta_max={theta_max:.4f} rad, waypoint rotation={dth:.4f} rad")
    print(f"[Eq3] expected translation={max(0, delta - 2*RADIUS*math.sin(dth/2)):.4f} m, got {dp:.4f} m")
    assert abs(dth - theta_max) < 1e-9 and abs(dp - (delta - 2 * RADIUS * math.sin(dth / 2))) < 1e-9

    rows = []
    cases = [
        ("nominal", dict(mode="nominal", target_xyz=(0.45, 0.1, 0.25), rpy=(0, 0, 0.5))),
        ("precision", dict(mode="nominal", target_xyz=(0.45, 0.1, 0.25), profile="precision")),
        ("blocked (z floor)", dict(mode="blocked", target_xyz=(0.3, 0.0, 0.1))),
        ("diverging", dict(mode="diverge", target_xyz=(0.5, 0.0, 0.3))),
        ("collision kick", dict(mode="jump", target_xyz=(0.5, 0.0, 0.3))),
        ("IK reject, no search", dict(mode="ik_reject", target_xyz=(0.565, 0.0, 0.3))),
        ("IK reject + goal search", dict(mode="ik_reject", target_xyz=(0.565, 0.0, 0.3), goal_search=True)),
        ("unreachable, budget", dict(mode="nominal", target_xyz=(0.3, 0.0, 0.3 + 2.0), gain=0.9)),
    ]
    for name, kw in cases:
        backend, fb, arm = run(**kw)
        max_t = max(m[0] for m in backend.waypoint_moves)
        rows.append((name, fb["stop_reason"], fb["steps_executed"], arm["position_error_m"],
                     fb["pose_reached"], max_t, fb["goal_recovery"] is not None and fb["goal_recovery"].get("found")))
    print(f"\n{'case':26s} {'stop_reason':18s} {'steps':>5s} {'pos_err_m':>9s} {'reached':>7s} {'max_cmd_step_m':>14s} recovered")
    for r in rows:
        print(f"{r[0]:26s} {r[1]:18s} {r[2]:5d} {r[3]:9.4f} {str(r[4]):>7s} {r[5]:14.4f} {r[6]}")
    expected = ["pose_reached", "pose_reached", "stalled", "diverging", "unsafe_motion",
                "planning_rejected", "pose_reached", "max_steps"]
    got = [r[1] for r in rows]
    print("\nall stop reasons as expected:", got == expected, "" if got == expected else f"expected {expected}")


if __name__ == "__main__":
    main()
