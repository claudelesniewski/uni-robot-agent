"""E2: depth-free localization by multi-view triangulation (paper Sec 4.3 / 5.1, App. E).

Uses the real triangulate() tool with synthetic calibrated pinhole cameras. Simulates an
agent picking corresponding pixels with a given pixel-selection error (sigma) and reports
the tool's rejection rate (the paper's spatial-grounding failure metric) and 3D error.
"""
from types import SimpleNamespace

import numpy as np

from uni_robot_agent.tools.scene.triangulation import triangulate

H, W, F = 480, 640, 600.0
K = np.array([[F, 0, W / 2], [0, F, H / 2], [0, 0, 1]])


def look_at(eye, target):
    """Optical frame: z forward, x right, y down; returns T_world_from_optical."""
    z = np.asarray(target, float) - eye
    z /= np.linalg.norm(z)
    x = np.cross(z, [0, 0, 1.0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    T = np.eye(4)
    T[:3, :3] = np.c_[x, y, z]
    T[:3, 3] = eye
    return T


CAMERAS = {
    "head": look_at(np.array([0.0, 0.0, 1.0]), [0.5, 0, 0]),
    "left": look_at(np.array([0.2, 0.5, 0.6]), [0.5, 0, 0]),
    "right": look_at(np.array([0.2, -0.5, 0.6]), [0.5, 0, 0]),
}


def frame(T):
    return SimpleNamespace(rgb=np.zeros((H, W, 3), np.uint8), intrinsic_K=K, T_world_from_optical=T, calibration=None)


def project(T, p):
    o = T[:3, :3].T @ (p - T[:3, 3])
    uv = K @ o
    return uv[:2] / uv[2]


def main(n=400, seed=0):
    rng = np.random.default_rng(seed)
    pts = np.c_[rng.uniform(0.35, 0.65, n), rng.uniform(-0.2, 0.2, n), rng.uniform(0.0, 0.15, n)]
    print(f"{'views':6s} {'sigma_px':>8s} {'reject%':>8s} {'mean_err_mm(accepted)':>22s} {'p95_err_mm':>10s}")
    for names in (("head", "left"), ("head", "left", "right")):
        for sigma in (0, 1, 2, 4, 8, 16):
            rejected, errs = 0, []
            for p in pts:
                pix = []
                for nm in names:
                    u, v = project(CAMERAS[nm], p) + rng.normal(0, sigma, 2)
                    r, c = int(round(np.clip(v, 0, H - 1))), int(round(np.clip(u, 0, W - 1)))
                    pix.append([r, c])
                res = triangulate([frame(CAMERAS[nm]) for nm in names], pix)
                if not res["valid"]:
                    rejected += 1
                else:
                    errs.append(np.linalg.norm(np.array(res["point_xyz"]) - p) * 1000)
            e = np.array(errs) if errs else np.array([np.nan])
            print(f"{len(names):<6d} {sigma:8d} {100*rejected/n:8.1f} {np.nanmean(e):22.2f} {np.nanpercentile(e,95):10.2f}")


if __name__ == "__main__":
    main()
