"""Reconstruct surface relief from a partial multi-light stack, and judge it.

Two kinds of measure live here and they must not be confused:

  * Blind measures — conditioning, stability, relief contrast — are computed
    from the captured images alone. Only these may drive a stopping decision,
    because at capture time nobody knows where the strokes are. That is the
    whole point: if the stopping rule could see the answer, the experiment
    would be measuring nothing.
  * Oracle measures — d_prime against the known stroke mask — exist only to
    score the experiment afterwards. Every function below that touches ground
    truth carries `oracle` in its name.
"""
from __future__ import annotations

import cv2
import numpy as np


# ---------------------------------------------------------------- blind ----
def solve_normals(L: np.ndarray, stack: np.ndarray) -> np.ndarray:
    """Least-squares photometric stereo. L is (k,3); stack is (k,H,W) in [0,1]."""
    k, h, w = stack.shape
    g = np.linalg.pinv(L) @ stack.reshape(k, -1)          # (3, H*W)
    g = g.reshape(3, h, w).transpose(1, 2, 0)
    norm = np.linalg.norm(g, axis=2, keepdims=True)
    return g / np.maximum(norm, 1e-6)


def conditioning(L: np.ndarray) -> tuple[float, float]:
    """(smallest singular value, condition number) of the light matrix.

    Three lights crowded together solve the same equation three times; the
    smallest singular value is what says so.
    """
    if len(L) < 3:
        return 0.0, float("inf")
    s = np.linalg.svd(L, compute_uv=False)
    return float(s[-1]), float(s[0] / max(s[-1], 1e-9))


def relief_map(normals: np.ndarray) -> np.ndarray:
    """Incision-enhancing rendering: divergence of the tilt field.

    A groove tilts the surface inward on both walls, so the in-plane normal
    components converge across it and the divergence spikes along the stroke.
    """
    nx, ny = normals[..., 0], normals[..., 1]
    d = cv2.Sobel(nx, cv2.CV_32F, 1, 0, ksize=3) + cv2.Sobel(ny, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.GaussianBlur(-d, (0, 0), 1.0)


def normal_change_deg(a: np.ndarray, b: np.ndarray) -> float:
    """Mean angle between two normal maps, in degrees."""
    dot = np.clip((a * b).sum(axis=2), -1.0, 1.0)
    return float(np.degrees(np.arccos(dot)).mean())


def relief_contrast(relief: np.ndarray) -> float:
    """Blind sharpness of the relief rendering: robust spread of its response."""
    x = relief.ravel()
    lo, hi = np.percentile(x, [1, 99])
    med = np.median(x)
    mad = np.median(np.abs(x - med)) + 1e-9
    return float((hi - lo) / mad)


# --------------------------------------------------------------- oracle ----
def oracle_dprime(relief: np.ndarray, stroke_mask: np.ndarray) -> float:
    """Separation between stroke and background in the relief rendering.

    Signal-detection d-prime. Scoring only — never available to the planner.
    """
    m = stroke_mask > 0.5
    a, b = relief[m], relief[~m]
    if a.size == 0 or b.size == 0:
        return 0.0
    pooled = np.sqrt(0.5 * (a.var() + b.var())) + 1e-9
    return float(abs(a.mean() - b.mean()) / pooled)
