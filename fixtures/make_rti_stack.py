"""Synthesise a physically-grounded RTI light stack of an incised inscription.

There is no real dome here, so the stack has to be built rather than captured.
It is built from geometry rather than drawn by hand: an incised height field is
rendered under Lambertian shading from known light directions on a dome, with
mottled albedo and sensor noise. That makes two things true that a hand-drawn
fixture could not deliver — the light directions the planner reasons about are
the ones that actually produced the pixels, and the stroke mask is known
independently of anything the planner computes.
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

SIZE = 320
DEPTH = 2.6          # groove depth in height units
GROOVE_BLUR = 1.6    # tool/weathering softening of the groove walls


def _strokes(size: int = SIZE) -> np.ndarray:
    """Binary mask of incised strokes: a few angular marks, as on a stone slab."""
    m = np.zeros((size, size), np.uint8)
    seg = [((60, 70), (60, 250)), ((60, 70), (150, 70)), ((60, 160), (135, 160)),
           ((150, 70), (150, 250)), ((60, 250), (150, 250)),
           ((190, 70), (265, 70)), ((228, 70), (228, 250)),
           ((190, 160), (265, 160)), ((190, 250), (265, 250))]
    for a, b in seg:
        cv2.line(m, a, b, 255, 7)
    return (m > 0).astype(np.float32)


def _height(mask: np.ndarray, rng: np.random.Generator, depth: float = DEPTH,
            roughness: float = 0.28) -> np.ndarray:
    """Incised surface: grooves cut below a slightly rough plane."""
    groove = cv2.GaussianBlur(mask, (0, 0), GROOVE_BLUR)
    rough = cv2.GaussianBlur(rng.normal(0, 1, mask.shape).astype(np.float32), (0, 0), 2.0) * roughness
    return -depth * groove + rough


def _normals(h: np.ndarray) -> np.ndarray:
    dx = cv2.Sobel(h, cv2.CV_32F, 1, 0, ksize=3) / 8.0
    dy = cv2.Sobel(h, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    n = np.dstack([-dx, -dy, np.ones_like(h)])
    return n / np.linalg.norm(n, axis=2, keepdims=True)


def dome_lights(elevations=(25.0, 45.0, 65.0), n_az: int = 16) -> np.ndarray:
    """Light unit vectors, ring by ring — the order a dome is usually wired."""
    out = []
    for el in elevations:
        e = np.radians(el)
        for k in range(n_az):
            a = 2 * np.pi * k / n_az
            out.append([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
    return np.asarray(out, np.float32)


def make_stack(out_dir: str | Path, seed: int = 7, noise: float = 0.012,
               depth: float = DEPTH, roughness: float = 0.28,
               elevations=(25.0, 45.0, 65.0), n_az: int = 16) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    mask = _strokes()
    h = _height(mask, rng, depth=depth, roughness=roughness)
    n = _normals(h)
    albedo = 0.62 + 0.10 * cv2.GaussianBlur(rng.normal(0, 1, mask.shape).astype(np.float32), (0, 0), 6.0)
    albedo = np.clip(albedo, 0.35, 0.95)

    L = dome_lights(elevations, n_az)
    for i, l in enumerate(L):
        shade = np.clip((n * l).sum(axis=2), 0, None)
        img = np.clip(0.05 + albedo * shade + rng.normal(0, noise, mask.shape), 0, 1)
        cv2.imwrite(str(out / f"light_{i:02d}.png"), (img * 255).astype(np.uint8))

    np.save(out / "_lights.npy", L)
    np.save(out / "_stroke_mask.npy", mask)
    meta = {"n_lights": int(len(L)), "size": SIZE, "seed": seed, "noise": noise,
            "depth": depth, "roughness": roughness,
            "elevations": list(elevations), "n_azimuth": n_az,
            "note": "Lambertian render of an incised height field; mask is ground truth."}
    (out / "_meta.json").write_text(json.dumps(meta, indent=2))
    return meta


if __name__ == "__main__":
    import sys
    print(make_stack(sys.argv[1] if len(sys.argv) > 1 else "runs/rti_stack"))
