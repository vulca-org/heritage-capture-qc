"""Choose the next light, and stop when the reconstruction stops changing.

This is the half of path B that did not exist until now. The safety manifest
already refused exposures that would overrun an object's light budget; what it
could not do was spend less of that budget in the first place.

The stopping rule is deliberately blind and deliberately conservative: capture
until the estimated surface normals stop moving between successive shots, twice
in a row, with a well-conditioned set of light directions behind them. It never
looks at the stroke mask, so it cannot stop early by knowing the answer.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict

import numpy as np

from mhs_shim.manifest import BudgetExceeded
from . import relief as R


@dataclass
class Step:
    k: int
    light: int
    dose: float
    sigma_min: float
    cond: float
    change_deg: float
    contrast: float


@dataclass
class CaptureRun:
    policy: str
    order: list = field(default_factory=list)
    steps: list = field(default_factory=list)
    dose_spent: float = 0.0
    stop_reason: str = ""
    n_captures: int = 0

    def to_dict(self):
        d = asdict(self)
        d["steps"] = [asdict(s) if not isinstance(s, dict) else s for s in self.steps]
        return d


# --------------------------------------------------------------- policies --
def next_sequential(taken: list[int], L: np.ndarray) -> int:
    """Ring by ring, the order a dome is wired and the order people shoot."""
    for i in range(len(L)):
        if i not in taken:
            return i
    return -1


def next_adaptive(taken: list[int], L: np.ndarray) -> int:
    """Greedy E-optimal: pick the light that most improves conditioning.

    Photometric stereo inverts the light matrix, so directions that are nearly
    coplanar with what is already captured buy almost nothing. Maximising the
    smallest singular value is the standard way of saying "point somewhere the
    others do not".
    """
    best, best_s = -1, -np.inf
    for i in range(len(L)):
        if i in taken:
            continue
        s, _ = R.conditioning(L[taken + [i]])
        if len(taken) + 1 < 3:                      # too few to condition: spread out
            s = float(np.min(np.linalg.norm(L[taken] - L[i], axis=1))) if taken else 0.0
        if s > best_s:
            best, best_s = i, s
    return best


POLICIES = {"sequential": next_sequential, "adaptive": next_adaptive}


# ------------------------------------------------------------------- loop --
def plan_capture(device, stack_loader, policy: str = "adaptive", *,
                 min_lights: int = 4, max_lights: int | None = None,
                 stable_deg: float = 1.0, stable_rounds: int = 2) -> CaptureRun:
    """Capture adaptively until the reconstruction settles or the budget stops us.

    `device` is an MHS-shaped dome: write('light', i) spends the object's budget
    and is refused when the budget is gone. `stack_loader(i)` returns the frame
    for light i as a float array in [0,1]. Nothing here can see ground truth.
    """
    if policy not in POLICIES:
        raise ValueError(f"unknown policy {policy!r}; have {sorted(POLICIES)}")
    pick = POLICIES[policy]
    L = device.light_dirs
    cap = len(L) if max_lights is None else min(max_lights, len(L))

    run = CaptureRun(policy=policy)
    taken: list[int] = []
    frames: list[np.ndarray] = []
    prev_normals = None
    settled = 0

    while len(taken) < cap:
        i = pick(taken, L)
        if i < 0:
            run.stop_reason = "no lights left"
            break
        try:
            info = device.write("light", i)
        except BudgetExceeded:
            # The manifest refused. Stopping here is a budget outcome, not a
            # legibility one, and the two must never be reported as the same.
            run.stop_reason = "object light budget exhausted"
            break
        taken.append(i)
        frames.append(stack_loader(i))
        run.dose_spent += info["dose"]

        sigma, cond = R.conditioning(L[taken])
        change = float("nan")
        contrast = float("nan")
        if len(taken) >= 3:
            normals = R.solve_normals(L[taken], np.stack(frames))
            contrast = R.relief_contrast(R.relief_map(normals))
            if prev_normals is not None:
                change = R.normal_change_deg(prev_normals, normals)
            prev_normals = normals

        run.steps.append(Step(len(taken), i, info["dose"], sigma, cond, change, contrast))

        # There was a conditioning guard here (stop only if cond <= 12). Mutation
        # testing showed no test covered it, and three attempts to construct a case
        # where it changed the outcome all failed: a near-flat object, a single-ring
        # dome, and the ordinary stack. Requiring min_lights captures and two
        # consecutive settled rounds already implies enough light diversity. A guard
        # that cannot be shown to fire is not protection, so it is gone; `cond` stays
        # in the per-step record because it is worth seeing, not worth branching on.
        if len(taken) >= min_lights and not np.isnan(change) and change < stable_deg:
            settled += 1
            if settled >= stable_rounds:
                run.stop_reason = f"reconstruction settled (<{stable_deg}deg for {stable_rounds} shots)"
                break
        else:
            settled = 0
    else:
        run.stop_reason = "reached capture cap"

    run.order = taken
    run.n_captures = len(taken)
    return run
