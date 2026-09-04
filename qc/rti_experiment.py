"""Does choosing the next light beat shooting the whole dome, at lower dose?

Three arms on one object, all reconstructing from the same stack:

  full        every LED, the way a dome is normally shot. Reference legibility.
  sequential  ring-by-ring order, stopped by the blind settling rule.
  adaptive    E-optimal next-light choice, stopped by the same blind rule.

`sequential` isolates the stopping rule; `adaptive` adds the selection. If
adaptive only matched sequential, the answer would be that stopping is what
mattered and the choosing was decoration. Scoring uses the stroke mask, which
neither policy can see.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from mhs_shim.backends.rti_sim import RTISimulator
from mhs_shim.manifest import ObjectSafetyManifest
from . import relief as R
from .capture_planner import plan_capture


def _score(dome: RTISimulator, order: list[int], mask: np.ndarray) -> dict:
    L = dome.light_dirs[order]
    stack = np.stack([dome.frame(i) for i in order])
    normals = R.solve_normals(L, stack)
    rel = R.relief_map(normals)
    sigma, cond = R.conditioning(L)
    return {"d_prime": R.oracle_dprime(rel, mask), "sigma_min": sigma, "cond": cond,
            "contrast": R.relief_contrast(rel)}


def run(stack_dir: str | Path, budget_lux_h: float = 100.0, out: str | Path | None = None,
        lux_at_object: float = 2000.0, exposure_s: float = 0.5) -> dict:
    stack_dir = Path(stack_dir)
    mask = np.load(stack_dir / "_stroke_mask.npy")
    meta = RTISimulator.stack_meta(stack_dir)

    def dome(budget: float) -> RTISimulator:
        m = ObjectSafetyManifest(object_id=stack_dir.name, sensitivity="high",
                                 lux_hours_budget=budget, granted_by="EXPERIMENT (invented number)")
        return RTISimulator(m, lux_at_object=lux_at_object, exposure_s=exposure_s, stack_dir=stack_dir)

    arms: dict[str, dict] = {}

    # full dome, the standard practice reference
    d = dome(budget_lux_h)
    order = list(range(d.n_lights))
    for i in order:
        d.write("light", i)
    arms["full"] = {"n_captures": len(order), "dose": d.manifest.lux_hours_used,
                    "stop_reason": "fixed full sequence", **_score(d, order, mask)}

    for policy in ("sequential", "adaptive"):
        d = dome(budget_lux_h)
        r = plan_capture(d, d.frame, policy=policy)
        arms[policy] = {"n_captures": r.n_captures, "dose": r.dose_spent,
                        "stop_reason": r.stop_reason, "order": r.order,
                        **_score(d, r.order, mask)}

    ref = arms["full"]["d_prime"]
    for k, a in arms.items():
        a["d_prime_pct_of_full"] = 100.0 * a["d_prime"] / ref if ref else float("nan")
        a["dose_pct_of_full"] = 100.0 * a["dose"] / arms["full"]["dose"]

    result = {"stack": str(stack_dir), "stack_meta": meta,
              "budget_lux_h": budget_lux_h, "dose_per_capture": arms["full"]["dose"] / arms["full"]["n_captures"],
              "arms": arms}
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(result, indent=2))
    return result


def render(res: dict) -> str:
    rows = ["| arm | captures | dose (lux·h) | dose vs full | d' | d' vs full | stop |",
            "|---|---|---|---|---|---|---|"]
    for k in ("full", "sequential", "adaptive"):
        a = res["arms"][k]
        rows.append(f"| {k} | {a['n_captures']} | {a['dose']:.3f} | {a['dose_pct_of_full']:.0f}% | "
                    f"{a['d_prime']:.3f} | {a['d_prime_pct_of_full']:.0f}% | {a['stop_reason']} |")
    return "\n".join(rows)
