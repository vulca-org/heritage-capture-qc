"""One page a conservator can read in a minute, and argue with.

The figure has to carry the honest version of the result, not the flattering
one: the saving comes mostly from stopping, the object is synthetic, and the
dose numbers are invented. All three are printed on the page rather than left
to the covering email, because a figure travels further than the message it
was attached to.

The d-prime curves here are scored against the stroke mask after the fact. The
planner never saw it; the marked stop points were chosen blind.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mhs_shim.backends.rti_sim import RTISimulator
from mhs_shim.manifest import ObjectSafetyManifest
from . import relief as R
from .capture_planner import plan_capture

FULL = "#1b1b1b"
SEQ = "#b45309"
ADA = "#1d4ed8"


def _dome(stack_dir, budget=100.0, lux=2000.0, exp_s=0.5):
    m = ObjectSafetyManifest(object_id=Path(stack_dir).name, sensitivity="high",
                             lux_hours_budget=budget, granted_by="ILLUSTRATION ONLY")
    return RTISimulator(m, lux_at_object=lux, exposure_s=exp_s, stack_dir=stack_dir)


def _curve(dome, order, mask):
    """Legibility after each successive capture, scored post hoc."""
    ks, ds = [], []
    for k in range(3, len(order) + 1):
        sub = order[:k]
        n = R.solve_normals(dome.light_dirs[sub], np.stack([dome.frame(i) for i in sub]))
        ks.append(k)
        ds.append(R.oracle_dprime(R.relief_map(n), mask))
    return np.array(ks), np.array(ds)


def _relief_at(dome, order, k):
    sub = order[:k]
    n = R.solve_normals(dome.light_dirs[sub], np.stack([dome.frame(i) for i in sub]))
    r = R.relief_map(n)
    lo, hi = np.percentile(r, [2, 98])
    return np.clip((r - lo) / max(hi - lo, 1e-9), 0, 1)


def build(stack_dir: str | Path, out_png: str | Path, sweep_note: str = "") -> str:
    stack_dir = Path(stack_dir)
    mask = np.load(stack_dir / "_stroke_mask.npy")
    dome = _dome(stack_dir)

    seq = plan_capture(_dome(stack_dir), dome.frame, policy="sequential")
    ada = plan_capture(_dome(stack_dir), dome.frame, policy="adaptive")
    full_order = list(range(dome.n_lights))

    k_seq, d_seq = _curve(dome, seq.order + [i for i in full_order if i not in seq.order], mask)
    k_ada, d_ada = _curve(dome, ada.order + [i for i in full_order if i not in ada.order], mask)
    d_full = _curve(dome, full_order, mask)[1][-1]
    dose = dome.dose_per_capture

    fig = plt.figure(figsize=(11.7, 8.3), dpi=150)          # A4 landscape
    gs = fig.add_gridspec(3, 4, height_ratios=[1.0, 1.0, 1.35], hspace=0.55, wspace=0.10,
                          left=0.085, right=0.965, top=0.865, bottom=0.135)

    fig.suptitle("Stopping a multi-light capture when the reconstruction stops changing",
                 x=0.085, ha="left", fontsize=15.5, weight="bold", y=0.965)
    fig.text(0.085, 0.918,
             "Synthetic incised inscription under a 48-LED dome. The planner picks the next light and decides when to "
             "stop without ever seeing\nwhere the strokes are; legibility is scored afterwards against the known mask.",
             ha="left", fontsize=9.5, color="#444", va="top")

    # row 1 — what the dome actually records
    for j, li in enumerate([0, 5, 21, 37]):
        ax = fig.add_subplot(gs[0, j])
        ax.imshow(dome.frame(li), cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"light {li}", fontsize=9.5, pad=4)
        ax.set_xticks([]); ax.set_yticks([])
        if j == 0:
            ax.set_ylabel("What the dome\nrecords", fontsize=10, weight="bold", labelpad=10)

    # row 2 — the same object rebuilt from different subsets
    full_seq = seq.order + [i for i in full_order if i not in seq.order]
    full_ada = ada.order + [i for i in full_order if i not in ada.order]
    panels = [(3, full_seq, "ring order", SEQ, False),
              (3, full_ada, "adaptive", ADA, False),
              (ada.n_captures, full_ada, "adaptive, stopped", ADA, True),
              (48, full_order, "full dome", FULL, False)]
    for j, (k, order, label, colour, thick) in enumerate(panels):
        ax = fig.add_subplot(gs[1, j])
        ax.imshow(_relief_at(dome, order, k), cmap="gray")
        n = R.solve_normals(dome.light_dirs[order[:k]], np.stack([dome.frame(i) for i in order[:k]]))
        dp = R.oracle_dprime(R.relief_map(n), mask)
        ax.set_title(f"{label}\n{k} captures · {k*dose:.2f} lux·h · d\u2032 {dp:.2f}", fontsize=9.5, pad=4)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor(colour); sp.set_linewidth(2.4 if thick else 1.2)
        if j == 0:
            ax.set_ylabel("Relief rebuilt\nfrom those frames", fontsize=10, weight="bold", labelpad=10)

    # row 3 — the curve
    ax = fig.add_subplot(gs[2, :])
    xmax = 48 * dose
    ax.axhline(d_full, color=FULL, lw=1.2, ls="--")
    ax.text(xmax * 0.995, d_full + 0.035, "full dome, 48 captures", ha="right", fontsize=9, color=FULL)
    ax.plot(k_seq * dose, d_seq, color=SEQ, lw=2.2, label="ring order (the usual wiring)")
    ax.plot(k_ada * dose, d_ada, color=ADA, lw=2.2, label="adaptive next-light")
    placement = ((ada.n_captures, d_ada, ADA, (105, -58), "left", "adaptive"),
                 (seq.n_captures, d_seq, SEQ, (105, -112), "left", "ring order"))
    for k, d, c, xytext, ha, name in placement:
        y = d[k - 3]
        ax.plot([k * dose], [y], "o", color=c, ms=9, zorder=5)
        ax.annotate(f"{name} stops here: {k} captures · {k*dose:.2f} lux·h · {100*y/d_full:.0f}% of full",
                    (k * dose, y), textcoords="offset points", xytext=xytext, ha=ha,
                    fontsize=9, color=c, zorder=6,
                    arrowprops=dict(arrowstyle="-", color=c, lw=0.9, shrinkA=0, shrinkB=5))
    ax.set_xlabel("light dose delivered to the object (lux·hours)", fontsize=10)
    ax.set_ylabel("stroke legibility\n(d\u2032, scored post hoc)", fontsize=10)
    ax.set_xlim(0, xmax * 1.01)
    ax.set_ylim(min(d_seq.min(), d_ada.min()) - 0.45, d_full + 0.18)
    ax.grid(alpha=0.25)
    ax.legend(loc="center right", frameon=False, fontsize=9.5, bbox_to_anchor=(1.0, 0.42))
    ax.set_title("Legibility saturates long before the dome is finished", fontsize=11,
                 weight="bold", loc="left", pad=6)

    caveat = ("Read this the unflattering way. Most of the saving is the stopping rule, not the choosing: stopping takes "
              "the dose to about a quarter, choosing takes it to about a sixth.\n"
              "The object, the dome and the dose figures are all synthetic. No real object has been imaged, and the light "
              "budget is invented \u2014 that is the part a conservator has to supply."
              + (("\n" + sweep_note) if sweep_note else ""))
    fig.text(0.085, 0.012, caveat, fontsize=8.8, color="#333", va="bottom", linespacing=1.5)

    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight", facecolor="white")
    # A PDF as well: an attachment a conservator can print at full size, and the
    # form an email is more likely to carry than a link to an unfamiliar host.
    out_pdf = out_png.with_suffix(".pdf")
    fig.savefig(out_pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return str(out_png)
