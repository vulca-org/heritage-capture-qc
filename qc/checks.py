"""Digitisation QC checks: technical (FADGI-style metadata) and visual.

Every check returns a CheckResult; thresholds are explicit and reported so an
archivist can argue with them.  Visual checks that depend on content (blur)
are expressed relative to the batch, not as absolute constants.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, asdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

PASS, WARN, FAIL = "pass", "warn", "fail"


@dataclass
class CheckResult:
    check: str
    status: str
    value: object
    threshold: object
    note: str = ""

    def to_dict(self):
        return asdict(self)


# ----------------------------------------------------------------------------
# technical
# ----------------------------------------------------------------------------
MASTER_FORMATS = {"TIFF", "PNG"}


def check_format(img: Image.Image) -> CheckResult:
    fmt = img.format or "?"
    ok = fmt in MASTER_FORMATS
    return CheckResult("format", PASS if ok else FAIL, fmt, sorted(MASTER_FORMATS),
                       "" if ok else "lossy or non-master format for a preservation master")


def check_dpi(img: Image.Image, expected: int, tol: float = 0.02) -> CheckResult:
    dpi = img.info.get("dpi")
    if not dpi:
        return CheckResult("dpi", FAIL, None, expected, "no resolution metadata embedded")
    x = float(dpi[0])
    ok = abs(x - expected) <= tol * expected
    return CheckResult("dpi", PASS if ok else FAIL, round(x, 1), expected,
                       "" if ok else "embedded resolution differs from declared capture resolution")


def check_bit_depth(img: Image.Image) -> CheckResult:
    mode = img.mode
    ok = mode in {"L", "RGB", "I;16", "RGB;16", "I;16B"}
    return CheckResult("bit_depth", PASS if ok else WARN, mode, "L/RGB/16-bit",
                       "" if ok else "unusual pixel mode for a master (palette, 1-bit, or alpha)")


# ----------------------------------------------------------------------------
# visual helpers
# ----------------------------------------------------------------------------

def to_gray(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert("L"), dtype=np.uint8)


def sharpness(gray: np.ndarray) -> float:
    """Exposure-invariant sharpness: var(Laplacian) / var(intensity)."""
    g = gray.astype(np.float32)
    lap = cv2.Laplacian(g, cv2.CV_32F)
    v = float(g.var())
    return float(lap.var() / v) if v > 1e-6 else 0.0


def check_blur(value: float, batch_median: float, fail_ratio=0.35, warn_ratio=0.6) -> CheckResult:
    if batch_median <= 0:
        return CheckResult("blur", WARN, value, None, "batch has no sharpness reference")
    r = value / batch_median
    status = FAIL if r < fail_ratio else WARN if r < warn_ratio else PASS
    return CheckResult("blur", status, round(r, 3), {"fail<": fail_ratio, "warn<": warn_ratio},
                       "sharpness relative to batch median" if status != PASS else "")


def check_glare(gray: np.ndarray, clip=250, blob_frac_fail=0.003, blob_frac_warn=0.001) -> CheckResult:
    mask = (gray >= clip).astype(np.uint8)
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    biggest = int(stats[1:, cv2.CC_STAT_AREA].max()) if n > 1 else 0
    frac = biggest / gray.size
    status = FAIL if frac > blob_frac_fail else WARN if frac > blob_frac_warn else PASS
    return CheckResult("glare", status, round(frac, 5), {"fail>": blob_frac_fail, "warn>": blob_frac_warn},
                       "largest clipped-highlight blob as fraction of frame" if status != PASS else "")


def check_exposure(gray: np.ndarray, lo=90, hi=245) -> CheckResult:
    med = float(np.median(gray))
    ok = lo <= med <= hi
    return CheckResult("exposure", PASS if ok else FAIL, round(med, 1), [lo, hi],
                       "" if ok else "median brightness outside working range (under/over-exposed)")


def ink_mask(gray: np.ndarray, min_contrast=20.0, frac=0.5) -> tuple[np.ndarray, bool]:
    """Separate ink from paper without assuming a fixed brightness.

    The paper dominates the histogram, so its median is the background level;
    ink sits near the dark tail.  Thresholding relative to those two makes the
    mask invariant to exposure, which a fixed cut is not.  Returns (mask, ok);
    ok is False when the page carries no real contrast (blank or ruined).
    """
    bg = float(np.median(gray))
    dark = float(np.percentile(gray, 1))
    if bg - dark < min_contrast:
        return np.zeros_like(gray, dtype=np.uint8), False
    thr = bg - frac * (bg - dark)
    return (gray < thr).astype(np.uint8), True


def estimate_skew(gray: np.ndarray, max_deg=5.0, coarse=0.25, fine=0.05, width=500) -> float:
    h0, w0 = gray.shape
    scale = width / w0
    small = cv2.resize(gray, (width, int(h0 * scale)), interpolation=cv2.INTER_AREA)
    binary, ok = ink_mask(small)
    if not ok or binary.sum() < 50:
        return 0.0
    h, w = binary.shape
    center = (w / 2, h / 2)

    def score(a):
        M = cv2.getRotationMatrix2D(center, a, 1.0)
        rot = cv2.warpAffine(binary, M, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)
        return rot.sum(axis=1).astype(np.float64).var()

    angles = np.arange(-max_deg, max_deg + 1e-9, coarse)
    best = max(angles, key=score)
    fine_angles = np.arange(best - coarse, best + coarse + 1e-9, fine)
    best = max(fine_angles, key=score)
    return float(best)


def check_skew(angle: float, fail_deg=1.0, warn_deg=0.5) -> CheckResult:
    a = abs(angle)
    status = FAIL if a > fail_deg else WARN if a > warn_deg else PASS
    return CheckResult("skew", status, round(angle, 2), {"fail>": fail_deg, "warn>": warn_deg},
                       "page rotation in degrees (correction angle)" if status != PASS else "")


def check_margins(gray: np.ndarray, edge_frac=0.01) -> CheckResult:
    mask, ok = ink_mask(gray)
    ys, xs = np.where(mask > 0)
    if not ok or len(xs) == 0:
        return CheckResult("margins", WARN, None, edge_frac, "no ink found (blank page?)")
    h, w = gray.shape
    touch = []
    if xs.min() <= edge_frac * w: touch.append("left")
    if xs.max() >= (1 - edge_frac) * w: touch.append("right")
    if ys.min() <= edge_frac * h: touch.append("top")
    if ys.max() >= (1 - edge_frac) * h: touch.append("bottom")
    ok = not touch
    return CheckResult("margins", PASS if ok else FAIL, touch, edge_frac,
                       "" if ok else "content reaches frame edge: possible crop")


# ----------------------------------------------------------------------------
# batch-level
# ----------------------------------------------------------------------------

def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def thumb_vector(gray: np.ndarray, size=64) -> np.ndarray:
    t = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32).ravel()
    t -= t.mean()
    n = np.linalg.norm(t)
    return t / n if n > 0 else t


def check_sequence(seqs: list[int | None]) -> tuple[list[int], list[int]]:
    nums = [s for s in seqs if s is not None]
    missing, dup = [], []
    if nums:
        seen = set()
        for s in nums:
            if s in seen: dup.append(s)
            seen.add(s)
        missing = [i for i in range(min(nums), max(nums) + 1) if i not in seen]
    return missing, dup


# ----------------------------------------------------------------------------
# batch-relative: a fault visible only by comparing frames to their siblings
# ----------------------------------------------------------------------------

def region_vector(gray: np.ndarray, region=(0.60, 0.70, 1.0, 1.0), size=48) -> np.ndarray:
    """Normalised thumbnail of a sub-region, for comparing frames to each other."""
    h, w = gray.shape
    x0, y0, x1, y1 = int(region[0] * w), int(region[1] * h), int(region[2] * w), int(region[3] * h)
    crop = gray[y0:y1, x0:x1]
    if crop.size == 0:
        return np.zeros(size * size, dtype=np.float32)
    return thumb_vector(crop, size)


def batch_region_consistency(vectors: list[np.ndarray], floor=0.5, drop=0.35,
                             min_consistency=0.6) -> list[CheckResult]:
    """Flag frames whose target region does not look like the rest of the batch.

    A missing colour or scale target is invisible in a single frame — nothing
    about that frame is wrong on its own.  It only shows up against the batch,
    which is why this cannot be delegated to a per-frame judge, human or model.
    Correlation against the batch's median region separates the odd frame out.
    """
    if len(vectors) < 3:
        return [CheckResult("target_region", PASS, None, None, "batch too small to compare") for _ in vectors]
    stack = np.vstack(vectors)
    med = np.median(stack, axis=0)
    n = np.linalg.norm(med)
    if n < 1e-6:
        return [CheckResult("target_region", PASS, None, None, "no structure in region") for _ in vectors]
    med = med / n
    corr = stack @ med
    typical = float(np.median(corr))
    if typical < min_consistency:
        # The region is not a repeated fixture in this batch — it is just page
        # content, which differs from frame to frame by design. There is no
        # "normal" to deviate from, so the check abstains rather than inventing
        # one. Abstaining is the correct answer here, not a weaker one.
        return [CheckResult("target_region", PASS, round(float(c), 3), {"batch_median": round(typical, 3)},
                            "no constant target region in this batch; check abstains") for c in corr]
    out = []
    for c in corr:
        c = float(c)
        status = FAIL if (c < floor and typical - c > drop) else PASS
        out.append(CheckResult("target_region", status, round(c, 3),
                               {"batch_median": round(typical, 3), "floor": floor, "drop": drop},
                               "target region unlike the rest of the batch" if status == FAIL else ""))
    return out
