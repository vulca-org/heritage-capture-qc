"""Synthesise a digitisation batch with known, injected defects.

Ground truth is independent of the checks, so the tests discriminate: a
check that never fires, or fires on clean pages, is caught.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

W, H, DPI, BG, INK, MARGIN = 1200, 1600, 400, 228, 30, 100
WORDS = ("archive tower basement reading room ledger minute book folio verso recto "
         "jute mill wages dundee tay bridge whale oil harbour committee annual report").split()


def _font(size=26):
    try:
        return ImageFont.load_default(size=size)
    except Exception:
        return ImageFont.load_default()


def render_page(seed: int) -> Image.Image:
    rng = random.Random(seed)
    img = Image.new("L", (W, H), BG)
    d = ImageDraw.Draw(img)
    f = _font()
    y = MARGIN
    while y < H - MARGIN - 30:
        n = rng.randint(6, 11)
        d.text((MARGIN, y), " ".join(rng.choice(WORDS) for _ in range(n)), fill=INK, font=f)
        y += 40
    return img


def save(img: Image.Image, path: Path, dpi=DPI, fmt=None):
    img.save(path, dpi=(dpi, dpi), format=fmt)


def make(out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    gt: dict[str, list[str]] = {}
    pages = {i: render_page(i) for i in range(1, 14)}

    def name(i, ext="png"):
        return f"batch01_p{i:03d}.{ext}"

    # clean pages
    for i in (1, 4, 6):
        save(pages[i], out / name(i)); gt[name(i)] = []
    # 2: lossy master
    pages[2].save(out / name(2, "jpg"), quality=90, dpi=(DPI, DPI)); gt[name(2, "jpg")] = ["format"]
    # 3: blur
    save(pages[3].filter(ImageFilter.GaussianBlur(4)), out / name(3)); gt[name(3)] = ["blur"]
    # 5: skew 2.5°
    save(pages[5].rotate(2.5, resample=Image.BICUBIC, fillcolor=BG), out / name(5)); gt[name(5)] = ["skew"]
    # 7: glare blob over text
    g = pages[7].copy(); ImageDraw.Draw(g).ellipse((500, 600, 820, 900), fill=255)
    save(g, out / name(7)); gt[name(7)] = ["glare"]
    # 8: cropped — content touches left edge
    c = Image.new("L", (W, H), BG); c.paste(pages[8].crop((MARGIN + 2, 0, W, H)), (0, 0))
    save(c, out / name(8)); gt[name(8)] = ["margins"]
    # 9: byte-identical duplicate of page 6
    (out / name(9)).write_bytes((out / name(6)).read_bytes()); gt[name(9)] = ["duplicate"]
    # 10: missing (no file)
    # 11: wrong embedded resolution
    save(pages[11], out / name(11), dpi=150); gt[name(11)] = ["dpi"]
    # 12: under-exposed
    arr = (np.asarray(pages[12], dtype=np.float32) * 0.35).astype(np.uint8)
    save(Image.fromarray(arr), out / name(12)); gt[name(12)] = ["exposure"]
    # 13: near-duplicate of page 6 (re-shot with sensor noise)
    rng = np.random.default_rng(0)
    arr = np.clip(np.asarray(pages[6], dtype=np.int16) + rng.integers(-3, 4, (H, W)), 0, 255).astype(np.uint8)
    save(Image.fromarray(arr), out / name(13)); gt[name(13)] = ["duplicate:warn"]

    (out / "_ground_truth.json").write_text(json.dumps({"missing": [10], "files": gt}, indent=2))
    return gt


if __name__ == "__main__":
    import sys
    make(Path(sys.argv[1] if len(sys.argv) > 1 else "runs/fixture_batch"))


# ---------------------------------------------------------------------------
# defects the rule checks are structurally blind to
# ---------------------------------------------------------------------------
TARGET_XY = (W - 260, H - 90)


def _add_target(img: Image.Image) -> Image.Image:
    """Paste a small greyscale step target, as most capture setups include."""
    out = img.copy()
    d = ImageDraw.Draw(out)
    x, y = TARGET_XY
    for i, v in enumerate((20, 70, 120, 170, 220)):
        d.rectangle((x + i * 44, y, x + i * 44 + 40, y + 44), fill=v)
    return out


def make_vlm_fixtures(out: Path) -> dict:
    """A second batch whose faults are visible but not measurable.

    Every frame here is sharp, correctly exposed, square to the platen and at
    the declared resolution, so every rule check passes. Only looking at the
    picture separates them.
    """
    out.mkdir(parents=True, exist_ok=True)
    gt: dict[str, list[str]] = {}
    pages = {i: _add_target(render_page(100 + i)) for i in range(1, 7)}

    def name(i):
        return f"vlmbatch_p{i:03d}.png"

    # 1, 2: clean
    for i in (1, 2):
        save(pages[i], out / name(i)); gt[name(i)] = []

    # 3: a gloved hand holding the page down, inset so no edge is touched
    g = pages[3].copy(); d = ImageDraw.Draw(g)
    d.rounded_rectangle((120, H - 430, 300, H - 130), radius=60, fill=155, outline=95, width=4)
    for k in range(3):
        d.rounded_rectangle((150 + k * 48, H - 500, 186 + k * 48, H - 380), radius=18, fill=155, outline=95, width=3)
    save(g, out / name(3)); gt[name(3)] = ["hand_or_glove"]

    # 4: page bound in upside down
    save(pages[4].rotate(180), out / name(4)); gt[name(4)] = ["wrong_orientation"]

    # 5: the batch's colour target is missing from this frame
    save(render_page(105), out / name(5)); gt[name(5)] = ["target_region"]  # caught by the batch check, not the VLM

    # 6: a pen left lying across the page
    g = pages[6].copy(); d = ImageDraw.Draw(g)
    d.line((260, 380, 980, 690), fill=45, width=26)
    d.polygon([(980, 690), (1030, 716), (975, 716)], fill=25)
    save(g, out / name(6)); gt[name(6)] = ["foreign_object"]

    (out / "_ground_truth.json").write_text(json.dumps({"missing": [], "files": gt}, indent=2))
    return gt
