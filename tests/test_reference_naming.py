"""Archive reference codes, photographs, and comparing frames within a group.

Why this exists: a batch of scanned photographs named by archive reference
code ("<prefix> <n>-<n>-<group>-<item> <title> <year>.tif") showed three
wrong assumptions in the tool:

- the trailing-number rule reads the year closing each title as the sequence
  number, so it invents gaps and repeats and misses the frame that is absent;
- the margin check is a page check (ink must not touch the frame edge); a
  photograph fills its frame by design, so it fails every file;
- blur was judged against one median for the whole folder, so a group shot on
  a softer medium reads as blurred next to a group on a sharper one.

Names and images here are invented.

Written red-first: every test below fails on eae3bdd.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qc.agent import run_qc  # noqa: E402


def _photo(seed: int, soft: float = 0.0, size=(480, 320)) -> Image.Image:
    """A photograph-like frame: mid-tone gradient plus dark detail up to every edge."""
    rng = np.random.default_rng(seed)
    w, h = size
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    base = 150 + 40 * np.sin(x / w * 3.1 + seed) * np.cos(y / h * 2.3)
    detail = rng.normal(0, 28, (h, w)).astype(np.float32)
    detail = cv2.GaussianBlur(detail, (0, 0), 1.2)
    img = base + detail
    for _ in range(40):  # dark shapes, some cut off by the frame edge
        cx, cy = int(rng.integers(-20, w + 20)), int(rng.integers(-20, h + 20))
        cv2.circle(img, (cx, cy), int(rng.integers(8, 30)), float(rng.integers(30, 70)), -1)
    if soft:
        img = cv2.GaussianBlur(img, (0, 0), soft)
    img = np.clip(img, 0, 240).astype(np.uint8)
    return Image.fromarray(np.stack([img] * 3, axis=-1), "RGB")


def _save(d: Path, name: str, im: Image.Image):
    im.save(d / name, dpi=(600, 600))


@pytest.fixture()
def jobs(tmp_path):
    """Two jobs. Job 7 lacks item 3. Job 9 is uniformly softer (another medium).
    Item 7-5 alone is badly out of focus against its own job."""
    for item in (1, 2, 4, 5, 6):
        _save(tmp_path, f"AB 12-3-7-{item} Street scene 1901.tif",
              _photo(item, soft=6.0 if item == 5 else 0.0))
    for item in (1, 2, 3):
        _save(tmp_path, f"AB 12-3-9-{item} Harbour view 1902.tif", _photo(100 + item, soft=2.0))
    return tmp_path


def _rows(s):
    return {f["file"]: f for f in s["files"]}


def _status(row, check):
    return {c["check"]: c["status"] for c in row["checks"]}.get(check, "pass")


def test_reference_code_gives_group_and_item_not_the_year(jobs):
    s = run_qc(jobs, out_dir=jobs / "_qc", expected_dpi=600, naming="ref")
    assert s["missing_sequence"] == ["AB 12-3-7-3"], s["missing_sequence"]
    for name, row in _rows(s).items():
        assert _status(row, "sequence") == "pass", f"{name}: {row['checks']}"
    assert s["groups"] == {"AB 12-3-7": 5, "AB 12-3-9": 3}


def test_blur_is_judged_within_the_job(jobs):
    s = run_qc(jobs, out_dir=jobs / "_qc", expected_dpi=600, naming="ref")
    rows = _rows(s)
    # the soft medium is not a focus fault ...
    for item in (1, 2, 3):
        assert _status(rows[f"AB 12-3-9-{item} Harbour view 1902.tif"], "blur") == "pass"
    # ... but a frame far softer than its own job still is
    assert _status(rows["AB 12-3-7-5 Street scene 1901.tif"], "blur") == "fail"


def test_photo_mode_does_not_apply_the_page_margin_check(jobs):
    s = run_qc(jobs, out_dir=jobs / "_qc", expected_dpi=600, naming="ref", material="photo")
    for name, row in _rows(s).items():
        m = next(c for c in row["checks"] if c["check"] == "margins")
        # not applied is not the same as passed
        assert m["status"] == "skip", (name, m)
    flags = [json.loads(l) for l in (jobs / "_qc" / "flags.jsonl").read_text().splitlines()]
    assert not [f for f in flags if f["check"] == "margins"]
    assert s["not_applied"].get("margins") == 8
    assert "margins" in (jobs / "_qc" / "report.md").read_text().split("## Reshoot list")[0]


def test_page_mode_still_fails_content_at_the_frame_edge(jobs):
    s = run_qc(jobs, out_dir=jobs / "_qc", expected_dpi=600, naming="ref")
    for name, row in _rows(s).items():
        assert _status(row, "margins") == "fail", name


def test_default_naming_is_unchanged(tmp_path):
    for i in (1, 2, 4):
        _save(tmp_path, f"batch01_p{i:03d}.tif", _photo(i))
    s = run_qc(tmp_path, out_dir=tmp_path / "_qc", expected_dpi=600)
    assert s["missing_sequence"] == [3]
    assert s["naming"] == "trailing"
