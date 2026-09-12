"""An unreadable file must not kill the batch, and must never read as accepted.

Why this exists: the tool is about to be pointed at a real archive folder for the
first time. A real folder contains strays — a Thumbs.db, a zero-byte file, a PDF
among the TIFFs, a truncated transfer. On the code as committed, the first such
file raises PIL.UnidentifiedImageError out of run_qc, so 200 good captures
produce no report at all and the operator sees a Python traceback rather than
"file 0001 could not be read".

The invariant is the same one the case scripts now hold: a file the tool could not
read must be visible as such, never silently dropped and never folded into a pass.

Written red-first: all four fail on the code as committed.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fixtures.make_fixtures import make  # noqa: E402
from qc.agent import run_qc  # noqa: E402


def _with_strays(tmp_path):
    """A normal fixture batch plus three files no image reader can open."""
    gt = make(tmp_path)
    (tmp_path / "zz_truncated.tif").write_bytes(b"II*\x00 not really a tiff")
    (tmp_path / "zz_empty.tif").write_bytes(b"")
    (tmp_path / "zz_notes.jpg").write_text("a stray text file someone renamed")
    return gt


def test_one_unreadable_file_does_not_kill_the_batch(tmp_path):
    gt = _with_strays(tmp_path)
    s = run_qc(tmp_path, out_dir=tmp_path / "_qc", expected_dpi=400)

    # the readable captures were still measured
    readable = [f for f in s["files"] if f["action"] != "unreadable"]
    assert len(readable) == len(gt), (len(readable), len(gt))
    assert s["n_unreadable"] == 3, s.get("n_unreadable")


def test_unreadable_files_appear_in_the_per_file_table_with_a_reason(tmp_path):
    _with_strays(tmp_path)
    s = run_qc(tmp_path, out_dir=tmp_path / "_qc", expected_dpi=400)

    rows = {f["file"]: f for f in s["files"]}
    for name in ("zz_truncated.tif", "zz_empty.tif", "zz_notes.jpg"):
        assert name in rows, f"{name} vanished from the report entirely"
        assert rows[name]["action"] == "unreadable", rows[name]["action"]
        assert rows[name].get("reason"), f"{name} has no reason recorded"


def test_an_unreadable_file_is_never_accepted(tmp_path):
    _with_strays(tmp_path)
    s = run_qc(tmp_path, out_dir=tmp_path / "_qc", expected_dpi=400)

    for f in s["files"]:
        if f["action"] == "unreadable":
            continue
        assert f["file"] not in ("zz_truncated.tif", "zz_empty.tif", "zz_notes.jpg")
    assert "unreadable" in s["actions"] and s["actions"]["unreadable"] == 3, s["actions"]
    # read coverage must be stated, so a mostly-unreadable folder cannot look complete
    assert s["read_coverage"]["attempted"] == s["n_files"] + 3
    assert s["read_coverage"]["readable"] == s["n_files"]


def test_a_batch_with_nothing_readable_refuses(tmp_path):
    for i in range(3):
        (tmp_path / f"scan_{i:04d}.tif").write_text("not an image")
    from qc.agent import CannotRun
    with pytest.raises(CannotRun):
        run_qc(tmp_path, out_dir=tmp_path / "_qc", expected_dpi=400)
