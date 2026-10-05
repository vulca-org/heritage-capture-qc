"""Virtual scanner: a capture folder standing in for a book/flatbed scanner.

Path C (archive digitisation) can be prototyped with zero hardware because the
scanner's *output* is a folder of images.  This backend exposes that folder
through the same read/write primitives a real MHS driver would, so the QC agent
written against it ports to a real device by swapping the backend.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

from PIL import Image

from ..device import Device

IMAGE_EXT = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}
SEQ_RE = re.compile(r"(\d+)(?=\.[A-Za-z]+$)")
# An archive reference code at the start of the name: a short letter prefix, then
# numbers joined by hyphens, e.g. "AB 12-3-7-4 Street scene 1901.tif". The last
# number is the item; everything before it names the group (a job, a box, a
# volume). Under the trailing rule the year closing such a title is read as the
# sequence number, so a reference-coded folder must be read with this rule.
REF_RE = re.compile(r"^([A-Za-z]{1,6} ?\d+(?:-\d+)+)-(\d+)(?=[\s_.]|$)")
NAMING = ("trailing", "ref")


class VirtualScanner(Device):
    READS = {
        "batch_manifest": "sorted list of image files in the capture folder with sequence numbers",
        "image": "PIL image for a given path (kw: path)",
        "expected_dpi": "the capture resolution the operator declared for this batch",
    }
    WRITES = {
        "flag": "append a QC flag {file, check, status, value, note} to the batch ledger",
        "reshoot_list": "write the list of files to recapture (value: list of file names)",
    }

    def __init__(self, folder: str | Path, expected_dpi: int = 400, out_dir: str | Path | None = None,
                 naming: str = "trailing"):
        if naming not in NAMING:
            raise ValueError(f"naming must be one of {NAMING}, not {naming!r}")
        self.folder = Path(folder)
        self.expected_dpi = expected_dpi
        self.naming = naming
        self.out_dir = Path(out_dir) if out_dir else self.folder / "_qc"
        self.out_dir.mkdir(parents=True, exist_ok=True)
        tags = {
            "what it is": "A capture folder that stands in for a book scanner. Images appear as files; "
                          "nothing here moves, so it cannot harm an object.",
            "naming": ("Files carry a trailing sequence number before the extension, e.g. batch01_p007.tif."
                       if naming == "trailing" else
                       "Files open with an archive reference code; the last number is the item and the "
                       "rest names its group, e.g. AB 12-3-7-4 Title.tif is item 4 of group AB 12-3-7."),
            "declared resolution": f"{expected_dpi} ppi",
            "safety": "No exposure is spent by reading files; no object manifest is needed.",
        }
        super().__init__(name=f"virtual-scanner:{self.folder.name}", kind="scanner-folder", tags=tags)
        self._ledger_path = self.out_dir / "flags.jsonl"

    # reads ---------------------------------------------------------------
    def read_batch_manifest(self):
        files = sorted(p for p in self.folder.iterdir() if p.suffix.lower() in IMAGE_EXT and not p.name.startswith("_"))
        out = []
        for p in files:
            group, seq = None, None
            if self.naming == "ref":
                m = REF_RE.match(p.name)
                if m:
                    group, seq = m.group(1), int(m.group(2))
            else:
                m = SEQ_RE.search(p.name)
                seq = int(m.group(1)) if m else None
            out.append({"path": str(p), "name": p.name, "seq": seq, "group": group,
                        "bytes": p.stat().st_size})
        return out

    def read_image(self, path: str):
        return Image.open(path)

    def read_expected_dpi(self):
        return self.expected_dpi

    # writes --------------------------------------------------------------
    def write_flag(self, value: dict):
        value = dict(value, t=time.time())
        with self._ledger_path.open("a") as fh:
            fh.write(json.dumps(value) + "\n")
        return str(self._ledger_path)

    def write_reshoot_list(self, value: list):
        p = self.out_dir / "reshoot.txt"
        p.write_text("\n".join(value) + ("\n" if value else ""))
        return str(p)
