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

    def __init__(self, folder: str | Path, expected_dpi: int = 400, out_dir: str | Path | None = None):
        self.folder = Path(folder)
        self.expected_dpi = expected_dpi
        self.out_dir = Path(out_dir) if out_dir else self.folder / "_qc"
        self.out_dir.mkdir(parents=True, exist_ok=True)
        tags = {
            "what it is": "A capture folder that stands in for a book scanner. Images appear as files; "
                          "nothing here moves, so it cannot harm an object.",
            "naming": "Files carry a trailing sequence number before the extension, e.g. batch01_p007.tif.",
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
            m = SEQ_RE.search(p.name)
            out.append({"path": str(p), "name": p.name, "seq": int(m.group(1)) if m else None,
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
