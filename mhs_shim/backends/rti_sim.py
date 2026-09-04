"""RTI dome backend (path B), replaying a stack whose light directions are known.

The dome is simulated, but the thing the planner reasons about is not invented:
each light index carries the unit vector that actually produced that frame, so a
selection policy is choosing among real geometry. What stays simulated is the
hardware and the object, which is exactly the part a conservator has to supply.

Every exposure spends the object's light budget through the manifest, and the
driver refuses once it is gone. Refusal is a mechanism here, not a convention.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from ..device import Device
from ..manifest import ObjectSafetyManifest


class RTISimulator(Device):
    READS = {
        "light_positions": "number of LED positions on the dome",
        "light_dirs": "unit vectors of every LED, as an (n,3) array",
        "capture": "frame at the current light position, float in [0,1], or None",
        "budget": "remaining object light budget in lux-hours",
    }
    WRITES = {
        "light": "select LED index and expose; spends lux-h = lux_at_object * exposure_s / 3600",
    }

    def __init__(self, manifest: Optional[ObjectSafetyManifest], n_lights: int = 48,
                 lux_at_object: float = 2000.0, exposure_s: float = 0.5,
                 stack_dir: str | Path | None = None):
        self.stack_dir = Path(stack_dir) if stack_dir else None
        self.light_dirs = self._load_dirs(n_lights)
        self.n_lights = len(self.light_dirs)
        self.lux_at_object = lux_at_object
        self.exposure_s = exposure_s
        self.current: Optional[int] = None
        self._cache: dict[int, np.ndarray] = {}
        tags = {
            "what it is": f"A {self.n_lights}-LED RTI dome. One capture = one LED on for "
                          f"{exposure_s}s at ~{lux_at_object:.0f} lux at the object plane.",
            "per-capture dose": f"{lux_at_object * exposure_s / 3600:.4f} lux-hours",
            "light geometry": ("replayed from the stack's recorded directions"
                               if self.stack_dir else "synthetic dome rings"),
            "safety": ("The driver charges the object's manifest before each exposure and "
                       "refuses when the budget is spent. There are no UV LEDs."),
        }
        super().__init__(name="rti-dome-sim", kind="rti-dome", tags=tags, manifest=manifest)

    # ---- geometry --------------------------------------------------------
    def _load_dirs(self, n_lights: int) -> np.ndarray:
        if self.stack_dir and (self.stack_dir / "_lights.npy").exists():
            return np.load(self.stack_dir / "_lights.npy").astype(np.float32)
        el = np.radians([25.0, 45.0, 65.0])
        per = max(1, n_lights // len(el))
        out = [[np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)]
               for e in el for a in 2 * np.pi * np.arange(per) / per]
        return np.asarray(out[:n_lights], np.float32)

    @property
    def dose_per_capture(self) -> float:
        return self.lux_at_object * self.exposure_s / 3600.0

    # ---- reads -----------------------------------------------------------
    def read_light_positions(self):
        return self.n_lights

    def read_light_dirs(self):
        return self.light_dirs

    def read_budget(self):
        return None if self.manifest is None else self.manifest.remaining

    def read_capture(self):
        return None if self.current is None else self.frame(self.current)

    def frame(self, i: int) -> Optional[np.ndarray]:
        """Frame for light i as float [0,1]; None when no stack is mounted."""
        if self.stack_dir is None:
            return None
        if i not in self._cache:
            p = self.stack_dir / f"light_{i:02d}.png"
            if not p.exists():
                return None
            self._cache[i] = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        return self._cache[i]

    # ---- writes ----------------------------------------------------------
    def write_light(self, value: int):
        idx = int(value)
        if not 0 <= idx < self.n_lights:
            raise ValueError(f"light index {idx} out of range 0..{self.n_lights - 1}")
        self._charge(self.dose_per_capture, reason=f"expose light {idx}")
        self.current = idx
        return {"light": idx, "dose": self.dose_per_capture, "remaining": self.manifest.remaining}

    @staticmethod
    def stack_meta(stack_dir: str | Path) -> dict:
        p = Path(stack_dir) / "_meta.json"
        return json.loads(p.read_text()) if p.exists() else {}
