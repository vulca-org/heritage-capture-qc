"""Object-side safety manifest.

MHS (per the public announcement) stores *device* limits: weight, safety
limits, adjustable parameters.  In heritage work the safety-critical thing is
the *object*: each object carries a light budget (lux·hours), a UV rule and a
contact rule that a conservator grants for a session.  This module makes that
budget machine-readable and lets a device driver refuse a write that would
exceed it.  Enforcement lives in the driver, not in a prompt.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml


class BudgetExceeded(RuntimeError):
    """Raised when a write would exceed the object's remaining budget."""


@dataclass
class ObjectSafetyManifest:
    object_id: str
    sensitivity: str  # 'high' | 'medium' | 'low' (CIE 157 / PAS 198 style class)
    lux_hours_budget: float  # granted for this session by the responsible conservator
    lux_hours_used: float = 0.0
    uv_allowed: bool = False
    contact_allowed: bool = False
    granted_by: str = ""
    notes: str = ""
    ledger: list = field(default_factory=list)

    # ---- persistence -----------------------------------------------------
    @classmethod
    def load(cls, path: str | Path) -> "ObjectSafetyManifest":
        data = yaml.safe_load(Path(path).read_text())
        return cls(**data)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(yaml.safe_dump(asdict(self), sort_keys=False, allow_unicode=True))

    # ---- budget ----------------------------------------------------------
    @property
    def remaining(self) -> float:
        return self.lux_hours_budget - self.lux_hours_used

    def charge(self, lux_hours: float, reason: str, device: str) -> None:
        entry = {
            "t": time.time(),
            "device": device,
            "reason": reason,
            "lux_hours": lux_hours,
            "remaining_before": self.remaining,
        }
        if lux_hours > self.remaining + 1e-12:
            entry["result"] = "REFUSED"
            self.ledger.append(entry)
            raise BudgetExceeded(
                f"{device}: {reason} needs {lux_hours:.4f} lux·h, "
                f"object {self.object_id} has {self.remaining:.4f} lux·h left"
            )
        self.lux_hours_used += lux_hours
        entry["result"] = "OK"
        self.ledger.append(entry)

    def require_uv(self, device: str, reason: str) -> None:
        if not self.uv_allowed:
            self.ledger.append({"t": time.time(), "device": device, "reason": reason, "result": "REFUSED_UV"})
            raise BudgetExceeded(f"{device}: UV not permitted on object {self.object_id}")

    def ledger_json(self) -> str:
        return json.dumps(self.ledger, indent=2)
