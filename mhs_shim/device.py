"""MHS-shaped device abstraction.

Shape inferred from the public MHS announcement (no spec is public):
  * devices expose simple primitives: read(key) / write(key, value)
  * natural-language tags describing the device are turned into a reference
    document an agent can read before acting
  * three control surfaces: MCP, CLI, code API (see mcp_server.py / cli.py)

Added here, because heritage needs it: an optional ObjectSafetyManifest that
the driver consults before any write that spends the object's light budget.
"""
from __future__ import annotations

from typing import Any, Optional

from .manifest import ObjectSafetyManifest


class Device:
    #: subclasses fill these: key -> one-line description
    READS: dict[str, str] = {}
    WRITES: dict[str, str] = {}

    def __init__(self, name: str, kind: str, tags: dict[str, str],
                 manifest: Optional[ObjectSafetyManifest] = None):
        self.name = name
        self.kind = kind
        self.tags = dict(tags)
        self.manifest = manifest

    # ---- documentation ---------------------------------------------------
    def reference_doc(self) -> str:
        lines = [f"# Device: {self.name}", f"kind: {self.kind}", ""]
        lines.append("## Characteristics (from natural-language tags)")
        for k, v in self.tags.items():
            lines.append(f"- **{k}**: {v}")
        lines += ["", "## Readable keys"]
        lines += [f"- `read('{k}')` — {v}" for k, v in self.READS.items()]
        lines += ["", "## Writable keys"]
        lines += [f"- `write('{k}', value)` — {v}" for k, v in self.WRITES.items()]
        if self.manifest is not None:
            m = self.manifest
            lines += ["", "## Object safety manifest (enforced by driver)",
                      f"- object: {m.object_id}  sensitivity: {m.sensitivity}",
                      f"- light budget: {m.lux_hours_used:.4f} / {m.lux_hours_budget:.4f} lux·h used",
                      f"- UV allowed: {m.uv_allowed}  contact allowed: {m.contact_allowed}"]
        return "\n".join(lines) + "\n"

    # ---- primitives ------------------------------------------------------
    def read(self, key: str, **kw: Any) -> Any:
        if key not in self.READS:
            raise KeyError(f"{self.name}: '{key}' is not readable; readable = {list(self.READS)}")
        return getattr(self, f"read_{key}")(**kw)

    def write(self, key: str, value: Any = None, **kw: Any) -> Any:
        if key not in self.WRITES:
            raise KeyError(f"{self.name}: '{key}' is not writable; writable = {list(self.WRITES)}")
        return getattr(self, f"write_{key}")(value, **kw)

    # ---- safety hook for subclasses --------------------------------------
    def _charge(self, lux_hours: float, reason: str) -> None:
        """Spend object light budget; raises BudgetExceeded if not permitted."""
        if self.manifest is None:
            raise RuntimeError(f"{self.name}: refusing to expose an object with no safety manifest")
        self.manifest.charge(lux_hours, reason, self.name)
