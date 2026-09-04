"""VLM triage: the layer that sees what the rule checks cannot.

Division of labour, deliberately strict:

  * Rule checks own everything measurable — resolution, format, sharpness,
    exposure, skew, edge crop, duplicates, sequence.  They are deterministic
    and cheap, so they run on every file.
  * The VLM owns everything that needs looking at the picture — a hand or a
    weight in frame, a page bound in upside down, a missing colour target, a
    pen left on the plate, a leaf that is not a page at all.
  * **The VLM may add a problem; it may never clear one.**  A rule failure is
    a measurement, and a model's opinion does not overturn a measurement.
    This is what keeps the layer from quietly lowering the standard.

Cost is controlled in three places: only the selected files are sent, each is
downscaled before encoding, and the reply is constrained to a small schema so
output tokens stay near-constant.
"""
from __future__ import annotations

import base64
import io
import json
import os
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Protocol

from PIL import Image

# Issues the model is allowed to report.  A closed vocabulary keeps the
# verdicts comparable across a batch and keeps the schema strict.
ISSUE_VOCAB = [
    "hand_or_glove",          # operator's hand, finger or glove in frame
    "foreign_object",         # pen, ruler, weight, snake, clip left on the page
    "wrong_orientation",      # page upside down or on its side
    "obscured_content",       # text or image hidden by shadow, curl, or an overlay
    "not_a_page",             # separator card, calibration shot, empty platen
    "object_at_risk",         # object handled in a way that could damage it
]

SYSTEM = """You inspect single frames from a library or archive digitisation batch.

Report ONLY what you can actually see in the frame. Do not guess at intent and
do not describe the document's contents.

Do NOT report sharpness, exposure, contrast, resolution, colour cast, rotation
of less than about ten degrees, or file format. Those are measured separately
by instruments that are better at it than you are, and duplicating them creates
disagreement for no gain.

Do NOT report a missing colour or scale target. You see one frame at a time and
cannot know what the rest of the batch contains; that comparison is made for you.

DO report, using only the allowed issue names:
- hand_or_glove: a hand, finger or glove is visible in the frame
- foreign_object: something that is not part of the document rests on or over it
- wrong_orientation: the page is upside down or rotated onto its side
- obscured_content: content is hidden by shadow, page curl, or something over it
- not_a_page: this is a separator card, a calibration shot, or an empty platen
- object_at_risk: the object appears held, bent or weighted in a way that risks damage

Severity is "fail" when the frame should be captured again, "warn" when a human
should look before deciding. If you see nothing from the list, return an empty
issues array. An empty array is the expected answer for most frames."""

SCHEMA = {
    "type": "object",
    "properties": {
        "issues": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "issue": {"type": "string", "enum": ISSUE_VOCAB},
                    "severity": {"type": "string", "enum": ["fail", "warn"]},
                    "evidence": {"type": "string", "description": "what in the frame shows this, in one clause"},
                },
                "required": ["issue", "severity", "evidence"],
                "additionalProperties": False,
            },
        },
        "note": {"type": "string", "description": "one short sentence, or empty"},
    },
    "required": ["issues", "note"],
    "additionalProperties": False,
}


@dataclass
class VLMVerdict:
    file: str
    issues: list[dict] = field(default_factory=list)
    note: str = ""
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    error: str = ""

    @property
    def worst(self) -> str | None:
        if any(i["severity"] == "fail" for i in self.issues):
            return "fail"
        if self.issues:
            return "warn"
        return None

    def to_dict(self):
        return asdict(self)


class VLMTriage(Protocol):
    def judge(self, image_path: str, batch_context: str) -> VLMVerdict: ...


# ---------------------------------------------------------------------------
# image preparation (cost control)
# ---------------------------------------------------------------------------

def encode_for_vlm(path: str | Path, max_side: int = 1024, quality: int = 80) -> tuple[str, str, tuple[int, int]]:
    """Downscale and JPEG-encode. Returns (base64, media_type, (w, h)).

    A hand in frame or an upside-down page is legible at 1024px; sending a
    400 ppi master would multiply the token cost for no extra signal.
    """
    with Image.open(path) as im:
        im = im.convert("RGB")
        w, h = im.size
        if max(w, h) > max_side:
            s = max_side / max(w, h)
            im = im.resize((max(1, int(w * s)), max(1, int(h * s))), Image.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=quality)
        return base64.standard_b64encode(buf.getvalue()).decode(), "image/jpeg", im.size


def estimate_image_tokens(size: tuple[int, int]) -> int:
    w, h = size
    return int(w * h / 750)


# ---------------------------------------------------------------------------
# backends
# ---------------------------------------------------------------------------

class AnthropicVLM:
    """Claude vision backend (primary).

    Written against the documented Messages API shape: base64 image block plus
    a text block, with the reply constrained by output_config.format.
    """

    def __init__(self, model: str = "claude-opus-5", max_side: int = 1024, max_tokens: int = 1024):
        import anthropic  # imported lazily so the rest of the tool runs without it
        self.client = anthropic.Anthropic()
        self.model = model
        self.max_side = max_side
        self.max_tokens = max_tokens

    def judge(self, image_path: str, batch_context: str) -> VLMVerdict:
        data, media_type, size = encode_for_vlm(image_path, self.max_side)
        try:
            resp = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=SYSTEM,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": data}},
                        {"type": "text", "text": batch_context},
                    ],
                }],
                output_config={"format": {"type": "json_schema", "schema": SCHEMA}},
            )
            if resp.stop_reason == "refusal":
                return VLMVerdict(file=Path(image_path).name, model=self.model, error="refusal")
            text = next(b.text for b in resp.content if b.type == "text")
            parsed = json.loads(text)
            return VLMVerdict(
                file=Path(image_path).name,
                issues=parsed.get("issues", []),
                note=parsed.get("note", ""),
                model=self.model,
                input_tokens=resp.usage.input_tokens,
                output_tokens=resp.usage.output_tokens,
            )
        except Exception as e:  # surfaced per file; a bad frame must not stop the batch
            return VLMVerdict(file=Path(image_path).name, model=self.model, error=f"{type(e).__name__}: {e}")


class StubVLM:
    """Deterministic backend for tests: filename -> issues."""

    def __init__(self, table: dict[str, list[dict]] | None = None):
        self.table = table or {}
        self.calls: list[str] = []

    def judge(self, image_path: str, batch_context: str) -> VLMVerdict:
        name = Path(image_path).name
        self.calls.append(name)
        return VLMVerdict(file=name, issues=list(self.table.get(name, [])), model="stub")


# ---------------------------------------------------------------------------
# selection policy
# ---------------------------------------------------------------------------

def select_for_vlm(report: list[dict], mode: str, seed: int = 0) -> list[str]:
    """Which files to send. `mode` is none | flagged | all | sample:N.

    'flagged' buys a reason for a decision the rules already made.
    'all' is what actually catches the defects rules are blind to.
    'sample:N' audits N of the pages the rules accepted, which is where a
    blind spot hides.
    """
    if mode in ("none", "", None):
        return []
    if mode == "all":
        return [r["file"] for r in report]
    if mode == "flagged":
        return [r["file"] for r in report if r["action"] != "accept"]
    if mode.startswith("sample:"):
        n = int(mode.split(":", 1)[1])
        flagged = [r["file"] for r in report if r["action"] != "accept"]
        accepted = [r["file"] for r in report if r["action"] == "accept"]
        rng = random.Random(seed)
        rng.shuffle(accepted)
        return flagged + accepted[:n]
    raise ValueError(f"unknown vlm mode: {mode!r}")


def make_backend(spec: str | None):
    """'anthropic[:model]' | 'gemini' | 'stub' | None."""
    if not spec or spec == "none":
        return None
    if spec.startswith("anthropic"):
        _, _, model = spec.partition(":")
        return AnthropicVLM(model=model or "claude-opus-5")
    if spec == "gemini":
        from .vlm_gemini import GeminiAskVLM
        return GeminiAskVLM()
    if spec == "stub":
        return StubVLM()
    raise ValueError(f"unknown vlm backend: {spec!r}")
