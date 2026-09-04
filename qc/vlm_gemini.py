"""Second VLM backend routed through the local `gemini-ask` channel.

Present for one reason: this machine has a Gemini key in the Keychain and no
Anthropic credential, so this is the backend that can actually be *run* today.
The primary implementation is AnthropicVLM in vlm.py; both satisfy the same
VLMTriage protocol, and the agent does not know which one it has.

Everything sent this way leaves the machine. Only send frames you are allowed
to send — the synthetic fixtures qualify, a reader's archival material does not
without the holding institution's permission.
"""
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

USAGE_RE = re.compile(r"in=(\d+)\s+out=(\d+)")

from .vlm import SYSTEM, ISSUE_VOCAB, VLMVerdict

GEMINI_ASK = Path.home() / ".gemini-worker" / "gemini-ask"
JSON_RE = re.compile(r"\{.*\}", re.S)

INSTRUCTION = (
    SYSTEM
    + "\n\nReply with JSON only, no prose and no code fence:\n"
      '{"issues": [{"issue": "<one of: ' + " | ".join(ISSUE_VOCAB) + '>", '
      '"severity": "fail|warn", "evidence": "<short clause>"}], "note": "<one sentence or empty>"}'
)


class GeminiAskVLM:
    def __init__(self, binary: str | Path = GEMINI_ASK, timeout: int = 120):
        self.binary = str(binary)
        self.timeout = timeout

    def judge(self, image_path: str, batch_context: str) -> VLMVerdict:
        name = Path(image_path).name
        try:
            p = subprocess.run(
                [self.binary, "--image", str(image_path), f"{INSTRUCTION}\n\nBatch context: {batch_context}"],
                capture_output=True, text=True, timeout=self.timeout,
                # gemini-ask also accepts a question on stdin. Inheriting an
                # open pipe (any non-interactive parent) makes it wait there
                # forever, which looks exactly like a slow model and is not.
                stdin=subprocess.DEVNULL,
            )
        except Exception as e:
            return VLMVerdict(file=name, model="gemini-ask", error=f"{type(e).__name__}: {e}")
        if p.returncode != 0:
            return VLMVerdict(file=name, model="gemini-ask", error=f"exit {p.returncode}: {p.stderr.strip()[:200]}")
        m = JSON_RE.search(p.stdout)
        if not m:
            return VLMVerdict(file=name, model="gemini-ask", error=f"no JSON in reply: {p.stdout.strip()[:160]}")
        try:
            d = json.loads(m.group(0))
        except json.JSONDecodeError as e:
            return VLMVerdict(file=name, model="gemini-ask", error=f"bad JSON: {e}")
        issues = [i for i in d.get("issues", []) if i.get("issue") in ISSUE_VOCAB]
        u = USAGE_RE.search(p.stderr or "")
        return VLMVerdict(file=name, issues=issues, note=d.get("note", ""), model="gemini-ask",
                          input_tokens=int(u.group(1)) if u else 0,
                          output_tokens=int(u.group(2)) if u else 0)
