"""MCP surface (MHS interface #1): exposes device primitives and the QC agent.

Run:  python mcp_server.py   (stdio transport; point Claude Desktop/Code at it)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mcp.server.fastmcp import FastMCP

from mhs_shim.backends.folder_scanner import VirtualScanner
from qc.agent import run_qc
from qc.vlm import make_backend

mcp = FastMCP("mhs-humanities-spike")


@mcp.tool()
def describe_scanner(folder: str, expected_dpi: int = 400) -> str:
    """Return the natural-language reference document for the capture folder device."""
    return VirtualScanner(folder, expected_dpi=expected_dpi).reference_doc()


@mcp.tool()
def read_batch_manifest(folder: str) -> list:
    """List image files in the capture folder with sequence numbers."""
    return VirtualScanner(folder).read("batch_manifest")


@mcp.tool()
def run_batch_qc(folder: str, expected_dpi: int = 400,
                 vlm: str = "none", vlm_mode: str = "flagged") -> dict:
    """Run the QC agent over a capture folder.

    Rule checks always run. `vlm` optionally adds a visual triage pass:
    "none", "anthropic[:model]", or "gemini". `vlm_mode` bounds what is sent:
    "flagged" (explain what the rules caught), "all" (also audit accepted
    frames, which is where a rule blind spot hides), or "sample:N".
    The VLM may add a fault; it never clears one.
    """
    backend = make_backend(vlm)
    s = run_qc(folder, expected_dpi=expected_dpi, vlm=backend,
               vlm_mode=vlm_mode if backend else "none")
    return {k: s[k] for k in ("n_files", "actions", "reshoot", "missing_sequence", "vlm", "out_dir")}


if __name__ == "__main__":
    mcp.run()
