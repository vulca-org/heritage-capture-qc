"""QC agent for path C: batch -> checks -> triage -> reshoot list + report.

No LLM in the loop yet.  The agent talks to the scanner only through the
device's read/write primitives and takes its expected resolution from the
device reference, so it ports to a real MHS-driven scanner by swapping the
backend.  A VLM triage hook is the obvious next layer; it is not wired.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

from mhs_shim.backends.folder_scanner import VirtualScanner
from . import checks as C
from .vlm import VLMVerdict, select_for_vlm

RESHOOT = {"blur", "glare", "skew", "margins", "exposure", "duplicate", "target_region"}
REPROCESS = {"format", "dpi", "bit_depth"}

#: how far each action is from "fine"; escalation only ever moves up this scale
RANK = {"accept": 0, "review": 1, "reprocess": 2, "reshoot": 3}


def escalate(action: str, worst: str | None) -> str:
    """Apply a VLM verdict to a rule-derived action.

    Upward only. A model that sees a hand in frame can send a page back; a
    model that sees nothing cannot rescue a page whose resolution is wrong.
    """
    if worst is None:
        return action
    want = "reshoot" if worst == "fail" else "review"
    return action if RANK[action] >= RANK[want] else want


def triage(results: list[C.CheckResult]) -> str:
    fails = {r.check for r in results if r.status == C.FAIL}
    warns = {r.check for r in results if r.status == C.WARN}
    if fails & RESHOOT:
        return "reshoot"
    if fails & REPROCESS:
        return "reprocess"
    if warns:
        return "review"
    return "accept"


def run_qc(batch_dir: str | Path, out_dir: str | Path | None = None, expected_dpi: int | None = None,
           vlm=None, vlm_mode: str = "none", batch_context: str = "", access_dpi: int | None = None) -> dict:
    dev = VirtualScanner(batch_dir, expected_dpi=expected_dpi or 400, out_dir=out_dir)
    # The agent reads the device's own description before acting.
    doc = dev.reference_doc()
    dpi = dev.read("expected_dpi")
    manifest = dev.read("batch_manifest")

    # pass 1: per-file measurements
    per_file = []
    for item in manifest:
        img = dev.read("image", path=item["path"])
        gray = C.to_gray(img)
        is_access = access_dpi is not None and (
            (img.format or "").upper() == "JPEG" or Path(item["name"]).suffix.lower() in (".jpg", ".jpeg"))
        per_file.append({
            "item": item, "img": img, "gray": gray,
            "role": "access" if is_access else "master",
            "sharp": C.sharpness(gray),
            "sha": C.file_sha256(item["path"]),
            "vec": C.thumb_vector(gray),
            "region": C.region_vector(gray),
        })

    # Batch-relative statistics are computed within a role. An access copy is,
    # by construction, a near-duplicate of its master and softer than it; only
    # its siblings are a fair reference.
    groups: dict[str, list[int]] = {}
    for i, f in enumerate(per_file):
        groups.setdefault(f["role"], []).append(i)
    med_sharp_by_role, dup_exact, dup_near, region_results = {}, {}, {}, {}
    missing_by_role, dupseq_by_role = {}, {}
    for role, idx in groups.items():
        med_sharp_by_role[role] = float(np.median([per_file[i]["sharp"] for i in idx]))
        seen_sha = {}
        for pos, i in enumerate(idx):
            f = per_file[i]
            if f["sha"] in seen_sha:
                dup_exact[i] = seen_sha[f["sha"]]
            else:
                seen_sha[f["sha"]] = i
                for j in idx[:pos]:
                    if j in dup_exact: continue
                    if float(np.dot(f["vec"], per_file[j]["vec"])) >= 0.998:
                        dup_near[i] = j
                        break
        missing_by_role[role], dupseq_by_role[role] = C.check_sequence([per_file[i]["item"]["seq"] for i in idx])
        for i, r in zip(idx, C.batch_region_consistency([per_file[i]["region"] for i in idx])):
            region_results[i] = r
    med_sharp = med_sharp_by_role.get("master", 0.0)
    missing = missing_by_role.get("master", [])

    # master <-> access pairing, by file stem (photo_0007.tif <-> photo_0007.jpg)
    stems_by_role = {role: {Path(per_file[i]["item"]["name"]).stem.lower() for i in idx}
                     for role, idx in groups.items()}

    # pass 2: checks + triage
    report, reshoot = [], []
    for i, f in enumerate(per_file):
        img, gray, item, role = f["img"], f["gray"], f["item"], f["role"]
        res = [
            C.check_format(img, role),
            C.check_dpi(img, access_dpi if role == "access" else dpi),
            C.check_bit_depth(img),
            C.check_blur(f["sharp"], med_sharp_by_role[role]),
            C.check_glare(gray),
            C.check_exposure(gray),
            C.check_skew(C.estimate_skew(gray)),
            C.check_margins(gray),
            region_results[i],
        ]
        if access_dpi is not None:
            other = "access" if role == "master" else "master"
            res.append(C.check_pairing(role, Path(item["name"]).stem.lower() in stems_by_role.get(other, set())))
        dup_seq = dupseq_by_role[role]
        if i in dup_exact:
            res.append(C.CheckResult("duplicate", C.FAIL, per_file[dup_exact[i]]["item"]["name"], "sha256",
                                     "byte-identical to an earlier capture"))
        elif i in dup_near:
            res.append(C.CheckResult("duplicate", C.WARN, per_file[dup_near[i]]["item"]["name"], "corr>=0.998",
                                     "near-identical to an earlier capture (double shot?)"))
        if item["seq"] in dup_seq:
            res.append(C.CheckResult("sequence", C.WARN, item["seq"], "unique", "sequence number used twice"))
        action = triage(res)
        if action == "reshoot":
            reshoot.append(item["name"])
        for r in res:
            if r.status != C.PASS:
                dev.write("flag", {"file": item["name"], **r.to_dict()})
        report.append({"file": item["name"], "seq": item["seq"], "role": role, "action": action,
                       "checks": [r.to_dict() for r in res]})

    # pass 3: VLM triage over the selected frames (adds problems, never clears one)
    vlm_records: list[dict] = []
    if vlm is not None and vlm_mode not in ("none", "", None):
        by_name = {r["file"]: r for r in report}
        path_of = {f["item"]["name"]: f["item"]["path"] for f in per_file}
        ctx = batch_context or (f"Batch of {len(per_file)} frames captured at {dpi} ppi. "
                                "Frames in this batch normally show a single page and nothing else.")
        # only masters go to the model; an access copy shows the same picture
        for fname in select_for_vlm([r for r in report if r["role"] == "master"], vlm_mode):
            v: VLMVerdict = vlm.judge(path_of[fname], ctx)
            vlm_records.append(v.to_dict())
            row = by_name[fname]
            row["vlm"] = v.to_dict()
            if v.error:
                continue
            for iss in v.issues:
                dev.write("flag", {"file": fname, "check": f"vlm:{iss['issue']}",
                                   "status": iss["severity"], "value": iss["evidence"], "threshold": "vlm"})
            before = row["action"]
            row["action"] = escalate(before, v.worst)
            if before != row["action"]:
                row["escalated_by_vlm"] = True
        reshoot = [r["file"] for r in report if r["action"] == "reshoot"]

    dev.write("reshoot_list", reshoot)
    summary = {
        "batch": str(Path(batch_dir)),
        "n_files": len(per_file),
        "expected_dpi": dpi,
        "access_dpi": access_dpi,
        "roles": dict(Counter(f["role"] for f in per_file)),
        "batch_median_sharpness": round(med_sharp, 4),
        "missing_sequence": missing,
        "missing_sequence_access": missing_by_role.get("access", []) if access_dpi is not None else None,
        "actions": dict(Counter(r["action"] for r in report)),
        "reshoot": reshoot,
        "files": report,
        "vlm": {
            "mode": vlm_mode,
            "backend": type(vlm).__name__ if vlm is not None else None,
            "n_judged": len(vlm_records),
            "n_errors": sum(1 for v in vlm_records if v.get("error")),
            "input_tokens": sum(v.get("input_tokens", 0) for v in vlm_records),
            "output_tokens": sum(v.get("output_tokens", 0) for v in vlm_records),
            "escalated": [r["file"] for r in report if r.get("escalated_by_vlm")],
        },
        "device_reference": doc,
    }
    out = dev.out_dir
    (out / "report.json").write_text(json.dumps(summary, indent=2, default=str))
    (out / "report.md").write_text(render_md(summary))
    summary["out_dir"] = str(out)
    return summary


def render_md(s: dict) -> str:
    lines = [f"# QC report — {Path(s['batch']).name}", "",
             f"files: {s['n_files']}  declared: {s['expected_dpi']} ppi"
             + (f" (access copies {s['access_dpi']} ppi, roles {s['roles']})" if s.get("access_dpi") else "")
             + f"  actions: {s['actions']}",
             f"missing sequence numbers: {s['missing_sequence'] or 'none'}",
             (f"VLM: {s['vlm']['backend']} mode={s['vlm']['mode']} judged={s['vlm']['n_judged']} "
              f"errors={s['vlm']['n_errors']} escalated={s['vlm']['escalated'] or 'none'}")
             if s.get("vlm", {}).get("backend") else "VLM: not run", "",
             "| file | seq | role | action | measured faults | seen by VLM |", "|---|---|---|---|---|---|"]
    for f in s["files"]:
        rules = ", ".join(f"{c['check']}={c['value']}" for c in f["checks"] if c["status"] != "pass")
        v = f.get("vlm")
        if not v:
            seen = "—"
        elif v.get("error"):
            seen = f"error: {v['error'][:40]}"
        elif v["issues"]:
            seen = "; ".join(f"**{i['issue']}** ({i['severity']}): {i['evidence']}" for i in v["issues"])
        else:
            seen = "nothing"
        lines.append(f"| {f['file']} | {f['seq']} | {f.get('role', 'master')} | **{f['action']}** | {rules or '—'} | {seen} |")
    lines += ["", "## Reshoot list"]
    lines += [f"- {n}" for n in s["reshoot"]] or ["- none"]
    return "\n".join(lines) + "\n"
