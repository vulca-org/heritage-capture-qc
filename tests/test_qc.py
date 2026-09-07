import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fixtures.make_fixtures import make, make_archive_fixtures  # noqa: E402
from qc.agent import run_qc  # noqa: E402
from mhs_shim.manifest import ObjectSafetyManifest, BudgetExceeded  # noqa: E402
from mhs_shim.backends.rti_sim import RTISimulator  # noqa: E402


@pytest.fixture(scope="module")
def batch(tmp_path_factory):
    d = tmp_path_factory.mktemp("batch")
    gt = make(d)
    summary = run_qc(d, out_dir=d / "_qc", expected_dpi=400)
    return gt, summary


def _status(summary, fname, check):
    f = next(x for x in summary["files"] if x["file"] == fname)
    return {c["check"]: c["status"] for c in f["checks"]}.get(check, "pass")


def test_each_injected_defect_is_caught(batch):
    gt, s = batch
    for fname, defects in gt.items():
        for d in defects:
            check, _, level = d.partition(":")
            want = level or "fail"
            assert _status(s, fname, check) == want, f"{fname}: {check} expected {want}"


def test_clean_pages_are_accepted(batch):
    gt, s = batch
    for fname, defects in gt.items():
        if not defects:
            f = next(x for x in s["files"] if x["file"] == fname)
            assert f["action"] == "accept", f"{fname} flagged: {[c for c in f['checks'] if c['status'] != 'pass']}"


def test_no_check_fires_where_not_injected(batch):
    gt, s = batch
    for fname, defects in gt.items():
        injected = {d.split(":")[0] for d in defects}
        f = next(x for x in s["files"] if x["file"] == fname)
        fired = {c["check"] for c in f["checks"] if c["status"] == "fail"}
        assert fired <= injected, f"{fname}: unexpected fails {fired - injected}"


def test_missing_sequence_and_reshoot_list(batch):
    gt, s = batch
    assert s["missing_sequence"] == [10]
    assert set(s["reshoot"]) == {"batch01_p003.png", "batch01_p005.png", "batch01_p007.png",
                                 "batch01_p008.png", "batch01_p009.png", "batch01_p012.png"}
    out = Path(s["out_dir"])
    assert (out / "report.md").exists() and (out / "flags.jsonl").exists()
    assert json.loads((out / "report.json").read_text())["n_files"] == 12


def test_rti_driver_refuses_when_object_budget_spent(tmp_path):
    m = ObjectSafetyManifest(object_id="teaching-obj-01", sensitivity="high", lux_hours_budget=0.6, granted_by="test")
    dome = RTISimulator(m, lux_at_object=2000, exposure_s=0.5)  # 0.2778 lux·h per capture
    dome.write("light", 0)
    dome.write("light", 1)
    with pytest.raises(BudgetExceeded):
        dome.write("light", 2)
    assert m.ledger[-1]["result"] == "REFUSED"
    assert sum(1 for e in m.ledger if e["result"] == "OK") == 2
    p = tmp_path / "m.yaml"; m.save(p)
    assert ObjectSafetyManifest.load(p).remaining == pytest.approx(m.remaining)


def test_device_without_manifest_cannot_expose():
    dome = RTISimulator(None)
    with pytest.raises(RuntimeError):
        dome.write("light", 0)


# ---------------------------------------------------------------------------
# VLM triage layer
#
# The batch splits three ways, and each test pins one part of the split:
#   RULE_CAUGHT — a fault the instruments can measure (here, batch-relative)
#   VLM_ONLY    — a fault only visible by looking at the frame
#   CLEAN       — neither layer may touch these
# ---------------------------------------------------------------------------
from fixtures.make_fixtures import make_vlm_fixtures  # noqa: E402
from qc.vlm import StubVLM, VLMVerdict, select_for_vlm, ISSUE_VOCAB  # noqa: E402
from qc.agent import escalate  # noqa: E402

CLEAN = {"vlmbatch_p001.png", "vlmbatch_p002.png"}
VLM_ONLY = {"vlmbatch_p003.png", "vlmbatch_p006.png"}     # gloved hand; pen left on the page
RULE_CAUGHT = {"vlmbatch_p004.png", "vlmbatch_p005.png"}  # upside down; target missing


@pytest.fixture(scope="module")
def vlm_batch(tmp_path_factory):
    d = tmp_path_factory.mktemp("vlmbatch")
    return make_vlm_fixtures(d), d


def _actions(summary):
    return {r["file"]: r["action"] for r in summary["files"]}


def test_rule_layer_catches_only_the_measurable_half(vlm_batch):
    gt, d = vlm_batch
    a = _actions(run_qc(d, out_dir=d / "_qc_rules", expected_dpi=400))
    assert {f for f, act in a.items() if act != "accept"} == RULE_CAUGHT
    for f in VLM_ONLY | CLEAN:
        assert a[f] == "accept", f"{f} should be invisible to the rule layer"


def test_region_check_abstains_when_the_batch_has_no_constant_target(batch):
    """On free-form pages the region is content, not a fixture; inventing a
    norm there would fail every frame that happens to look different."""
    _, s = batch
    fired = [f["file"] for f in s["files"]
             for c in f["checks"] if c["check"] == "target_region" and c["status"] != "pass"]
    assert fired == []


def test_vlm_catches_what_rules_miss(vlm_batch):
    gt, d = vlm_batch
    stub = StubVLM({f: [{"issue": i, "severity": "fail", "evidence": "stub"}
                        for i in iss if i in ISSUE_VOCAB] for f, iss in gt.items()})
    s = run_qc(d, out_dir=d / "_qc_vlm", expected_dpi=400, vlm=stub, vlm_mode="all")
    assert set(s["vlm"]["escalated"]) == VLM_ONLY
    assert set(s["reshoot"]) == VLM_ONLY | RULE_CAUGHT
    for f in CLEAN:
        assert _actions(s)[f] == "accept"


def test_escalation_is_upward_only():
    for action in ("accept", "review", "reprocess", "reshoot"):
        assert escalate(action, None) == action
        assert escalate(action, "fail") == "reshoot"
    assert escalate("reprocess", "warn") == "reprocess"   # never demoted to review
    assert escalate("accept", "warn") == "review"
    assert escalate("reshoot", "warn") == "reshoot"


def test_a_silent_vlm_cannot_clear_a_measured_failure(tmp_path):
    from fixtures.make_fixtures import make as make_rule_batch
    d = tmp_path / "b"
    make_rule_batch(d)
    s = run_qc(d, out_dir=d / "_qc", expected_dpi=400, vlm=StubVLM({}), vlm_mode="all")
    assert len(s["reshoot"]) == 6 and s["vlm"]["escalated"] == []


def test_selection_modes_control_what_is_sent(vlm_batch):
    gt, d = vlm_batch
    rep = run_qc(d, out_dir=d / "_qc_sel", expected_dpi=400)["files"]
    assert select_for_vlm(rep, "none") == []
    assert len(select_for_vlm(rep, "all")) == 6
    assert set(select_for_vlm(rep, "flagged")) == RULE_CAUGHT
    assert len(select_for_vlm(rep, "sample:2")) == len(RULE_CAUGHT) + 2

    stub = StubVLM({})
    run_qc(d, out_dir=d / "_qc_sel2", expected_dpi=400, vlm=stub, vlm_mode="sample:1")
    assert len(stub.calls) == len(RULE_CAUGHT) + 1


def test_a_failing_frame_does_not_stop_the_batch(vlm_batch):
    gt, d = vlm_batch

    class Flaky:
        def judge(self, image_path, batch_context):
            n = Path(image_path).name
            return VLMVerdict(file=n, error="boom") if n == "vlmbatch_p003.png" else VLMVerdict(file=n)

    s = run_qc(d, out_dir=d / "_qc_err", expected_dpi=400, vlm=Flaky(), vlm_mode="all")
    assert s["vlm"]["n_judged"] == 6 and s["vlm"]["n_errors"] == 1
    assert _actions(s)["vlmbatch_p003.png"] == "accept"   # an error is not a verdict


# ---------------------------------------------------------------------------
# Path B: adaptive capture planning under an object light budget
# ---------------------------------------------------------------------------
import ast  # noqa: E402
import inspect  # noqa: E402

import numpy as np  # noqa: E402

from fixtures.make_rti_stack import make_stack  # noqa: E402
from mhs_shim.backends.rti_sim import RTISimulator  # noqa: E402
from qc import relief as REL  # noqa: E402
from qc import capture_planner as CP  # noqa: E402
from qc.rti_experiment import run as run_rti  # noqa: E402

GENEROUS = 100.0   # lux-hours: far more than any arm needs


@pytest.fixture(scope="module")
def stack(tmp_path_factory):
    d = tmp_path_factory.mktemp("rti")
    make_stack(d, seed=7, noise=0.012)
    return d


def _dome(stack_dir, budget=GENEROUS):
    m = ObjectSafetyManifest(object_id="test-obj", sensitivity="high",
                             lux_hours_budget=budget, granted_by="test")
    return RTISimulator(m, stack_dir=stack_dir)


def test_planner_cannot_see_ground_truth():
    """Structural guard, not a promise in a comment.

    A stopping rule that can read the stroke mask would stop when the answer
    looks right, which is not a result. Strip the docstrings and no
    ground-truth identifier may remain in the planner's executable code.
    """
    tree = ast.parse(inspect.getsource(CP))
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.ClassDef)):
            if (node.body and isinstance(node.body[0], ast.Expr)
                    and isinstance(node.body[0].value, ast.Constant)
                    and isinstance(node.body[0].value.value, str)):
                node.body.pop(0)
    code = ast.unparse(tree)
    for forbidden in ("oracle", "mask", "d_prime", "stroke"):
        assert forbidden not in code, f"planner references ground truth: {forbidden}"


def test_planner_stops_early_on_its_own(stack):
    d = _dome(stack)
    r = CP.plan_capture(d, d.frame, policy="adaptive")
    assert "settled" in r.stop_reason
    assert 4 <= r.n_captures < d.n_lights
    assert r.dose_spent == pytest.approx(r.n_captures * d.dose_per_capture)


def test_budget_exhaustion_is_not_reported_as_success(stack):
    """Stopping because the object ran out of budget is a different fact from
    stopping because the reconstruction settled, and must read differently."""
    d = _dome(stack, budget=1.0)
    r = CP.plan_capture(d, d.frame, policy="adaptive")
    assert "budget" in r.stop_reason and "settled" not in r.stop_reason
    assert r.dose_spent <= 1.0
    assert d.manifest.ledger[-1]["result"] == "REFUSED"


def test_adaptive_needs_fewer_captures_than_ring_order(stack):
    seq = CP.plan_capture(_dome(stack), _dome(stack).frame, policy="sequential")
    ada = CP.plan_capture(_dome(stack), _dome(stack).frame, policy="adaptive")
    assert ada.n_captures < seq.n_captures


def test_adaptive_conditions_the_light_matrix_better_early(stack):
    """The mechanism behind the saving, checked directly rather than inferred."""
    d = _dome(stack)
    L = d.light_dirs
    seq, ada = [], []
    for _ in range(5):
        seq.append(CP.next_sequential(seq, L))
        ada.append(CP.next_adaptive(ada, L))
    assert REL.conditioning(L[ada])[0] > REL.conditioning(L[seq])[0]


def test_early_stopping_keeps_most_of_the_full_dome_legibility(stack):
    res = run_rti(stack, budget_lux_h=GENEROUS)
    a = res["arms"]
    assert a["full"]["n_captures"] == 48
    for policy in ("sequential", "adaptive"):
        assert a[policy]["d_prime_pct_of_full"] > 90
        assert a[policy]["dose_pct_of_full"] < 50


def test_the_fixture_actually_contains_a_legible_inscription(stack):
    """If the synthetic object had no readable relief, every arm above would be
    comparing noise to noise and all of them would 'pass'."""
    d = _dome(stack)
    order = list(range(d.n_lights))
    n = REL.solve_normals(d.light_dirs[order], np.stack([d.frame(i) for i in order]))
    mask = np.load(stack / "_stroke_mask.npy")
    assert REL.oracle_dprime(REL.relief_map(n), mask) > 2.0


def test_never_stops_before_the_minimum_number_of_lights(stack):
    """The invariant the removed conditioning guard was standing in for."""
    for policy in ("sequential", "adaptive"):
        for lo in (4, 6, 9):
            d = _dome(stack)
            r = CP.plan_capture(d, d.frame, policy=policy, min_lights=lo)
            assert r.n_captures >= lo, f"{policy} stopped at {r.n_captures} with min_lights={lo}"


def test_unknown_policy_is_refused(stack):
    d = _dome(stack)
    with pytest.raises(ValueError):
        CP.plan_capture(d, d.frame, policy="whatever")


def test_the_conservator_figure_builds_and_states_its_caveats(stack, tmp_path):
    """A figure travels further than the email it was attached to, so the
    unflattering reading has to be printed on the page itself."""
    from qc.rti_figure import build as build_fig
    png = Path(build_fig(stack, tmp_path / "fig.png", sweep_note="six stacks"))
    pdf = png.with_suffix(".pdf")
    assert png.exists() and png.stat().st_size > 50_000
    assert pdf.exists() and pdf.stat().st_size > 10_000
    src = inspect.getsource(__import__("qc.rti_figure", fromlist=["build"]))
    for must_say in ("stopping rule, not the choosing", "synthetic", "invented"):
        assert must_say in src, f"figure no longer states: {must_say}"


# ---------------------------------------------------------------------------
# archive-shaped batch: masters + access copies, missing metadata, 16-bit
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def archive(tmp_path_factory):
    d = tmp_path_factory.mktemp("archive")
    gt = make_archive_fixtures(d)
    summary = run_qc(d, out_dir=d / "_qc", expected_dpi=600, access_dpi=300)
    return gt, summary


def _row(summary, fname):
    return next(x for x in summary["files"] if x["file"] == fname)


def test_archive_access_copies_are_neither_format_failures_nor_duplicates(archive):
    gt, s = archive
    for fname, meta in gt.items():
        if meta["role"] != "access":
            continue
        row = _row(s, fname)
        assert row["role"] == "access"
        assert _status(s, fname, "format") == "pass", fname
        assert _status(s, fname, "duplicate") == "pass", f"{fname} flagged as a duplicate of its master"
        assert _status(s, fname, "dpi") == "pass", fname


def test_archive_every_injected_outcome_and_nothing_else(archive):
    gt, s = archive
    for fname, meta in gt.items():
        row = _row(s, fname)
        expected = {d.partition(":")[0]: (d.partition(":")[2] or "fail") for d in meta["defects"]}
        fired = {c["check"]: c["status"] for c in row["checks"] if c["status"] != "pass"}
        assert fired == expected, f"{fname}: fired {fired}, expected {expected}"
        assert row["action"] == ("accept" if not expected else "review"), fname


def test_missing_resolution_metadata_is_a_review_not_a_reprocess(archive):
    gt, s = archive
    row = _row(s, "photo_0003.tif")
    dpi = next(c for c in row["checks"] if c["check"] == "dpi")
    assert dpi["status"] == "warn" and dpi["value"] is None
    assert "metadata" in dpi["note"]
    assert row["action"] == "review"


def test_16bit_master_is_read_at_the_right_scale(archive):
    gt, s = archive
    row = _row(s, "photo_0004.tif")
    exp = next(c for c in row["checks"] if c["check"] == "exposure")
    assert exp["status"] == "pass", exp
    assert 90 <= exp["value"] <= 245


def test_pairing_flags_the_master_without_access_copy_and_the_orphan(archive):
    gt, s = archive
    assert _status(s, "photo_0005.tif", "pairing") == "warn"
    assert _status(s, "photo_0006.jpg", "pairing") == "warn"
    for fname in ("photo_0001.tif", "photo_0001.jpg", "photo_0004.jpg"):
        assert _status(s, fname, "pairing") == "pass", fname


def test_without_access_dpi_the_old_behaviour_is_unchanged(tmp_path):
    d = tmp_path / "plain"
    make_archive_fixtures(d)
    s = run_qc(d, out_dir=d / "_qc", expected_dpi=600)
    jpg = _row(s, "photo_0001.jpg")
    assert jpg["action"] == "reprocess" and _status(s, "photo_0001.jpg", "format") == "fail"
    assert "role" not in jpg or jpg["role"] == "master"
