"""CLI surface (MHS interface #2)."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fixtures.make_fixtures import make, make_vlm_fixtures, make_archive_fixtures
from fixtures.make_rti_stack import make_stack
from mhs_shim.backends.folder_scanner import VirtualScanner
from qc.agent import run_qc
from qc.vlm import make_backend
from qc.rti_experiment import run as run_rti, render as render_rti
from qc.rti_figure import build as build_rti_figure


def main(argv=None):
    ap = argparse.ArgumentParser(prog="mhs-spike")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("fixtures", help="synthesise a batch with injected defects")
    a.add_argument("out"); a.add_argument("--kind", choices=["rules", "vlm", "archive"], default="rules")
    b = sub.add_parser("describe", help="print the device reference an agent would read"); b.add_argument("folder")
    c = sub.add_parser("qc", help="run QC agent over a capture folder")
    c.add_argument("folder"); c.add_argument("--dpi", type=int, default=400); c.add_argument("--out")
    c.add_argument("--vlm", default="none", help="backend: none | anthropic[:model] | gemini | stub")
    c.add_argument("--vlm-mode", default="flagged", help="none | flagged | all | sample:N")
    c.add_argument("--access-dpi", type=int, default=None,
                   help="declared resolution of JPEG access copies; when set, JPEGs are checked as access copies and paired with masters")
    d = sub.add_parser("rti-stack", help="synthesise a physically-rendered RTI light stack")
    d.add_argument("out"); d.add_argument("--seed", type=int, default=7)
    d.add_argument("--noise", type=float, default=0.012)
    e = sub.add_parser("rti", help="compare full-dome vs sequential vs adaptive capture")
    e.add_argument("stack"); e.add_argument("--budget", type=float, default=100.0)
    e.add_argument("--out")
    f = sub.add_parser("rti-figure", help="one page for a conservator: PNG + PDF")
    f.add_argument("stack"); f.add_argument("out", nargs="?", default="runs/figures/rti_stopping.png")
    f.add_argument("--note", default="")
    args = ap.parse_args(argv)
    if args.cmd == "fixtures":
        fn = {"rules": make, "vlm": make_vlm_fixtures, "archive": make_archive_fixtures}[args.kind]
        gt = fn(Path(args.out)); print(f"wrote {len(gt)} files to {args.out}")
    elif args.cmd == "rti-stack":
        print(make_stack(Path(args.out), seed=args.seed, noise=args.noise))
    elif args.cmd == "rti":
        r = run_rti(args.stack, budget_lux_h=args.budget, out=args.out)
        print(render_rti(r))
        a = r["arms"]
        print(f"\nadaptive vs sequential: {a['adaptive']['n_captures']} vs "
              f"{a['sequential']['n_captures']} captures for "
              f"{a['adaptive']['d_prime_pct_of_full']:.0f}% vs "
              f"{a['sequential']['d_prime_pct_of_full']:.0f}% of full-dome legibility")
    elif args.cmd == "rti-figure":
        png = build_rti_figure(args.stack, args.out, sweep_note=args.note)
        print(f"{png}\n{Path(png).with_suffix('.pdf')}")
    elif args.cmd == "describe":
        print(VirtualScanner(args.folder).reference_doc())
    elif args.cmd == "qc":
        backend = make_backend(args.vlm)
        s = run_qc(args.folder, out_dir=args.out, expected_dpi=args.dpi, access_dpi=args.access_dpi,
                   vlm=backend, vlm_mode=args.vlm_mode if backend else "none")
        print(f"{s['n_files']} files  actions={s['actions']}  missing={s['missing_sequence']}")
        v = s["vlm"]
        if v["backend"]:
            print(f"vlm {v['backend']} mode={v['mode']} judged={v['n_judged']} errors={v['n_errors']} "
                  f"escalated={v['escalated']} tokens={v['input_tokens']}in/{v['output_tokens']}out")
        print(f"reshoot: {s['reshoot']}\nreport: {s['out_dir']}/report.md")


if __name__ == "__main__":
    main()
