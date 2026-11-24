#!/usr/bin/env python3
"""Kernel-level KB vs NESSi timing comparison (dimension sweep).

For every (kernel, dimension) pair of the benchmark grid, runs the KB
side (examples/{conv,dyson,vie2}_bench.rs) and the NESSi side
(.nessi/bench/{kernel_conv_bench,nessi_dyson_bench,kernel_vie2_bench})
in alternating rounds and reports the per-pair medians. Ratio > 1
means KB is slower. Usage:

    python3 .nessi/bench/compare_kernels.py [--rounds N] [--nt N]
                                            [--ntau N] [--dims 1,2,4]

Prerequisites:
  - cargo build --release --example bench_convolution --example bench_dyson \
                         --example bench_vie2
  - .nessi/bench/build.sh
"""
import argparse
import re
import statistics
import subprocess
from datetime import datetime
from pathlib import Path

BENCH = Path(__file__).resolve().parent
ROOT = BENCH.parent.parent

SWEEP = [
    ("conv", "bench_convolution", "kernel_conv_bench"),
    ("dyson", "bench_dyson", "nessi_dyson_bench"),
    ("vie2", "bench_vie2", "kernel_vie2_bench"),
]
DEFAULT_DIMS = {"conv": [1, 2, 4], "dyson": [1, 2, 4, 8], "vie2": [1, 2, 4]}


def run_once(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, (cmd, r.returncode, r.stderr[-200:])
    m = re.search(r"([0-9.]+) s", r.stdout)
    assert m, r.stdout
    return float(m.group(1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--nt", type=int, default=0, help="override nt (default per kernel)")
    ap.add_argument("--ntau", type=int, default=200)
    ap.add_argument("--dims", type=str, default="", help="comma list, e.g. 1,2")
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()
    dims = [int(x) for x in args.dims.split(",")] if args.dims else None

    res = {}
    for kern, rex, nex in SWEEP:
        nt = args.nt or {"conv": 400, "dyson": 500, "vie2": 500}[kern]
        for d in dims or DEFAULT_DIMS[kern]:
            key = f"{kern} d={d}"
            res[key] = {"kb": [], "nessi": []}
            for _ in range(args.rounds):
                res[key]["kb"].append(run_once([
                    str(ROOT / f"target/release/examples/{rex}"),
                    "--nt", str(nt), "--ktau", str(args.ntau),
                    "--dim", str(d), "--k", str(args.k),
                ]))
                res[key]["nessi"].append(run_once([
                    str(BENCH / nex),
                    str(nt), str(args.ntau), str(d), str(args.k),
                ]))

    lines = [f"# Kernel sweep KB vs NESSi, {datetime.now().strftime('%Y-%m-%d %H:%M')}",
             "", f"rounds={args.rounds}, ntau={args.ntau}, k={args.k}",
             "", f"{'kernel':<22} {'KB med':>9} {'NESSi med':>10} {'ratio':>9}"]
    for key, v in res.items():
        kb = statistics.median(v["kb"])
        nn = statistics.median(v["nessi"])
        lines.append(f"{key:<22} {kb:9.4f} {nn:10.4f} {kb / nn:8.2f}x")
    lines.append("")
    for key, v in res.items():
        lines.append(f"# {key}: kb {['%.3f' % x for x in v['kb']]}  "
                     f"nessi {['%.3f' % x for x in v['nessi']]}")
    report = "\n".join(lines)
    dest = BENCH / f"bench-kernels-{datetime.now().strftime('%Y%m%d-%H%M')}.md"
    dest.write_text(report + "\n")
    print(report)
    print(f"\nreport: {dest}")


if __name__ == "__main__":
    main()
