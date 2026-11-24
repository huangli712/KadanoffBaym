#!/usr/bin/env python3
"""Instruction/cycle comparison KB vs NESSi (perf hardware counters).

Runs every (kernel, dimension) pair of the benchmark grid on each side,
pinned to CPU 0 (a P-core) and counted with the core PMU, and reports
medians of retired instructions, cycles and IPC. A ratio > 1 means KB
executes more instructions. Retired instruction counts are deterministic
and unaffected by machine load; cycles and IPC still drift with the core
frequency.

Usage:
    python3 .nessi/bench/perf_compare.py [--rounds N] [--ntau N]
                                         [--dims 1,2] [--k 5]

Prerequisites:
  - cargo build --release --example bench_convolution --example bench_dyson \
                         --example bench_vie2
  - .nessi/bench/build.sh
  - perf with core-PMU access (kernel.perf_event_paranoid <= 1)
"""
import argparse
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
PERF = [
    "taskset", "-c", "0", "perf", "stat", "-x,",
    "-e", "cpu_core/instructions/,cpu_core/cycles/",
]


def run_once(cmd):
    r = subprocess.run(PERF + cmd, capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, (cmd, r.returncode, r.stderr[-300:])
    vals = {}
    for ln in r.stderr.splitlines():
        p = ln.split(",")
        if len(p) >= 3 and p[2].startswith("cpu_core/"):
            vals[p[2]] = float(p[0])
    return vals["cpu_core/instructions/"], vals["cpu_core/cycles/"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=2)
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

    lines = [
        f"# Instruction/cycle comparison KB vs NESSi, "
        f"{datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        f"rounds={args.rounds}, ntau={args.ntau}, k={args.k}, "
        "pinned to CPU 0 (P-core), cpu_core PMU",
        "",
        f"{'kernel':<12} {'KB instr':>14} {'NESSi instr':>14} {'instr x':>8} "
        f"{'KB IPC':>7} {'NESSi IPC':>9}",
    ]
    for key, v in res.items():
        kb_i = statistics.median(x[0] for x in v["kb"])
        nn_i = statistics.median(x[0] for x in v["nessi"])
        kb_c = statistics.median(x[1] for x in v["kb"])
        nn_c = statistics.median(x[1] for x in v["nessi"])
        lines.append(
            f"{key:<12} {kb_i:14.4g} {nn_i:14.4g} {kb_i / nn_i:7.2f}x "
            f"{kb_i / kb_c:7.2f} {nn_i / nn_c:9.2f}"
        )
    lines.append("")
    for key, v in res.items():
        lines.append(
            f"# {key}: kb {['%.4g' % x[0] for x in v['kb']]} "
            f"nessi {['%.4g' % x[0] for x in v['nessi']]}"
        )
        lines.append(
            f"# {key}: kb cycles {['%.4g' % x[1] for x in v['kb']]} "
            f"nessi cycles {['%.4g' % x[1] for x in v['nessi']]}"
        )
    report = "\n".join(lines)
    dest = BENCH / f"bench-perf-{datetime.now().strftime('%Y%m%d-%H%M')}.md"
    dest.write_text(report + "\n")
    print(report)
    print(f"\nreport: {dest}")


if __name__ == "__main__":
    main()
