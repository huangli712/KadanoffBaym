#!/usr/bin/env python3
"""End-to-end KB vs NESSi timing comparison (median protocol).

Runs the quick suite on both sides in alternating rounds (Rust x5 +
NESSi x3) and writes a per-entry median report. Usage:

    python3 .nessi/bench/compare_e2e.py

Prerequisites:
  - cargo build --release --examples          (KB side, incl. bench)
  - .nessi/bench/build.sh                     (NESSi side + kbref suite)
"""
import statistics
import subprocess
import time
from datetime import datetime
from pathlib import Path

BENCH = Path(__file__).resolve().parent
ROOT = BENCH.parent.parent
OUT = Path("/tmp/opencode/kb-nessi-timing")
OUT.mkdir(parents=True, exist_ok=True)

NESSI = [
    ("hubbard_2b", "hubbard_chain_text", "2B"),
    ("hubbard_gw", "hubbard_chain_text", "GW"),
    ("hubbard_tpp", "hubbard_chain_text", "TPP"),
    ("fkm_bethe_quench", "fkm_bethe_quench_text", "none"),
    ("holstein_bethe_migdal", "holstein_bethe_text", "Migdal"),
    ("holstein_bethe_umig", "holstein_bethe_text", "uMig"),
    ("holstein_nambu_migdal", "holstein_nambu_text", "Migdal"),
    ("holstein_nambu_umig", "holstein_nambu_text", "uMig"),
    ("holstein_impurity_migdal", "holstein_impurity_text", "Migdal"),
    ("holstein_impurity_umig", "holstein_impurity_text", "uMig"),
]


def rust_run():
    out = subprocess.run(
        [str(ROOT / "target/release/examples/bench"), "--profile", "quick"],
        capture_output=True, text=True, cwd=ROOT,
    ).stdout
    d = {}
    for ln in out.splitlines():
        p = ln.split()
        if len(p) == 4 and p[3] == "[ok]":
            d[p[0]] = float(p[1])
    assert len(d) == 10, d
    return d


def nessi_round():
    d = {}
    for name, target, approx in NESSI:
        t0 = time.perf_counter()
        r = subprocess.run(
            [str(ROOT / f".nessi/kbref/build/{target}"), approx, "quick", str(OUT)],
            capture_output=True, cwd=ROOT,
        )
        d[name] = time.perf_counter() - t0
        assert r.returncode == 0, (name, r.returncode)
    return d


def main():
    rust, nessi = [], []
    for rnd in range(3):
        rust.append(rust_run())
        nessi.append(nessi_round())
    rust.append(rust_run())
    rust.append(rust_run())

    names = [n for n, _, _ in NESSI]
    lines = []
    lines.append(f"# Rust vs NESSi quick suite, {time.strftime('%Y-%m-%d %H:%M')}")
    lines.append("")
    lines.append(f"{'entry':<26} {'rust med':>9} {'nessi med':>10} {'speedup':>8}")
    tot_r = statistics.median(sum(r.values()) for r in rust)
    tot_n = statistics.median(sum(n.values()) for n in nessi)
    for n in names:
        mr = statistics.median(r[n] for r in rust)
        mn = statistics.median(x[n] for x in nessi)
        lines.append(f"{n:<26} {mr:9.4f} {mn:10.4f} {mn / mr:7.2f}x")
    lines.append("-" * 56)
    lines.append(f"{'TOTAL':<26} {tot_r:9.4f} {tot_n:10.4f} {tot_n / tot_r:7.2f}x")
    lines.append("")
    lines.append("# raw rust runs (totals): " + ", ".join(f"{sum(r.values()):.3f}" for r in rust))
    lines.append("# raw nessi rounds (totals): " + ", ".join(f"{sum(x.values()):.3f}" for x in nessi))
    report = "\n".join(lines)
    dest = BENCH / f"bench-e2e-{datetime.now().strftime('%Y%m%d-%H%M')}.md"
    dest.write_text(report + "\n")
    print(report)
    print(f"\nreport: {dest}")


if __name__ == "__main__":
    main()
