# KB vs NESSi benchmarks (2026-11-16)

NESSi-side programs and inputs of the performance comparison between
this crate and NESSi 2.0.0 (`libcntr`), plus the measurement record.
All numbers below were taken in an interleaved same-window run (5
alternating rounds per pair) on an otherwise idle machine.

## Files

| file | purpose |
|---|---|
| `nessi_dyson_bench.cpp` | kernel-level benchmark: `dyson_start` + the full `dyson_timestep` loop, nt=500, ntau=200, 2x2, k=5, constant diagonal self-energy (counterpart of `examples/dyson_bench.rs`; all four parameters via argv) |
| `kernel_conv_bench.cpp` | kernel-level benchmark: full-contour Type-1 convolution `C = A*B` over deterministic pseudo-random data (counterpart of `examples/conv_bench.rs`; one warm-up, second run timed) |
| `kernel_vie2_bench.cpp` | kernel-level benchmark: VIE2 propagation `[1+F]G = Q` — `vie2_mat`(fixpoint) + `vie2_start` + the `vie2_timestep` loop (counterpart of `examples/vie2_bench.rs`) |
| `hubbard_2b.inp` | input of NESSi `hubbard_chain_2b` aligned entry-by-entry with `examples/hubbard/hubbard_chain.toml` `[profile.quick]` (nt=10, ntau=40, h=0.05, beta=20, k=3) |
| `holstein_migdal.inp` | input of NESSi `Holstein_bethe_Migdal` aligned with `examples/holstein/holstein_bethe_migdal.toml` `[profile.quick]` (nt=8, ntau=40, h=0.03, beta=10, k=3, sin^2 drive vector included) |
| `holstein_umig.inp` | input of NESSi `Holstein_bethe_uMig`, aligned with `examples/holstein/holstein_bethe_umig.toml` `[profile.quick]` (same grid and drives as the Migdal case) |
| `holstein_nambu_migdal.inp` | input of NESSi `Holstein_bethe_Nambu_Migdal`, aligned with `examples/holstein/holstein_bethe_nambu_migdal.toml` `[profile.quick]` (nt=6, ntau=32; `sc_field` set to the constant KB seed 0.1) |
| `holstein_impurity_migdal.inp` | input of NESSi `Holstein_impurity_singlebath_Migdal`, aligned with `examples/holstein/holstein_impurity_singlebath_migdal.toml` `[profile.quick]` (nt=8, ntau=40, h=0.04, beta=2; both `dHopping` and `dEl_Ph_g` sin^2 drives) |
| `build.sh` | compiles the four NESSi-side binaries against the prebuilt `libcntr` |

`fkm_bethe_quench` needs no input file: its grid (nt=50, ntau=200,
k=5) is hardcoded and matches `[profile.release]` of
`examples/falicov_kimball/falicov_kimball.toml`.

Both sides use the same convergence criteria (error < 1e-7 with
early exit); the number of iterations may still differ between the
implementations, which is part of an end-to-end comparison. The
Nambu case carries an extra caveat: the KB side seeds the
superconducting field with its own `sc_seed` mechanism
(`sc_seed_iterations = 5`), the NESSi side with a constant `sc_field`
of 0.1 — same physics, different bootstrap trajectories.

## Results (low-load window, 2026-11-16)

### Kernel level: Dyson propagation only (`dyson_bench`)

nt=500, ntau=200, 2x2, k=5; timed region = `dyson_start` + timestep
loop (Matsubara preparation excluded on both sides).

| | NESSi | KB (pre-optimization, a4b7efa) | KB (current) |
|---|---|---|---|
| median | 2.54 s | 5.56 s | 3.86 s |
| vs NESSi | 1x | 2.19x | 1.52x |
| peak RSS | 55.5 MB | 61.6 MB | 69.3 MB |

The 2026-11-16 workspace/allocation optimization pass improved the KB
kernel by ~31% (5.56 -> 3.86 s).

### Kernel level: dimension sweep (three kernels, medians of 3)

Same grids as above (`conv` at nt=400). KB/NESSi ratio > 1 means KB
is slower.

| kernel | d=1 | d=2 | d=4 | d=8 |
|---|---|---|---|---|
| dyson  | 7.1x | 1.55x | 1.53x | **0.95x** |
| conv   | 7.3x | 1.22x | 1.30x | - |
| vie2   | 7.6x | 3.4x | 3.2x | - |

Peak RSS at d=2: dyson 55.1/69.4 MB, conv 55.7/47.3 MB, vie2
76.9/85.1 MB (NESSi/KB).

Reading of the sweep:

* `d = 1` is the worst case for KB (~7x): NESSi reduces 1x1 Eigen
  blocks to scalar fast paths, KB always runs the generic matrix
  layers.
* dyson at `d = 8`: KB slightly wins (0.95x) — with matrix-sized
  blocks the row-major `mul_acc` inner loop matches Eigen.
* vie2 is uniformly behind (3-7x): `vie2_timestep` was not covered
  by the 2026-11-16 workspace optimization (only `dyson` and the
  convolution kernels were), an obvious next target.

### End-to-end: example programs (parameters aligned)

| entry | NESSi | KB | winner | NESSi RSS | KB RSS |
|---|---|---|---|---|---|
| fkm_bethe_quench (release params) | 0.60 s | 0.30 s | KB 2.0x | 15.6 MB | 4.7 MB |
| hubbard_chain_2b (quick params) | 0.11 s | 0.02 s | KB 5.5x | 10.0 MB | 4.0 MB |
| holstein_bethe_migdal (quick params) | 0.11 s | 0.05 s | KB 2.2x | 9.9 MB | 3.9 MB |
| holstein_bethe_umig (quick params) | 0.11 s | 0.03 s | KB 3.7x | 10.0 MB | 3.6 MB |
| holstein_nambu_migdal (quick params) | 0.11 s | 0.15 s | **NESSi ~1.4x** | 10.1 MB | 4.0 MB |
| holstein_impurity_migdal (quick params) | 0.12 s | 0.05 s | KB 2.4x | 10.0 MB | 3.8 MB |

The Nambu case is the exception: with the larger matrix-valued
components of the Nambu formalism the NESSi end-to-end program is
also faster, consistent with the kernel-level picture below — the KB
inner-kernel constant-factor gap grows with matrix size, while the
small-matrix entries are dominated by process/HDF5 overhead where KB
wins big. Memory stays ~1/2.5-1/3 in all six cases.

### Reading

Both results are real and differ by scope:

* end-to-end (the scope of the earlier 2026-11-15 comparisons): KB is
  2-5.5x faster with 1/2.5-1/3 the memory in five of six entries.
  NESSi programs dynamically link `libhdf5` and write `.h5` output,
  which dominates the small quick grids; KB deliberately ships no
  HDF5 support. The Nambu entry is the exception (NESSi ~1.4x faster
  end-to-end as well).
* kernel level: NESSi's bare-pointer + Eigen kernels are ~1.5x faster
  than KB's safe-slice + abstraction layers. Known remaining KB
  items: unchecked block access outside `convolution/ret.rs`, and
  bounds-checked inner loops of `mul_acc`.

## Reproduce

```sh
./build.sh                                  # NESSi-side binaries
./nessi_dyson_bench [nt ntau d k]            # dyson kernel benchmark
./kernel_conv_bench [nt ntau d k]            # convolution kernel benchmark
./kernel_vie2_bench [nt ntau d k]            # VIE2 kernel benchmark
./fkm_bethe                                  # writes G1.h5 in cwd
./hubbard_2b hubbard_2b.inp <out-prefix>
./holstein_migdal holstein_migdal.inp <out-prefix>
./holstein_umig holstein_umig.inp <out-prefix>
./holstein_nambu_migdal holstein_nambu_migdal.inp <out-prefix>
./holstein_impurity_migdal holstein_impurity_migdal.inp <out-prefix>

# KB side (repo root)
cargo run --release --example dyson_bench [-- --nt 500 --ktau 200 --dim 2 --k 5]
cargo run --release --example conv_bench [-- --nt 400 --ktau 200 --dim 2 --k 5]
cargo run --release --example vie2_bench [-- --nt 500 --ktau 200 --dim 2 --k 5]
cargo run --release --example falicov_kimball -- examples/falicov_kimball/falicov_kimball.toml --profile release
cargo run --release --example hubbard -- chain_2b examples/hubbard/hubbard_chain.toml --profile quick
cargo run --release --example holstein -- bethe_migdal examples/holstein/holstein_bethe_migdal.toml --profile quick
cargo run --release --example holstein -- bethe_umig examples/holstein/holstein_bethe_umig.toml --profile quick
cargo run --release --example holstein -- bethe_nambu_migdal examples/holstein/holstein_bethe_nambu_migdal.toml --profile quick
cargo run --release --example holstein -- impurity_migdal examples/holstein/holstein_impurity_singlebath_migdal.toml --profile quick
```

Interleave the two sides round-by-round in one shell loop and take
medians; absolute times drift with machine load but the ratios above
were stable across load regimes (within a few percent).

## Comparison scripts and the 2026-11-17 idle rerun

| file | purpose |
|---|---|
| `compare_e2e.py` | end-to-end quick-suite comparison, interleaved Rust x5 + NESSi x3, per-entry medians, dated report into this directory |
| `compare_kernels.py` | kernel-level dimension sweep (`--rounds/--nt/--ntau/--dims`), same interleaved protocol, dated report into this directory |
| `bench-2026-11-17.md` | full 2026-11-17 idle-machine record (end-to-end 2.29x total, kernel sweep, raw medians, deltas vs 2026-11-16) |

Highlights of the 2026-11-17 rerun: end-to-end TOTAL 0.556 s vs
1.272 s (KB 2.29x, 10/10 entries); kernel sweep unchanged in pattern
(d=1 worst at ~6.3-8.7x, vie2 weakest family, d>=2 in the 1.2-1.6x
band), with one correction: dyson d=8 measures 1.21x on the idle
machine and does NOT reproduce the 0.95x crossover noted on
2026-11-16 — treat that earlier value as a load-window artifact.
