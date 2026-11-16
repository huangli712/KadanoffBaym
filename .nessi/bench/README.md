# KB vs NESSi benchmarks (2026-11-16)

NESSi-side programs and inputs of the performance comparison between
this crate and NESSi 2.0.0 (`libcntr`), plus the measurement record.
All numbers below were taken in an interleaved same-window run (5
alternating rounds per pair) on an otherwise idle machine.

## Files

| file | purpose |
|---|---|
| `nessi_dyson_bench.cpp` | kernel-level benchmark: `dyson_start` + the full `dyson_timestep` loop, nt=500, ntau=200, 2x2, k=5, constant diagonal self-energy (counterpart of `examples/dyson_bench.rs`) |
| `hubbard_2b.inp` | input of NESSi `hubbard_chain_2b` aligned entry-by-entry with `examples/hubbard/hubbard_chain.toml` `[profile.quick]` (nt=10, ntau=40, h=0.05, beta=20, k=3) |
| `holstein_migdal.inp` | input of NESSi `Holstein_bethe_Migdal` aligned with `examples/holstein/holstein_bethe_migdal.toml` `[profile.quick]` (nt=8, ntau=40, h=0.03, beta=10, k=3, sin^2 drive vector included) |
| `build.sh` | compiles the four NESSi-side binaries against the prebuilt `libcntr` |

`fkm_bethe_quench` needs no input file: its grid (nt=50, ntau=200,
k=5) is hardcoded and matches `[profile.release]` of
`examples/falicov_kimball/falicov_kimball.toml`.

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

### End-to-end: example programs (parameters aligned)

| entry | NESSi | KB | KB/NESSi | NESSi RSS | KB RSS |
|---|---|---|---|---|---|
| fkm_bethe_quench (release params) | 0.60 s | 0.30 s | 2.0x faster | 15.6 MB | 4.7 MB |
| hubbard_chain_2b (quick params) | 0.11 s | 0.02 s | 5.5x faster | 10.0 MB | 4.0 MB |
| holstein_bethe_migdal (quick params) | 0.11 s | 0.05 s | 2.2x faster | 9.9 MB | 3.9 MB |

### Reading

Both results are real and differ by scope:

* end-to-end (the scope of the earlier 2026-11-15 comparisons): KB is
  2-5.5x faster with 1/2.5-1/3 the memory. NESSi programs dynamically
  link `libhdf5` and write `.h5` output, which dominates the small
  quick grids; KB deliberately ships no HDF5 support.
* kernel level: NESSi's bare-pointer + Eigen kernels are ~1.5x faster
  than KB's safe-slice + abstraction layers. Known remaining KB
  items: unchecked block access outside `convolution/ret.rs`, and
  bounds-checked inner loops of `mul_acc`.

## Reproduce

```sh
./build.sh                                  # NESSi-side binaries
./nessi_dyson_bench                          # kernel benchmark
./fkm_bethe                                  # writes G1.h5 in cwd
./hubbard_2b hubbard_2b.inp <out-prefix>
./holstein_migdal holstein_migdal.inp <out-prefix>

# KB side (repo root)
cargo run --release --example dyson_bench
cargo run --release --example falicov_kimball -- examples/falicov_kimball/falicov_kimball.toml --profile release
cargo run --release --example hubbard -- chain_2b examples/hubbard/hubbard_chain.toml --profile quick
cargo run --release --example holstein -- bethe_migdal examples/holstein/holstein_bethe_migdal.toml --profile quick
```

Interleave the two sides round-by-round in one shell loop and take
medians; absolute times drift with machine load but the ratios above
were stable across load regimes (within a few percent).
