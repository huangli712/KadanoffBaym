# KadanoffBaym

The `KadanoffBaym` library implements the basic operations and numerical
algorithms for contour-ordered Green's functions (i.e. non-equilibrium
Green's functions), and a minimal dynamical mean-field theory engine for
strongly correlated systems in non-equilibrium states.

The numerical results are validated pointwise against NESSi 2.0.0
(see `.nessi/`).

## Layout

- `src/` — the library (single crate, zero dependencies,
  `#![forbid(unsafe_code)]`): complex/matrix/LU/eigensolver/expm
  numerics, contour-ordered Green's function containers, convolution
  kernels and drivers, Fourier/Matsubara transforms, Dyson/VIE2 solvers,
  moving-window (truncated) propagation, NESS steady-state and
  pseudo-particle families, plain-text IO.
- `tests/` — the integration test suite (including NESSi
  reference-data comparisons).
- `examples/` — the tutorial programs: Hubbard
  chain 2B/GW/TPP, FKM Bethe, Holstein Bethe/impurity (Migdal/uMig,
  normal and Nambu), steady-state and truncated DMFT, GW with the
  Peierls phase, plus the `validate` and `bench` drivers.
- `examples/common/` — shared helpers of the examples (a handwritten
  TOML subset parser, configuration validation, text output, iteration
  loops, kbref comparison tools).
- `data/` — the NESSi reference data (`data/test`, `data/tutor`).

## Build and test

```
cargo build --release --examples   # library + all examples
cargo test                         # quick suite (515 tests)
cargo test --release               # release suite (order gates)
```

## Validation against NESSi

```
cargo build --release --examples
./target/release/examples/validate          # runs every example and
                                            # compares with data/tutor
```

The library test suite also compares pointwise with `data/test`
(NESSi kbref reference files) across all phases.

## Benchmarks

```
./target/release/examples/bench --profile=quick
```

## Version

v0.1.0-devel (Rust rewrite, phases R0-R10 complete)

## License

GNU General Public License Version 3
