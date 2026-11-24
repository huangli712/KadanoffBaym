# KadanoffBaym

The `KadanoffBaym` library implements the basic operations and numerical
algorithms for contour-ordered Green's functions (i.e. non-equilibrium
Green's functions), and a minimal dynamical mean-field theory engine for
strongly correlated systems in non-equilibrium states.

The numerical results are validated pointwise against the reference data
under `data/`.

## Layout

- `src/` — the library (single crate, zero dependencies,
  `#![forbid(unsafe_code)]`): complex/matrix/LU/eigensolver/expm
  numerics, contour-ordered Green's function containers, convolution
  kernels and drivers, Fourier/Matsubara transforms, Dyson/VIE2 solvers,
  moving-window (truncated) propagation, steady-state and
  pseudo-particle families, plain-text IO.
- `tests/` — the integration test suite (including reference-data
  comparisons).
- `examples/` — the tutorial programs: Hubbard
  chain 2B/GW/TPP, Falicov-Kimball Bethe, Holstein Bethe/impurity (Migdal/uMig,
  normal and Nambu), steady-state and truncated DMFT, GW with the
  Peierls phase, plus the `validate` and `bench` drivers.
- `util/` — shared helpers of the examples (a handwritten TOML subset
  parser, configuration validation, text output, iteration loops, reference
  comparison tools).
- `data/` — the reference data (`data/test`, `data/tutor`).

## Build and test

```
cargo build --release --examples   # library + all examples
cargo test                         # quick suite (522 tests)
cargo test --release               # release suite (order gates)
```

## Validation against the reference data

```
cargo build --release --examples
./target/release/examples/validate          # runs every example and
                                            # compares with data/tutor
```

The library test suite also compares pointwise with `data/test`
(reference `ref 1` files) across the whole suite.

## Benchmarks

```
./target/release/examples/bench --profile=quick
```

## Version

v0.7.4 (Rust implementation)

## License

GNU General Public License Version 3
