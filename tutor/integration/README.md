# integration

Port of NESSi `examples/programs/integration.cpp`.

The integral `int_0^x exp(i t) dt = (exp(i x) - 1) / i` is evaluated by the
Gregory quadrature of order `k` (the weights of `Integrator(k)`) on the grid
`x = n h`, `n = 0, ..., npts`.

## Run

```bash
julia --project=. tutor/integration/integration.jl \
      tutor/integration/integration.toml out.dat
```

## Input (TOML)

| Key    | Meaning                                         |
|--------|-------------------------------------------------|
| `npts` | the grid is `x = n h`, `n = 0, ..., npts`       |
| `h`    | grid spacing (`0.05 pi` as in the NESSi demo)   |
| `k`    | order of the Gregory quadrature, 1 to 5         |

## Output

The output file is overwritten. It contains a header and two columns,
`x abs(error)`.

In addition, the script checks the convergence order with the shared gate
`test/common/order_gate.jl`: for every `k = 1, ..., 5`, the error at the
fixed point `x = 2.5` must decrease as `h^(k+2)` on four grids. The
rectangle rule (first order) is used as a reduced-order mutant, which must
fail the same gate.
