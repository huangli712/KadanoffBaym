# test_equilibrium

Port of NESSi `examples/programs/test_equilibrium.cpp`.

A level `eps1` is coupled to a second level `eps2` by the hopping `i lambda`.
The Matsubara Dyson equation of the first level is solved with the embedding
self-energy `Sigma = lambda^2 g2`, where `g2` is the free Green's function of
`eps2`. The result is compared with the (1,1) element of the exact 2 x 2
Green's function.

## Run

```bash
julia --project=. tutor/test_equilibrium/test_equilibrium.jl \
      tutor/test_equilibrium/test_equilibrium.toml out.dat
```

## Input (TOML)

| Key          | Meaning                                       |
|--------------|-----------------------------------------------|
| `Ntau`       | number of imaginary time intervals (even)     |
| `SolveOrder` | order of the integration, 1 to 5              |
| `beta`       | inverse temperature                           |
| `eps1`       | energy of the first level                     |
| `eps2`       | energy of the second level                    |
| `lambda`     | hopping between the levels                    |
| `mu`         | chemical potential                            |

## Output

The output file is overwritten. It contains a header and one line with two
columns, `err_fixpoint err_cg`, where `err = distance(G_exact, G, 0) / Ntau`
for the FIXPOINT and CG methods of `dyson_mat!`.

## Difference from NESSi

NESSi writes `err_fourier err_fixpoint`. The FOURIER method is deferred to
Phase 7, so this example writes `err_fixpoint err_cg` instead.
