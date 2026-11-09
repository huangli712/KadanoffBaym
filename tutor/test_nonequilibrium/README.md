# test_nonequilibrium

Port of NESSi `examples/programs/test_nonequilibrium.cpp`.

A level `eps1` is coupled to a second level `eps2` by the hopping `i lambda`.
The 1 x 1 Green's function of the first level is solved with the embedding
self-energy `Sigma = lambda^2 g2`, where `g2` is the free Green's function of
`eps2`, by two methods:

* the Dyson equation (integro-differential form): `dyson_mat!`,
  `dyson_start!`, and `dyson_timestep!`;
* the VIE2 (integral form) `[1 + F] * G = G0` with `F = -G0 * Sigma` and
  `Fcc = -Sigma * G0`: `vie2_mat!`, `vie2_start!`, and `vie2_timestep!`.
  The kernels are built time step by time step, as in NESSi.

Both results are compared with the (1,1) element of the exact 2 x 2 Green's
function.

## Run

```bash
julia --project=. tutor/test_nonequilibrium/test_nonequilibrium.jl \
      tutor/test_nonequilibrium/test_nonequilibrium.toml out.dat
```

## Input (TOML)

| Key          | Meaning                                         |
|--------------|-------------------------------------------------|
| `Nt`         | number of real time intervals, `h = Tmax / Nt`  |
| `Ntau`       | number of imaginary time intervals (even)       |
| `Tmax`       | length of the real time axis                    |
| `SolveOrder` | order of the integration, 1 to 5                |
| `beta`       | inverse temperature                             |
| `eps1`       | energy of the first level                       |
| `eps2`       | energy of the second level                      |
| `lambda`     | hopping between the levels                      |
| `mu`         | chemical potential                              |

`Nt` and `Ntau` are numbers of intervals, as in NESSi. The KB contour has
`Nt + 1` real time points and `Ntau + 1` imaginary time points.

## Output

The output file is overwritten. It contains a header and one line with two
columns, `err_dyson err_vie2`. The error is the normalized squared norm of
NESSi:

```
err = sum_t [ 2 d_les(t) / Nt^2 + 2 d_ret(t) / Nt^2 + d_tv(t) / (Nt Ntau) ],
```

where `d_X(t)` is the block Frobenius distance of the component `X` at time
step `t` (`distance(G1, G2, t, Val(:X))`).
