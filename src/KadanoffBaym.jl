"""
    KadanoffBaym

The `KadanoffBaym` package is a state-of-the-art computational framework
for simulating the non-equilibrium strongly correlated electron systems.
It provides some useful application programming interfaces to manipulate
the non-equilibrium Green's functions defined on the 𝐿-shape Kadanoff-Baym
contour, including:

* Basic integration and differentiation rules
* Basic operations for Contour Green's functions
* Basic diagrammatic algorithms based on many-body perturbation theory
* Solve Volterra integral equations
* Solve Volterra integro-differential equations
* Convolution between two contour Green's functions

This package is inspired by the `NESSi` (The Non-Equilibrium Systems
Simulation package) code, which was developed and maintained by Martin
Eckstein *et al*. Actually, it can be regarded as a replacement of the
`NESSi` package for those peoples who don't like or aren't familiar
with `C++`.

### References:

[`NESSi`]
Martin Eckstein, *et al.*,
NESSi: The Non-Equilibrium Systems Simulation package,
*Computer Physics Communications* **257**, 107484 (2020)

[`REVIEW`]
Philipp Werner, *et al.*,
Nonequilibrium dynamical mean-field theory and its applications,
*Reviews of Modern Physics* **86**, 779 (2014)

[`MABOOK`]
Johan de Villiers,
Mathematics of Approximation,
*Atlantis Press* (2012)

[`MATABLE`]
Dan Zwillinger (editor),
CRC Standard Mathematical Tables and Formulas (33rd edition),
*CRC Press* (*Taylor & Francis Group*) (2018)

[`QUADRATURE`]
Ruben J. Espinosa-Maldonado and George D. Byrne,
On the Convergence of Quadrature Formulas,
*SIAM J. Numer. Anal.* **8**, 110 (1971)
"""
module KadanoffBaym
end