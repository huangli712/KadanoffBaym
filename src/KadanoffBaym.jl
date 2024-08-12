#
# Project : Lavender
# Source  : KadanoffBaym.jl
# Author  : Li Huang (huangli@caep.cn)
# Status  : Unstable
#
# Last modified: 2024/08/12
#

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

#=
### *Using Standard Libraries*
=#

using LinearAlgebra
using InteractiveUtils

#=
### *Using Third-Party Libraries*
=#

#=
### *Includes And Exports* : *global.jl*
=#

#=
*Summary* :

Define some type aliases and string constants for the ACFlow toolkit.

*Members* :

```text
I32, I64, API -> Numerical types (Integer).
F32, F64, APF -> Numerical types (Float).
C32, C64, APC -> Numerical types (Complex).
R32, R64, APR -> Numerical types (Union of Integer and Float).
N32, N64, APN -> Numerical types (Union of Integer, Float, and Complex).
#
__LIBNAME__   -> Name of this julia toolkit.
__VERSION__   -> Version of this julia toolkit.
__RELEASE__   -> Released date of this julia toolkit.
__AUTHORS__   -> Authors of this julia toolkit.
#
authors       -> Print the authors of ACFlow to screen.
```
=#

#
include("global.jl")
#
export I32, I64, API
export F32, F64, APF
export C32, C64, APC
export R32, R64, APR
export N32, N64, APN
#
export __LIBNAME__
export __VERSION__
export __RELEASE__
export __AUTHORS__
#
export authors

#=
### *Includes And Exports* : *types.jl*
=#

#=
*Summary* :

Define some dicts and structs, which are used to store the config
parameters or represent some essential data structures.

*Members* :

```text
DType           -> Customized type.
ADT             -> Customized type.
#
PBASE           -> Configuration dict for general setup.
PMaxEnt         -> Configuration dict for MaxEnt solver.
PBarRat         -> Configuration dict for BarRat solver.
PNevanAC        -> Configuration dict for NevanAC solver.
PStochAC        -> Configuration dict for StochAC solver.
PStochSK        -> Configuration dict for StochSK solver.
PStochOM        -> Configuration dict for StochOM solver.
PStochPX        -> Configuration dict for StochPX solver.
#
AbstractSolver  -> Abstract AC solver.
MaxEntSolver    -> It represents the MaxEnt solver.
BarRatSolver    -> It represents the BarRat solver.
NevanACSolver   -> It represents the NevanAC solver.
StochACSolver   -> It represents the StochAC solver.
StochSKSolver   -> It represents the StochSK solver.
StochOMSolver   -> It represents the StochOM solver.
StochPXSolver   -> It represents the StochPX solver.
#
AbstractData    -> Abstract input data in imaginary axis.
RawData         -> Raw input data.
GreenData       -> Preprocessed input data.
#
AbstractGrid    -> Abstract mesh for input data.
FermionicImaginaryTimeGrid -> Grid in fermionic imaginary time axis.
FermionicFragmentTimeGrid -> Grid in fermionic imaginary time axis (incomplete).
FermionicMatsubaraGrid -> Grid in fermionic Matsubara frequency axis.
FermionicFragmentMatsubaraGrid -> Grid in fermionic Matsubara frequency axis (incomplete).
BosonicImaginaryTimeGrid -> Grid in bosonic imaginary time axis.
BosonicFragmentTimeGrid -> Grid in bosonic imaginary time axis (incomplete).
BosonicMatsubaraGrid -> Grid in bosonic Matsubara frequency axis.
BosonicFragmentMatsubaraGrid -> Grid in bosonic Matsubara frequency axis (incomplete).
#
AbstractMesh    -> Abstract grid for calculated spectral function.
LinearMesh      -> Linear mesh.
TangentMesh     -> Tangent mesh.
LorentzMesh     -> Lorentzian mesh.
HalfLorentzMesh -> Lorentzian mesh at half-positive axis.
DynamicMesh     -> Dynamic (very fine) mesh for stochastic-like solvers.
#
AbstractMC      -> Abstract Monte Carlo engine.
StochACMC       -> Monte Carlo engine used in the StochAC solver.
StochSKMC       -> Monte Carlo engine used in the StochSK solver.
StochOMMC       -> Monte Carlo engine used in the StochOM solver.
StochPXMC       -> Monte Carlo engine used in the StochPX solver.
```
=#

#
include("types.jl")
#
export DType
export ADT
#
export PBASE
export PMaxEnt
export PBarRat
export PNevanAC
export PStochAC
export PStochSK
export PStochOM
export PStochPX
#
export AbstractSolver
export MaxEntSolver
export BarRatSolver
export NevanACSolver
export StochACSolver
export StochSKSolver
export StochOMSolver
export StochPXSolver
#
export AbstractData
export RawData
export GreenData
#
export AbstractGrid
export FermionicImaginaryTimeGrid
export FermionicFragmentTimeGrid
export FermionicMatsubaraGrid
export FermionicFragmentMatsubaraGrid
export BosonicImaginaryTimeGrid
export BosonicFragmentTimeGrid
export BosonicMatsubaraGrid
export BosonicFragmentMatsubaraGrid
#
export AbstractMesh
export LinearMesh
export TangentMesh
export LorentzMesh
export HalfLorentzMesh
export DynamicMesh
#
export AbstractMC
export StochACMC
export StochSKMC
export StochOMMC
export StochPXMC

end