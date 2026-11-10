#
# Project : Lavender
# Source  : statistics.jl
# Author  : Li Huang (huangli@caep.cn)
# Status  : Unstable
#
# Last modified: 2026/11/10
#

#=
### *Basic Physical Constants*
=#

"""
    FERMI

Basic physical constant. It is used to denote the fermionic system.

### Notes
This constant is used throughout the codebase to identify the fermionic
systems and to setup correct sign in calculations.

See also: [`BOSE`](@ref).
"""
const FERMI = -1

"""
    BOSE

Basic physical constant. It is used to denote the bosonic system.

### Notes
This constant is used throughout the codebase to identify the bosonic
systems and to setup correct sign in calculations.

See also: [`FERMI`](@ref).
"""
const BOSE = 1

#=
*Remarks* :

The `fermi()` and `bose()` functions live in `math/distributions.jl`. The
constants and the functions are separated only because the constructors in
`types/` use `FERMI`/`BOSE` as default arguments, which forces these
constants to precede the `types/` layer in the include order.
=#
