#
# Project : Lavender
# Source  : convention.jl
# Author  : Li Huang (huangli@caep.cn)
# Status  : Unstable
#
# Last modified: 2026/11/10
#

#=
*Remarks* : *Pseudo-Particle Convention*

The standard (`ℱ`) and pseudo-particle (`PseudoGF`) Green's functions
share the same storage layout. They differ in the meaning of the stored
real time triangle (``G^{R}`` versus ``G^{>}``) and in the Matsubara and
left-mixing algorithms. The convention is a static property of the type,
and it is used to select the algorithms by dispatch.

*References* :

Please see [`NESSi`] `cntr_herm_pseudo_decl.hpp`.
=#

"""
    gf_convention(::Type{<:ℱ})
    gf_convention(::Type{<:PseudoGF})

Return the convention of the contour-ordered Green's function, namely
`Val(:standard)` or `Val(:pseudo)`. It is used to select the Matsubara
and left-mixing algorithms only.

See also: [`ℱ`](@ref), [`PseudoGF`](@ref).
"""
gf_convention(::Type{<:ℱ}) = Val(:standard)

gf_convention(::Type{<:PseudoGF}) = Val(:pseudo)
#

gf_convention(G::Union{ℱ,PseudoGF}) = gf_convention(typeof(G))

