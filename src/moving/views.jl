#
# Project : Lavender
# Source  : %s
# Author  : Li Huang (huangli@caep.cn)
# Status  : Unstable
#
# Last modified: 2026/11/10
#

#=
### *Moving Window* : *Traits*
=#

#=
*Remarks* : *Moving Time Steps*

The traits of the moving types act on one time step (a row of the
window): an owning `𝒻ᵐᵒᵛ` or a view `𝒻ᵐᵒᵛView` of a row of `ℱᵐᵒᵛ`. They
work on the contiguous row slices directly. A view writes through to its
parent and is checked against the generation of the parent first.

Please see [`NESSi`] `cntr_herm_matrix_timestep_moving_view_impl.hpp`.
=#

"""
    _mrows(x::Union{𝒻ᵐᵒᵛ,𝒻ᵐᵒᵛView}, c::Symbol)

Zero-copy view `(ndim1, ndim2, tc + 1)` of the component `c` (`:ret` or
`:less`) of a moving time step. It is a private function.
"""
@inline function _mrows(x::𝒻ᵐᵒᵛ, c::Symbol)
    return c == :ret ? x.ret.data : x.less.data
end
#

@inline function _mrows(v::𝒻ᵐᵒᵛView, c::Symbol)
    _check_view(v)
    A = c == :ret ? v.parent.ret : v.parent.less
    return _mrowview(A, v.delti)
end


"""
    _check_mrow(x, y)

Check that two moving time steps have the same cut-off and matrix size.
It is a private function.
"""
function _check_mrow(x, y)
    getsize(x) == getsize(y) ||
        throw(ArgumentError("the cut-offs differ: $(getsize(x)) and $(getsize(y))"))
    getdims(x) == getdims(y) ||
        throw(DimensionMismatch("the matrix sizes differ"))
    return nothing
end


