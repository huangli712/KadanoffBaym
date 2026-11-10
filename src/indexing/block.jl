#
# Project : Lavender
# Source  : %s
# Author  : Li Huang (huangli@caep.cn)
# Status  : Unstable
#
# Last modified: 2026/11/10
#

#=
### *Private Service Functions* : *Contiguous Storage*
=#

#=
*Remarks* :

The four independent components of the full contour-ordered Green's
function are stored in contiguous arrays, with the orbital dimensions in
front, so that each block is a contiguous `ndim1 × ndim2` matrix.

* `Gᵐᵃᵗ`  : `data[:,:,m]`,   1 ≤ m ≤ ntau.
* `Gʳᵉᵗ`  : `data[:,:,k]`,   k = (i-1)i/2 + j for i ≥ j (lower triangle).
* `Gˡᵐⁱˣ` : `data[:,:,i,m]`, 1 ≤ i ≤ ntime, 1 ≤ m ≤ ntau.
* `Gˡᵉˢˢ` : `data[:,:,k]`,   k = (j-1)j/2 + i for i ≤ j (upper triangle).

The other triangles of `Gʳᵉᵗ` and `Gˡᵉˢˢ` are reconstructed by the
hermitian conjugate relations, see [`NESSi`] Eq.~(18a) and (20).
=#

"""
    _ntri(ntime::I64)

Number of blocks in a triangle (including the diagonal) of a square time
grid with `ntime` points.
"""
@inline _ntri(ntime::I64) = (ntime * (ntime + 1)) ÷ 2


"""
    _idx_ret(i::I64, j::I64)

Linear block index of G^{R}(tᵢ, tⱼ) in the lower triangle (i ≥ j).
"""
@inline _idx_ret(i::I64, j::I64) = ((i - 1) * i) ÷ 2 + j


"""
    _idx_less(i::I64, j::I64)

Linear block index of G^{<}(tᵢ, tⱼ) in the upper triangle (i ≤ j).
"""
@inline _idx_less(i::I64, j::I64) = ((j - 1) * j) ÷ 2 + i


"""
    _block(mat::Gᵐᵃᵗ{T}, i::I64)
    _block(ret::Gʳᵉᵗ{T}, i::I64, j::I64)
    _block(lmix::Gˡᵐⁱˣ{T}, i::I64, j::I64)
    _block(less::Gˡᵉˢˢ{T}, i::I64, j::I64)

Return a zero-copy (write-through) view of the stored block. Only the
independent storage domain is accepted (i ≥ j for `Gʳᵉᵗ`, i ≤ j for
`Gˡᵉˢˢ`). It is a private function.
"""
@inline function _block(mat::Gᵐᵃᵗ{T}, i::I64) where {T}
    @assert 1 ≤ i ≤ mat.ntau
    return view(mat.data, :, :, i)
end
#

@inline function _block(ret::Gʳᵉᵗ{T}, i::I64, j::I64) where {T}
    @assert 1 ≤ j ≤ i ≤ ret.ntime
    return view(ret.data, :, :, _idx_ret(i, j))
end
#

@inline function _block(lmix::Gˡᵐⁱˣ{T}, i::I64, j::I64) where {T}
    @assert 1 ≤ i ≤ lmix.ntime
    @assert 1 ≤ j ≤ lmix.ntau
    return view(lmix.data, :, :, i, j)
end
#

@inline function _block(less::Gˡᵉˢˢ{T}, i::I64, j::I64) where {T}
    @assert 1 ≤ i ≤ j ≤ less.ntime
    return view(less.data, :, :, _idx_less(i, j))
end


_mat_at(G::Union{ℱ,PseudoGF}, m::I64) = _block(G.mat, m)

_mat_at(G::Union{𝒻,PseudoGFStep}, m::I64) = G.mat.data[m]
#

_ret_at(G::Union{ℱ,PseudoGF}, tstp::I64, j::I64) = _block(G.ret, tstp, j)

_ret_at(G::Union{𝒻,PseudoGFStep}, tstp::I64, j::I64) = G.ret.data[j]
#

_lmix_at(G::Union{ℱ,PseudoGF}, tstp::I64, m::I64) = _block(G.lmix, tstp, m)

_lmix_at(G::Union{𝒻,PseudoGFStep}, tstp::I64, m::I64) = G.lmix.data[m]
#

_less_at(G::Union{ℱ,PseudoGF}, tstp::I64, i::I64) = _block(G.less, i, tstp)

_less_at(G::Union{𝒻,PseudoGFStep}, tstp::I64, i::I64) = G.less.data[i]


