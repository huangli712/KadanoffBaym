#
# Project : Lavender
# Source  : rmul.jl
# Author  : Li Huang (huangli@caep.cn)
# Status  : Unstable
#
# Last modified: 2026/11/10
#

"""
    right_multiply!(A::FFTArray{T}, M::AbstractMatrix, dom::FFTDomain;
                    hermconj::Bool = false)

Replace the matrix of every mode in the domain `dom` by ``A M`` (or
``A M^\\dagger`` if `hermconj` is true).

*References* :

Please see [`NESSi`] `ness2_fft_decl.hpp`, `fft_array::right_multiply`.
"""
function right_multiply!(A::FFTArray{T}, M::AbstractMatrix, dom::FFTDomain;
                         hermconj::Bool = false) where {T}
    size(M) == getdims(A) ||
        throw(DimensionMismatch("M must have the size $(getdims(A))"))
    X = _fft_data(A, dom)
    R = hermconj ? Matrix{T}(M') : Matrix{T}(M)
    tmp = similar(R)
    for k in axes(X, 3)
        Z = view(X, :, :, k)
        mul!(tmp, Z, R)
        copyto!(Z, tmp)
    end
    return A
end

"""
    right_multiply!(G::HermMatrixNESS{T}, M::AbstractMatrix, dom::FFTDomain;
                    hermconj::Bool = false)

Right multiplication of the retarded and lesser components by `M` (or
``M^\\dagger``).
"""
function right_multiply!(G::HermMatrixNESS{T}, M::AbstractMatrix, dom::FFTDomain;
                         hermconj::Bool = false) where {T}
    right_multiply!(G.ret, M, dom; hermconj = hermconj)
    right_multiply!(G.les, M, dom; hermconj = hermconj)
    return G
end

