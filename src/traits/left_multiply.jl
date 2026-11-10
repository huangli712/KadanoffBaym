#
# Project : Lavender
# Source  : left_multiply.jl
# Author  : Li Huang (huangli@caep.cn)
# Status  : Unstable
#
# Last modified: 2026/11/10
#

"""
    left_multiply!(A::FFTArray{T}, M::AbstractMatrix, dom::FFTDomain;
                   hermconj::Bool = false)

Replace the matrix of every mode in the domain `dom` by ``M A`` (or
``M^\\dagger A`` if `hermconj` is true).

*References* :

Please see [`NESSi`] `ness2_fft_decl.hpp`, `fft_array::left_multiply`.
"""
function left_multiply!(A::FFTArray{T}, M::AbstractMatrix, dom::FFTDomain;
                        hermconj::Bool = false) where {T}
    size(M) == getdims(A) ||
        throw(DimensionMismatch("M must have the size $(getdims(A))"))
    X = _fft_data(A, dom)
    L = hermconj ? Matrix{T}(M') : Matrix{T}(M)
    tmp = similar(L)
    for k in axes(X, 3)
        Z = view(X, :, :, k)
        mul!(tmp, L, Z)
        copyto!(Z, tmp)
    end
    return A
end

"""
    left_multiply!(G::HermMatrixNESS{T}, M::AbstractMatrix, dom::FFTDomain;
                   hermconj::Bool = false)

Left multiplication of the retarded and lesser components by `M` (or
``M^\\dagger``).
"""
function left_multiply!(G::HermMatrixNESS{T}, M::AbstractMatrix, dom::FFTDomain;
                        hermconj::Bool = false) where {T}
    left_multiply!(G.ret, M, dom; hermconj = hermconj)
    left_multiply!(G.les, M, dom; hermconj = hermconj)
    return G
end

