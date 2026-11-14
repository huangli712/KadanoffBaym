#
# Project : Lavender
# Source  : element.jl
# Author  : Li Huang (huangli@caep.cn)
# Status  : Unstable
#
# Last modified: 2026/11/10
#

"""
    set_matrixelement!(A::NessArray{T}, i₁::I64, i₂::I64, B::NessArray{T},
                       j₁::I64, j₂::I64, dom::FFTDomain)

Copy the matrix element `(j₁, j₂)` of `B` to the matrix element `(i₁, i₂)`
of `A` for all modes in the domain `dom`. The indices are 1-based.

*References* :

Please see [`NESSi`] `ness2_fft_impl.hpp`, `fft_array::set_matrixelement`.
"""
function set_matrixelement!(A::NessArray{T}, i₁::I64, i₂::I64, B::NessArray{T},
                            j₁::I64, j₂::I64, dom::FFTDomain) where {T}
    getnft(A) == getnft(B) ||
        throw(DimensionMismatch("A and B must have the same number of grid points"))
    X = _fft_data(A, dom)
    Y = _fft_data(B, dom)
    for k in axes(X, 3)
        X[i₁, i₂, k] = Y[j₁, j₂, k]
    end
    return A
end

"""
    set_matrixelement!(A::NessHermMat{T}, i₁::I64, i₂::I64,
                       B::NessHermMat{T}, j₁::I64, j₂::I64, dom::FFTDomain)

Copy the matrix element `(j₁, j₂)` of `B` to the element `(i₁, i₂)` of `A`
for both components. The indices are 1-based.
"""
function set_matrixelement!(A::NessHermMat{T}, i₁::I64, i₂::I64,
                            B::NessHermMat{T}, j₁::I64, j₂::I64,
                            dom::FFTDomain) where {T}
    set_matrixelement!(A.ret, i₁, i₂, B.ret, j₁, j₂, dom)
    set_matrixelement!(A.les, i₁, i₂, B.les, j₁, j₂, dom)
    return A
end

