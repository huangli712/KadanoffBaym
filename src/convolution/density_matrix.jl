#
# Project : Lavender
# Source  : %s
# Author  : Li Huang (huangli@caep.cn)
# Status  : Unstable
#
# Last modified: 2026/11/10
#

"""
    convolution_density_matrix(
        tstp::I64,
        A::_ConvGF{T}, Acc::_ConvGF{T},
        f::Union{Cf{T},Nothing},
        B::_ConvGF{T}, Bcc::_ConvGF{T},
        I::Integrator,
        beta::F64,
        h::F64
    ) where {T}

Calculate the density matrix of the convolution `A ∗ f ∗ B` in the
convention of `NESSi`, i.e., `-i [A ∗ f ∗ B]^<(t,t)` at time step
`tstp ≥ 1`, or `-[A ∗ f ∗ B]ᴹ(β)` if `tstp = 0`. Note that it is not the
same as [`density`](@ref).

### Arguments
See [`convolution_les_timediag`](@ref).

### Returns
* x -> An `Element{T}` object.
"""
function convolution_density_matrix(
    tstp::I64,
    A::_ConvGF{T}, Acc::_ConvGF{T},
    f::Union{Cf{T},Nothing},
    B::_ConvGF{T}, Bcc::_ConvGF{T},
    I::Integrator,
    beta::F64,
    h::F64
) where {T}
    x = convolution_les_timediag(tstp, A, Acc, f, B, Bcc, I, beta, h)
    x .= (-im) .* x
    return x
end


"""
    convolution_density_matrix(
        tstp::I64,
        A::_ConvGF{T}, f::Union{Cf{T},Nothing}, B::_ConvGF{T},
        I::Integrator,
        beta::F64,
        h::F64
    ) where {T}

Hermitian version (`Acc = A`, `Bcc = B`) of the density matrix
convolution.
"""
function convolution_density_matrix(
    tstp::I64,
    A::_ConvGF{T}, f::Union{Cf{T},Nothing}, B::_ConvGF{T},
    I::Integrator,
    beta::F64,
    h::F64
) where {T}
    convolution_density_matrix(tstp, A, A, f, B, B, I, beta, h)
end


"""
    convolution_density_matrix(
        tstp::I64,
        A::_ConvGF{T}, B::_ConvGF{T},
        I::Integrator,
        beta::F64,
        h::F64
    ) where {T}

Hermitian version (`Acc = A`, `Bcc = B`, `f = 1`) of the density matrix
convolution.
"""
function convolution_density_matrix(
    tstp::I64,
    A::_ConvGF{T}, B::_ConvGF{T},
    I::Integrator,
    beta::F64,
    h::F64
) where {T}
    convolution_density_matrix(tstp, A, A, nothing, B, B, I, beta, h)
end

"""
    convolution_density_matrix(A::HermMatrixNESS{C64}, B::HermMatrixNESS{C64},
                               sign::Integer)

Density matrix ``\\xi i C^<(0)`` of the steady-state convolution ``C = A
\\ast B``, with ``C^<(\\omega) = A^<(\\omega) B^A(\\omega) + A^R(\\omega)
B^<(\\omega)``. `A` and `B` are transformed to the frequency domain by the
trapezoidal rule on local copies (NESSi transforms the inputs in place),
the inputs are not changed.

### Arguments
* A    -> Steady-state Green's function, the time data is the input.
* B    -> Steady-state Green's function, the time data is the input.
* sign -> FERMI or BOSE.

### Returns
* ρ -> The density matrix.

*References* :

Please see [`NESSi`] `ness2_utils_impl.hpp`, `convolution_density_matrix`.
NESSi checks the shapes with `&&`, here any mismatch is an error.
"""
function convolution_density_matrix(A::HermMatrixNESS{C64}, B::HermMatrixNESS{C64},
                                    sign::Integer)
    sign in (FERMI, BOSE) || throw(ArgumentError("sign must be FERMI or BOSE"))
    iscompatible(A, B) ||
        throw(DimensionMismatch("convolution_density_matrix: input mismatch"))
    Ac = deepcopy(A)
    Bc = deepcopy(B)
    integral_transform_to_freq!(Ac, FFT_TRAPEZ)
    integral_transform_to_freq!(Bc, FFT_TRAPEZ)
    d = getdims(A)[1]
    C = zeros(C64, d, d)
    for k = 1:getnft(A)
        Al = view(Ac.les.freq, :, :, k)
        Ar = view(Ac.ret.freq, :, :, k)
        Bl = view(Bc.les.freq, :, :, k)
        Br = view(Bc.ret.freq, :, :, k)
        C .+= Al * Br' .+ Ar * Bl
    end
    return C .* complex(0.0, sign * A.grid.dw * 0.5 / π)
end


