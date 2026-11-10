#
# Project : Lavender
# Source  : lmix_rmix.jl
# Author  : Li Huang (huangli@caep.cn)
# Status  : Unstable
#
# Last modified: 2026/11/10
#

"""
    conv_lmix_rmix(
        n::I64,
        C::Gˡᵉˢˢ{T},
        A::Gˡᵐⁱˣ{T}, Acc::Gˡᵐⁱˣ{T},
        B::Gˡᵐⁱˣ{T}, Bcc::Gˡᵐⁱˣ{T},
        I::Integrator,
        beta::F64,
        sign::I64
    ) where {T}

Try to calculate the lesser component (C^<) of contour-ordered Green's
function (C) from convolution of two contour-ordered Green's functions
(A and B). Actually, it implements `C^<(t,t') = A^⌉(t,τ) ∗ B^⌈(τ,t')` at
time step `t' = nh` for all `t` where `t ≤ t'`. That is to say, only the
C₃ part of C^< is calculated.

### Arguments
* n -> Index for given time step.
* A -> Left-mixing component of contour-ordered Green's function, A^⌉(t,τ).
* Acc -> Hermitian conjugate partner of A.
* B -> Left-mixing component of contour-ordered Green's function, B^⌉(t',τ).
* Bcc -> Hermitian conjugate partner of B.
* I -> Struct for numerical integration.
* beta -> Inverse temperature, β.
* sign -> Set `sign = -1` for fermions or `sign = +1` for bosons.

### Returns
* C -> Less component of contour-ordered Green's function, C^<(t,t').

### Notes
The right-mixing component of contour-ordered Green's function, B^⌈(τ,t'),
is not ready. It should be calculated from the corresponding left-mixing
component, B^⌉(t',τ).

See also: [`convolution_time_step`](@ref).
"""
function conv_lmix_rmix(
    n::I64,
    C::Gˡᵉˢˢ{T},
    A::Gˡᵐⁱˣ{T}, Acc::Gˡᵐⁱˣ{T},
    B::Gˡᵐⁱˣ{T}, Bcc::Gˡᵐⁱˣ{T},
    I::Integrator,
    beta::F64,
    sign::I64
) where {T}
    # Extract parameters
    ntime = getntime(A)
    ntau = getntau(A)

    # Sanity check
    @assert getntime(A) == getntime(C)
    @assert iscompatible(A, B)
    @assert iscompatible(A, Acc)
    @assert iscompatible(B, Bcc)
    @assert ntime ≥ n ≥ 1
    @assert beta > 0.0
    @assert sign in (FERMI, BOSE)

    # Evaluate δτ
    δτ = beta / (ntau - 1)

    #
    # Evaluate the lesser convolution at a given time step
    #
    # See [NESSi] Eq. (121) - (122)
    #
    # For the contributions from C₁ and C₂, see conv_ret_less() and
    # conv_less_adv() please.
    #
    ws = _ConvWorkspace(A.data, B.data)
    _conv_lmix_rmix_kernel!(_LessCol(C, n), 1:n, n, A, nothing, Bcc, I, sign, -im * δτ, ws)
end

"""
    conv_lmix_rmix(
        n::I64,
        C::Gˡᵉˢˢ{T},
        A::Gˡᵐⁱˣ{T}, Acc::Gˡᵐⁱˣ{T},
        f₀::Element{T},
        B::Gˡᵐⁱˣ{T}, Bcc::Gˡᵐⁱˣ{T},
        I::Integrator,
        beta::F64,
        sign::I64
    ) where {T}

Try to calculate the lesser component (C^<) of contour-ordered Green's
function (C) from convolution of two contour-ordered Green's functions
(A and B) and a constant matrix (time-independent) f₀. Actually, it just
implements `C^<(t,t') = A^⌉(t,τ) ∗ f₀ ∗ B^⌈(τ,t')` at time step `t' = nh`
for all `t` where `t ≤ t'`. That is to say, only the C₃ part of C^< is
calculated.

### Arguments
* n -> Index for given time step.
* A -> Left-mixing component of contour-ordered Green's function, A^⌉(t,τ).
* Acc -> Hermitian conjugate partner of A.
* f₀ -> Value of function f at τ = -iβ.
* B -> Left-mixing component of contour-ordered Green's function, B^⌉(t',τ).
* Bcc -> Hermitian conjugate partner of B.
* I -> Struct for numerical integration.
* beta -> Inverse temperature, β.
* sign -> Set `sign = -1` for fermions or `sign = +1` for bosons.

### Returns
* C -> Less component of contour-ordered Green's function, C^<(t,t').

### Notes
The right-mixing component of contour-ordered Green's function, B^⌈(τ,t'),
is not ready. It should be calculated from the corresponding left-mixing
component, B^⌉(t',τ).

The argument `f₀` is assumed to be constant matrix.

See also: [`convolution_time_step`](@ref).
"""
function conv_lmix_rmix(
    n::I64,
    C::Gˡᵉˢˢ{T},
    A::Gˡᵐⁱˣ{T}, Acc::Gˡᵐⁱˣ{T},
    f₀::Element{T},
    B::Gˡᵐⁱˣ{T}, Bcc::Gˡᵐⁱˣ{T},
    I::Integrator,
    beta::F64,
    sign::I64
) where {T}
    # Extract parameters
    ntime = getntime(A)
    ntau = getntau(A)

    # Sanity check
    @assert getntime(A) == getntime(C)
    @assert iscompatible(A, B)
    @assert iscompatible(A, Acc)
    @assert iscompatible(B, Bcc)
    @assert ntime ≥ n ≥ 1
    @assert beta > 0.0
    @assert sign in (FERMI, BOSE)

    # Evaluate δτ
    δτ = beta / (ntau - 1)

    #
    # Evaluate the lesser convolution at a given time step
    #
    # See [NESSi] Eq. (121) - (122)
    #
    # For the contributions from C₁ and C₂, see conv_ret_less() and
    # conv_less_adv() please.
    #
    ws = _ConvWorkspace(A.data, B.data)
    _conv_lmix_rmix_kernel!(_LessCol(C, n), 1:n, n, A, f₀, Bcc, I, sign, -im * δτ, ws)
end

"""
    _conv_lmix_rmix_kernel!(tgt, ms, n, A, f₀, Bcc, I, sign, s, ws)

Accumulate `s * [A^⌉ ∗ f₀ ∗ B^⌈](tₘ,tₙ) / (-i)` into `tgt[m]` for `m ∈ ms`,
i.e., the factor `-i δτ` should be included in `s`. Here
`ms ⊆ 1:max(n,k+1)`. See [NESSi] Eqs. (121) - (122). It is a private
function.
"""
function _conv_lmix_rmix_kernel!(
    tgt, ms::UnitRange{I64}, n::I64,
    A::Gˡᵐⁱˣ{T},
    f₀,
    Bcc::Gˡᵐⁱˣ{T},
    I::Integrator,
    sign::I64,
    s::Number,
    ws::_ConvWorkspace{T}
) where {T}
    ntau = A.ntau
    @assert isempty(ms) || (first(ms) ≥ 1 && last(ms) ≤ max(n - 1, I.k) + 1)

    for j = 1:ntau
        w = I.GIW[ntau-1,j-1]
        # The right-mixing component f₀ B^⌈(τⱼ,tₙ), see [NESSi] Eq. (18d)
        b = _f0b!(ws, f₀, adjoint(_block(Bcc, n, ntau - j + 1)))
        _conv_lmix_rmix_col!(tgt, ms, j, b, -sign * s * w, A)
    end
end

"""
    _conv_lmix_rmix_col!(tgt, ms, j, b, α, A)

Accumulate the `j`-th term of [NESSi] Eq. (122). It is a private function.
"""
function _conv_lmix_rmix_col!(tgt, ms::UnitRange{I64}, j::I64, b, α::Number,
                              A::Gˡᵐⁱˣ)
    for m in ms
        mul!(tgt[m], _block(A, m, j), b, α, true)
    end
end

