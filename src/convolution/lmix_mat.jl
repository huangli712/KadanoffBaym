#
# Project : Lavender
# Source  : %s
# Author  : Li Huang (huangli@caep.cn)
# Status  : Unstable
#
# Last modified: 2026/11/10
#

"""
    conv_lmix_mat(
        n::I64,
        C::Gˡᵐⁱˣ{T}, A::Gˡᵐⁱˣ{T}, B::Gᵐᵃᵗ{T},
        I::Integrator,
        beta::F64,
        sign::I64
    ) where {T}

Try to calculate the left-mixing component (C^⌉) of contour-ordered Green's
function (C) from convolution of two contour-ordered Green's functions
(A and B). Actually, it implements `C^⌉(t,τ) = A^⌉(t,τ') ∗ Bᴹ(τ'-τ)` at
time step `t = nh` for all `τ ∈ [0,β]`. That is to say, only the C₂ and
C₃ parts of C^⌉ are calculated.

### Arguments
* n -> Index for given time step.
* A -> Left-mixing component of contour-ordered Green's function, A^⌉(t,τ').
* B -> Retarded component of contour-ordered Green's function, Bᴹ(τ'-τ).
* I -> Struct for numerical integration.
* beta -> Inverse temperature, β.
* sign -> Set `sign = -1` for fermions or `sign = +1` for bosons.

### Returns
* C -> Left-mixing component of contour-ordered Green's function, C^⌉(t,τ).

See also: [`conv_ret_lmix`](@ref).
"""
function conv_lmix_mat(
    n::I64,
    C::Gˡᵐⁱˣ{T}, A::Gˡᵐⁱˣ{T}, B::Gᵐᵃᵗ{T},
    I::Integrator,
    beta::F64,
    sign::I64
) where {T}
    # Extract parameters
    ntime = getntime(A)
    ntau = getntau(A)

    # Sanity check
    @assert getntau(A) == getntau(B)
    @assert getntau(B) == getntau(C)
    @assert iscompatible(A, C)
    @assert ntime ≥ n ≥ 1
    @assert beta > 0.0
    @assert sign in (FERMI, BOSE)

    # Evaluate δτ
    δτ = beta / (ntau - 1)

    #
    # Evaluate the left-mixing convolution at a given time step
    #
    # It is very similar to computing the Matsubara convolution.
    #
    ws = _ConvWorkspace(A.data, B.data)
    _conv_lmix_mat_kernel!(_LmixRow(C, n), n, A, nothing, B, I, sign, δτ, ws)
end


"""
    conv_lmix_mat(
        n::I64,
        C::Gˡᵐⁱˣ{T}, A::Gˡᵐⁱˣ{T}, f₀::Element{T}, B::Gᵐᵃᵗ{T},
        I::Integrator,
        beta::F64,
        sign::I64
    ) where {T}

Try to calculate the left-mixing component (C^⌉) of contour-ordered Green's
function (C) from convolution of two contour-ordered Green's functions
(A and B) and a constant matrix (time-independent) f₀. Actually, it just
implements `C^⌉(t,τ) = A^⌉(t,τ') ∗ f₀ ∗ Bᴹ(τ'-τ)` at time step `t = nh`
for all `τ ∈ [0,β]`. That is to say, only the C₂ and C₃ parts of C^⌉ are
calculated.

### Arguments
* n -> Index for given time step.
* A -> Left-mixing component of contour-ordered Green's function, A^⌉(t,τ').
* f₀ -> Value of function f at τ = -iβ.
* B -> Retarded component of contour-ordered Green's function, Bᴹ(τ'-τ).
* I -> Struct for numerical integration.
* beta -> Inverse temperature, β.
* sign -> Set `sign = -1` for fermions or `sign = +1` for bosons.

### Returns
* C -> Left-mixing component of contour-ordered Green's function, C^⌉(t,τ).

### Notes
The argument `f₀` is assumed to be constant matrix.

See also: [`conv_ret_lmix`](@ref).
"""
function conv_lmix_mat(
    n::I64,
    C::Gˡᵐⁱˣ{T}, A::Gˡᵐⁱˣ{T}, f₀::Element{T}, B::Gᵐᵃᵗ{T},
    I::Integrator,
    beta::F64,
    sign::I64
) where {T}
    # Extract parameters
    ntime = getntime(A)
    ntau = getntau(A)

    # Sanity check
    @assert getntau(A) == getntau(B)
    @assert getntau(B) == getntau(C)
    @assert iscompatible(A, C)
    @assert ntime ≥ n ≥ 1
    @assert beta > 0.0
    @assert sign in (FERMI, BOSE)

    # Evaluate δτ
    δτ = beta / (ntau - 1)

    #
    # Evaluate the left-mixing convolution at a given time step
    #
    # f₀ = f(-iβ) is applied block by block, B is not copied.
    #
    ws = _ConvWorkspace(A.data, B.data)
    _conv_lmix_mat_kernel!(_LmixRow(C, n), n, A, f₀, B, I, sign, δτ, ws)
end


"""
    _conv_lmix_mat_kernel!(tgt, n, A, f₀, B, I, sign, s, ws)

Accumulate `s * [A^⌉ ∗ f₀ ∗ Bᴹ](tₙ,τₘ)` into `tgt[m]` for all `m`. See
[NESSi] Eqs. (113) - (116). It is a private function.
"""
function _conv_lmix_mat_kernel!(
    tgt, n::I64,
    A::Gˡᵐⁱˣ{T},
    f₀,
    B::Gᵐᵃᵗ{T},
    I::Integrator,
    sign::I64,
    s::Number,
    ws::_ConvWorkspace{T};
    ms::UnitRange{I64} = 1:A.ntau
) where {T}
    _conv_lmix_mat_loop!(tgt, n, A, _fbmat!(ws, f₀, B), I, sign, s, ms)
end


"""
    _conv_lmix_mat_loop!(tgt, n, A, Bt, I, sign, s)

The integration loop of [`_conv_lmix_mat_kernel!`](@ref). Here `Bt[m]`
returns the block `f₀ Bᴹ(τₘ)`. It is a private function.
"""
function _conv_lmix_mat_loop!(tgt, n::I64, A::Gˡᵐⁱˣ, Bt, I::Integrator,
                              sign::I64, s::Number,
                              ms::UnitRange{I64} = 1:A.ntau)
    k = I.k
    ntau = A.ntau

    for m in ms
        c = tgt[m]

        # Try to calculate the contributions from 0 to τ
        #
        # See [NESSi] Eq. (113) - (114)
        #
        if m == 1
            # PASS
        elseif m < k + 1 # Strange boundary correction
            for j = 1:k+1
                indb = ntau
                for l = 1:k+1
                    w = I.BCW[m-2,l-1,j-1]
                    mul!(c, _block(A, n, j), Bt[indb], sign * s * w, true)
                    indb = indb - 1
                end
            end
        else # Usual Gregory integration
            inda = m
            indb = ntau
            for l = 1:m
                w = I.GIW[m-1,l-1]
                mul!(c, _block(A, n, inda), Bt[indb], sign * s * w, true)
                inda = inda - 1
                indb = indb - 1
            end
        end

        # Try to calculate the contributions from τ to β
        #
        # See [NESSi] Eq. (115) - (116)
        #
        if m == ntau
            # PASS
        elseif m > ntau - k # Strange boundary correction
            inda = ntau
            for l = 1:k+1
                for j = 1:k+1
                    w = I.BCW[ntau-m-1,l-1,j-1]
                    mul!(c, _block(A, n, inda), Bt[j], s * w, true)
                end
                inda = inda - 1
            end
        elseif m > ntau - 2*k - 1 # Usual Gregory integration
            inda = m
            for l = 1:ntau-m+1
                w = I.GIW[ntau-m,l-1]
                mul!(c, _block(A, n, inda), Bt[l], s * w, true)
                inda = inda + 1
            end
        else # Usual Gregory integration
            inda = m
            indb = 1
            for l = m:ntau
                w = I.GIW[ntau-m,ntau-l]
                mul!(c, _block(A, n, inda), Bt[indb], s * w, true)
                inda = inda + 1
                indb = indb + 1
            end
        end
    end
end


