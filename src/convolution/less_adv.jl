#
# Project : Lavender
# Source  : less_adv.jl
# Author  : Li Huang (huangli@caep.cn)
# Status  : Unstable
#
# Last modified: 2026/11/10
#

"""
    conv_less_adv(
        n::I64,
        C::Gˡᵉˢˢ{T},
        A::Gˡᵉˢˢ{T}, Acc::Gˡᵉˢˢ{T},
        B::Gʳᵉᵗ{T}, Bcc::Gʳᵉᵗ{T},
        I::Integrator,
        h::F64
    ) where {T}

Try to calculate the lesser component (C^<) of contour-ordered Green's
function (C) from convolution of two contour-ordered Green's functions
(A and B). Actually, it implements `C^<(t,t') = A^<(t,t'') ∗ Bᴬ(t'',t')`
at time step `t' = nh` for all `t` where `t ≤ t'`. That is to say, only
the C₂ part of C^< is calculated.

### Arguments
* n -> Index for given time step.
* A -> Lesser component of contour-ordered Green's function, A^<(t,t'').
* Acc -> Hermitian conjugate partner of A.
* B -> Retarded component of contour-ordered Green's function, Bᴿ(t',t'').
* Bcc -> Hermitian conjugate partner of B.
* I -> Struct for numerical integration.
* h -> Time step interval.

### Returns
* C -> Lesser component of contour-ordered Green's function, C^<(t,t').

### Notes
The advanced component of contour-ordered Green's function, Bᴬ(t'',t), is
not ready. It should be calculated from the corresponding lesser
component, Bᴿ(t',t'').

See also: [`convolution_time_step`](@ref).
"""
function conv_less_adv(
    n::I64,
    C::Gˡᵉˢˢ{T},
    A::Gˡᵉˢˢ{T}, Acc::Gˡᵉˢˢ{T},
    B::Gʳᵉᵗ{T}, Bcc::Gʳᵉᵗ{T},
    I::Integrator,
    h::F64
) where {T}
    # Extract parameters
    ntime = getntime(A)

    # Sanity check
    @assert getntime(A) == getntime(B)
    @assert getntime(B) == getntime(C)
    @assert iscompatible(A, Acc)
    @assert iscompatible(B, Bcc)
    @assert ntime ≥ n ≥ 1
    @assert h > 0.0

    #
    # Evaluate the lesser convolution at a given time step
    #
    # See [NESSi] Eq. (119) - (120)
    #
    # For the contributions from C₁ and C₃, see conv_ret_less() and
    # conv_lmix_rmix() please.
    #
    ws = _ConvWorkspace(A.data, B.data)
    _conv_less_adv_kernel!(_LessCol(C, n), 1:n, n, A, Acc, nothing, B, Bcc, I, h, ws)
end


"""
    conv_less_adv(
        n::I64,
        C::Gˡᵉˢˢ{T},
        A::Gˡᵉˢˢ{T}, Acc::Gˡᵉˢˢ{T},
        fₜ::VecArray{T},
        B::Gʳᵉᵗ{T}, Bcc::Gʳᵉᵗ{T},
        I::Integrator,
        h::F64
    ) where {T}

Try to calculate the lesser component (C^<) of contour-ordered Green's
function (C) from convolution of two contour-ordered Green's functions
(A and B) and a time-dependent function fₜ. Actually, it just implements
`C^<(t,t') = A^<(t,t'') ∗ f(t'') ∗ Bᴬ(t'',t')` at time step `t' = nh`
for all `t` where `t ≤ t'`. That is to say, only the C₂ part of C^< is
calculated.

### Arguments
* n -> Index for given time step.
* A -> Lesser component of contour-ordered Green's function, A^<(t,t'').
* Acc -> Hermitian conjugate partner of A.
* fₜ -> Values of time-dependent function, f(t'').
* B -> Retarded component of contour-ordered Green's function, Bᴿ(t',t'').
* Bcc -> Hermitian conjugate partner of B.
* I -> Struct for numerical integration.
* h -> Time step interval.

### Returns
* C -> Lesser component of contour-ordered Green's function, C^<(t,t').

### Notes
The advanced component of contour-ordered Green's function, Bᴬ(t'',t), is
not ready. Thus, it should be calculated from the corresponding lesser
component, Bᴿ(t',t'').

See also: [`convolution_time_step`](@ref).
"""
function conv_less_adv(
    n::I64,
    C::Gˡᵉˢˢ{T},
    A::Gˡᵉˢˢ{T}, Acc::Gˡᵉˢˢ{T},
    fₜ::VecArray{T},
    B::Gʳᵉᵗ{T}, Bcc::Gʳᵉᵗ{T},
    I::Integrator,
    h::F64
) where {T}
    # Extract parameters
    ntime = getntime(A)
    k = I.k

    # Sanity check
    @assert getntime(A) == getntime(B)
    @assert getntime(B) == getntime(C)
    @assert iscompatible(A, Acc)
    @assert iscompatible(B, Bcc)
    @assert length(fₜ) ≥ n
    @assert length(fₜ) ≥ k + 1
    @assert ntime ≥ n ≥ 1
    @assert h > 0.0

    #
    # Evaluate the lesser convolution at a given time step
    #
    # See [NESSi] Eq. (119) - (120)
    #
    # For the contributions from C₁ and C₃, see conv_ret_less() and
    # conv_lmix_rmix() please.
    #
    ws = _ConvWorkspace(A.data, B.data)
    _conv_less_adv_kernel!(_LessCol(C, n), 1:n, n, A, Acc, fₜ, B, Bcc, I, h, ws)
end


"""
    _conv_less_adv_kernel!(tgt, ms, n, A, Acc, f, B, Bcc, I, s, ws)

Accumulate `s * [A^< ∗ f ∗ Bᴬ](tₘ,tₙ)` into `tgt[m]` for `m ∈ ms`. Here
`ms ⊆ 1:max(n,k+1)`. See [NESSi] Eqs. (119) - (120). It is a private
function.
"""
function _conv_less_adv_kernel!(
    tgt, ms::UnitRange{I64}, n::I64,
    A::Gˡᵉˢˢ{T}, Acc::Gˡᵉˢˢ{T},
    f,
    B::Gʳᵉᵗ{T}, Bcc::Gʳᵉᵗ{T},
    I::Integrator,
    s::Number,
    ws::_ConvWorkspace{T};
    diagcc::Bool = false
) where {T}
    n₁ = max(n - 1, I.k) + 1
    @assert isempty(ms) || (first(ms) ≥ 1 && last(ms) ≤ n₁)

    for j = 1:n₁
        w = I.GIW[n-1,j-1]
        # The advanced component f(tⱼ) Bᴬ(tⱼ,tₙ), see [NESSi] Eq. (18a)
        if j ≤ n
            _conv_less_adv_col!(tgt, ms, j, _fb!(ws, f, j, adjoint(_block(Bcc, n, j))),
                                s * w, A, Acc, diagcc)
        else
            _conv_less_adv_col!(tgt, ms, j, _fb!(ws, f, j, _block(B, j, n)),
                                -s * w, A, Acc, diagcc)
        end
    end
end


"""
    _conv_less_adv_col!(tgt, ms, j, b, α, A, Acc, diagcc = false)

Accumulate the `j`-th term of [NESSi] Eq. (120). If `diagcc = true`, the
diagonal block ``A^{<}(t_m,t_m)`` is replaced by ``-Acc^{<}(t_m,t_m)^{\\dagger}``,
which follows `incr_convolution_les` of NESSi for the early time steps.
It is a private function.
"""
function _conv_less_adv_col!(tgt, ms::UnitRange{I64}, j::I64, b, α::Number,
                             A::Gˡᵉˢˢ, Acc::Gˡᵉˢˢ, diagcc::Bool = false)
    for m in ms
        if j < m || (diagcc && j == m)
            mul!(tgt[m], adjoint(_block(Acc, j, m)), b, -α, true)
        else
            mul!(tgt[m], _block(A, m, j), b, α, true)
        end
    end
end


