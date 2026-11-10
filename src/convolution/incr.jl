#
# Project : Lavender
# Source  : %s
# Author  : Li Huang (huangli@caep.cn)
# Status  : Unstable
#
# Last modified: 2026/11/10
#

#=
### *Incremental Convolution* : *Public API*
=#

"""
    incr_convolution_mat!(
        C::Union{ℱ{T},𝒻{T}},
        A::ℱ{T}, f::Union{Cf{T},Nothing}, B::ℱ{T},
        I::Integrator,
        beta::F64,
        alpha::Number
    ) where {T}

Increment the Matsubara component of `C` by `alpha * [A ∗ f ∗ B]ᴹ`. Here
`f[0]` is used. If `f = nothing`, then `f = 1`.

### Arguments
* C     -> Target, a `ℱ` or `𝒻` object.
* A     -> Contour-ordered Green's function, A(t,t'').
* f     -> Contour function, f(t''), or `nothing`.
* B     -> Contour-ordered Green's function, B(t'',t').
* I     -> Struct for numerical integration.
* beta  -> Inverse temperature, β.
* alpha -> Weight.

### Returns
N/A
"""
function incr_convolution_mat!(
    C::Union{ℱ{T},𝒻{T}},
    A::ℱ{T}, f::Union{Cf{T},Nothing}, B::ℱ{T},
    I::Integrator,
    beta::F64,
    alpha::Number
) where {T}
    # Sanity check
    _check_incr(0, C, A, A, f, B, B; target = ())
    @assert beta > 0.0

    # Evaluate [A ∗ f ∗ B]ᴹ
    ndim1, ndim2 = getdims(C)
    ntau = getntau(A)
    tmp = Gᵐᵃᵗ(ntau, ndim1, ndim2, zero(T))
    if f isa Cf
        conv_mat(tmp, A.mat, f[0], B.mat, I, beta, A.sign)
    else
        conv_mat(tmp, A.mat, B.mat, I, beta, A.sign)
    end

    # Cᴹ ← Cᴹ + α [A ∗ f ∗ B]ᴹ
    tgt = _mat_target(C, 0)
    for m = 1:ntau
        tgt[m] .= tgt[m] .+ alpha .* _block(tmp, m)
    end
end


"""
    incr_convolution_ret!(
        tstp::I64,
        C::_ConvTarget{T},
        A::_ConvGF{T}, Acc::_ConvGF{T},
        f::Union{Cf{T},Nothing},
        B::_ConvGF{T}, Bcc::_ConvGF{T},
        I::Integrator,
        h::F64,
        alpha::Number
    ) where {T}

Increment the retarded component of `C` at time step `tstp` by
`alpha * [A ∗ f ∗ B]ᴿ(t,t')`, `t' ≤ t`.

### Arguments
* tstp  -> Index of time step, tstp ≥ 1.
* C     -> Target, a `ℱ` or `𝒻` object.
* A     -> Contour-ordered Green's function, A(t,t'').
* Acc   -> Hermitian conjugate partner of A.
* f     -> Contour function, f(t''), or `nothing`.
* B     -> Contour-ordered Green's function, B(t'',t').
* Bcc   -> Hermitian conjugate partner of B.
* I     -> Struct for numerical integration.
* h     -> Time step interval.
* alpha -> Weight.

### Returns
N/A
"""
function incr_convolution_ret!(
    tstp::I64,
    C::_ConvTarget{T},
    A::_ConvGF{T}, Acc::_ConvGF{T},
    f::Union{Cf{T},Nothing},
    B::_ConvGF{T}, Bcc::_ConvGF{T},
    I::Integrator,
    h::F64,
    alpha::Number
) where {T}
    # Sanity check
    _check_incr(tstp, C, A, Acc, f, B, Bcc; target = (:ret,))
    @assert tstp ≥ 1
    @assert h > 0.0

    ws = _ConvWorkspace(A, B)
    _incr_convolution_ret!(tstp, C, A, Acc, f, B, Bcc, I, h, alpha, ws)
end

#
# Workspace-bearing version of incr_convolution_ret!(). It is a private
# function.
#

function _incr_convolution_ret!(tstp::I64, C, A::_ConvGF{T}, Acc::_ConvGF{T}, f,
                                B::_ConvGF{T}, Bcc::_ConvGF{T}, I::Integrator, h::F64,
                                alpha::Number, ws::_ConvWorkspace{T}) where {T}
    tgt = _ret_target(C, tstp)
    _conv_ret_kernel!(tgt, tstp, A.ret, Acc.ret, f, B.ret, Bcc.ret, I, alpha * h, ws)
end


"""
    incr_convolution_tv!(
        tstp::I64,
        C::Union{ℱ{T},𝒻{T}},
        A::ℱ{T}, Acc::ℱ{T},
        f::Union{Cf{T},Nothing},
        B::ℱ{T}, Bcc::ℱ{T},
        I::Integrator,
        beta::F64,
        h::F64,
        alpha::Number
    ) where {T}

Increment the left-mixing component of `C` at time step `tstp` by
`alpha * [A ∗ f ∗ B]^⌉(t,τ)`. The real-time part uses `f(t'')`, while the
imaginary-time part uses `f[0]`.

### Arguments
* tstp  -> Index of time step, tstp ≥ 1.
* C     -> Target, a `ℱ` or `𝒻` object.
* A     -> Contour-ordered Green's function, A(t,t'').
* Acc   -> Hermitian conjugate partner of A.
* f     -> Contour function, f(t''), or `nothing`.
* B     -> Contour-ordered Green's function, B(t'',t').
* Bcc   -> Hermitian conjugate partner of B.
* I     -> Struct for numerical integration.
* beta  -> Inverse temperature, β.
* h     -> Time step interval.
* alpha -> Weight.

### Returns
N/A
"""
function incr_convolution_tv!(
    tstp::I64,
    C::Union{ℱ{T},𝒻{T}},
    A::ℱ{T}, Acc::ℱ{T},
    f::Union{Cf{T},Nothing},
    B::ℱ{T}, Bcc::ℱ{T},
    I::Integrator,
    beta::F64,
    h::F64,
    alpha::Number
) where {T}
    # Sanity check
    _check_incr(tstp, C, A, Acc, f, B, Bcc; target = (:lmix,))
    @assert tstp ≥ 1
    @assert beta > 0.0
    @assert h > 0.0

    ws = _ConvWorkspace(A, B)
    _incr_convolution_tv!(tstp, C, A, Acc, f, B, Bcc, I, beta, h, alpha, ws)
end

#
# Workspace-bearing version of incr_convolution_tv!(). It is a private
# function.
#

function _incr_convolution_tv!(tstp::I64, C, A::ℱ{T}, Acc::ℱ{T}, f,
                               B::ℱ{T}, Bcc::ℱ{T}, I::Integrator,
                               beta::F64, h::F64,
                               alpha::Number, ws::_ConvWorkspace{T}) where {T}
    δτ = beta / (getntau(A) - 1)
    tgt = _lmix_target(C, tstp)
    _conv_ret_lmix_kernel!(tgt, tstp, A.ret, Acc.ret, f, B.lmix, I, alpha * h, ws)
    _conv_lmix_mat_kernel!(tgt, tstp, A.lmix, _f0(f), B.mat, I, A.sign, alpha * δτ, ws)
end


#=
*Remarks* : *Partial Lesser Convolutions*

The lesser component of `A ∗ f ∗ B` consists of three terms:

```math
\begin{equation}
C^{<} = A^{R} \ast f \ast B^{<} + A^{<} \ast f \ast B^{A}
      + A^{\rceil} \ast f \ast B^{\lceil}.
\end{equation}
```

They are evaluated by `incr_convolution_ret_less!`,
`incr_convolution_less_adv!`, and `incr_convolution_lmix_rmix!`,
respectively. In the time stepping of the Dyson and VIE2 equations, only
the latter two terms enter the source term, while the first one belongs
to the implicit linear system. None of them reads the target `C`.

*References* :

Please see [`NESSi`] `cntr_convolution_impl.hpp`,
`convolution_timestep_les_retles`, `convolution_timestep_les_lesadv`, and
`convolution_timestep_les_tvvt`.
=#

"""
    incr_convolution_ret_less!(
        tstp::I64,
        C::_ConvTarget{T},
        A::_ConvGF{T}, Acc::_ConvGF{T},
        f::Union{Cf{T},Nothing},
        B::_ConvGF{T}, Bcc::_ConvGF{T},
        I::Integrator,
        beta::F64,
        h::F64,
        alpha::Number
    ) where {T}

Increment `C^<(t',t)` (`t' ≤ t`, `t` is given by `tstp`) by
`alpha * [Aᴿ ∗ f ∗ B^<](t',t)`. The argument `beta` is not used. It is
kept for an unified interface.

### Arguments
See [`incr_convolution_tv!`](@ref).

### Returns
N/A
"""
function incr_convolution_ret_less!(
    tstp::I64,
    C::_ConvTarget{T},
    A::_ConvGF{T}, Acc::_ConvGF{T},
    f::Union{Cf{T},Nothing},
    B::_ConvGF{T}, Bcc::_ConvGF{T},
    I::Integrator,
    beta::F64,
    h::F64,
    alpha::Number
) where {T}
    # Sanity check
    _check_incr(tstp, C, A, Acc, f, B, Bcc; target = (:less,))
    @assert tstp ≥ 1
    @assert h > 0.0

    ws = _ConvWorkspace(A, B)
    _incr_ret_less!(_less_target(C, tstp), 1:tstp, tstp,
                    A, Acc, f, B, Bcc, I, beta, h, alpha, ws)
end


"""
    incr_convolution_less_adv!(
        tstp::I64,
        C::_ConvTarget{T},
        A::_ConvGF{T}, Acc::_ConvGF{T},
        f::Union{Cf{T},Nothing},
        B::_ConvGF{T}, Bcc::_ConvGF{T},
        I::Integrator,
        beta::F64,
        h::F64,
        alpha::Number
    ) where {T}

Increment `C^<(t',t)` (`t' ≤ t`, `t` is given by `tstp`) by
`alpha * [A^< ∗ f ∗ Bᴬ](t',t)`. The argument `beta` is not used. It is
kept for an unified interface.

### Arguments
See [`incr_convolution_tv!`](@ref).

### Returns
N/A
"""
function incr_convolution_less_adv!(
    tstp::I64,
    C::_ConvTarget{T},
    A::_ConvGF{T}, Acc::_ConvGF{T},
    f::Union{Cf{T},Nothing},
    B::_ConvGF{T}, Bcc::_ConvGF{T},
    I::Integrator,
    beta::F64,
    h::F64,
    alpha::Number
) where {T}
    # Sanity check
    _check_incr(tstp, C, A, Acc, f, B, Bcc; target = (:less,))
    @assert tstp ≥ 1
    @assert h > 0.0

    ws = _ConvWorkspace(A, B)
    _incr_less_adv!(_less_target(C, tstp), 1:tstp, tstp,
                    A, Acc, f, B, Bcc, I, beta, h, alpha, ws)
end


"""
    incr_convolution_lmix_rmix!(
        tstp::I64,
        C::_ConvTarget{T},
        A::_ConvGF{T}, Acc::_ConvGF{T},
        f::Union{Cf{T},Nothing},
        B::_ConvGF{T}, Bcc::_ConvGF{T},
        I::Integrator,
        beta::F64,
        h::F64,
        alpha::Number
    ) where {T}

Increment `C^<(t',t)` (`t' ≤ t`, `t` is given by `tstp`) by
`alpha * [A^⌉ ∗ f₀ ∗ B^⌈](t',t)`, including the factor `-i`. Here `f[0]`
is used. The argument `h` is not used. It is kept for an unified
interface.

### Arguments
See [`incr_convolution_tv!`](@ref).

### Returns
N/A
"""
function incr_convolution_lmix_rmix!(
    tstp::I64,
    C::_ConvTarget{T},
    A::_ConvGF{T}, Acc::_ConvGF{T},
    f::Union{Cf{T},Nothing},
    B::_ConvGF{T}, Bcc::_ConvGF{T},
    I::Integrator,
    beta::F64,
    h::F64,
    alpha::Number
) where {T}
    # Sanity check
    _check_incr(tstp, C, A, Acc, f, B, Bcc; target = (:less,))
    @assert tstp ≥ 1
    @assert beta > 0.0

    ws = _ConvWorkspace(A, B)
    _incr_lmix_rmix!(_less_target(C, tstp), 1:tstp, tstp,
                     A, Acc, f, B, Bcc, I, beta, h, alpha, ws)
end

#
# Workspace-bearing versions of the three partial lesser convolutions.
# The target `tgt[m]` (m ∈ ms) is C^<(tₘ,tₙ). They are private functions.
#

function _incr_ret_less!(tgt, ms::UnitRange{I64}, n::I64,
                         A::_ConvGF{T}, Acc::_ConvGF{T}, f, B::_ConvGF{T}, Bcc::_ConvGF{T},
                         I::Integrator, beta::F64, h::F64,
                         alpha::Number, ws::_ConvWorkspace{T}) where {T}
    _conv_ret_less_kernel!(tgt, ms, n, A.ret, Acc.ret, f,
                           B.less, Bcc.less, I, alpha * h, ws)
end
#

function _incr_less_adv!(tgt, ms::UnitRange{I64}, n::I64,
                         A::_ConvGF{T}, Acc::_ConvGF{T}, f, B::_ConvGF{T}, Bcc::_ConvGF{T},
                         I::Integrator, beta::F64, h::F64,
                         alpha::Number, ws::_ConvWorkspace{T};
                         diagcc::Bool = false) where {T}
    _conv_less_adv_kernel!(tgt, ms, n, A.less, Acc.less, f,
                           B.ret, Bcc.ret, I, alpha * h, ws; diagcc = diagcc)
end
#
# In incr_convolution_les of NESSi, the diagonal block A^<(t,t) of the
# term A^< ∗ f ∗ Bᴬ is taken as -Acc^<(t,t)† if tstp ≤ 2k + 2 (NESSi
# index). The two choices agree if Acc is the hermitian conjugate partner
# of A. See [NESSi] cntr_convolution_impl.hpp, incr_convolution_les.
#

_incr_les_diagcc(n::I64, I::Integrator) = n - 1 ≤ 2 * I.k + 2
#

function _incr_lmix_rmix!(tgt, ms::UnitRange{I64}, n::I64,
                          A::_ConvGF{T}, Acc::_ConvGF{T}, f, B::_ConvGF{T}, Bcc::_ConvGF{T},
                          I::Integrator, beta::F64, h::F64,
                          alpha::Number, ws::_ConvWorkspace{T}) where {T}
    δτ = beta / (getntau(A) - 1)
    _conv_lmix_rmix_kernel!(tgt, ms, n, A.lmix, _f0(f), Bcc.lmix, I,
                            A.sign, alpha * (-im * δτ), ws)
end


"""
    _incr_convolution_less_adv!(
        buf::AbstractVector{<:AbstractMatrix},
        tstp::I64,
        A::_ConvGF{T}, Acc::_ConvGF{T},
        f::Union{Cf{T},Nothing},
        B::_ConvGF{T}, Bcc::_ConvGF{T},
        I::Integrator,
        beta::F64,
        h::F64,
        alpha::Number,
        ws::_ConvWorkspace{T}
    ) where {T}

Scratch-target version of [`incr_convolution_less_adv!`](@ref) for the
start-up time steps `1 ≤ tstp ≤ k`. Let `n₁ = max(tstp - 1, k)`, then
`buf[j]` (`j = 1, ⋯, n₁ + 1`) is incremented by
`alpha * [A^< ∗ f ∗ Bᴬ](tⱼ,t)`. Note that the blocks `j > tstp` are also
evaluated, which are needed in the start-up procedure. It is a private
function.

### Returns
* buf -> The same vector.
"""
function _incr_convolution_less_adv!(
    buf::AbstractVector{<:AbstractMatrix},
    tstp::I64,
    A::_ConvGF{T}, Acc::_ConvGF{T},
    f::Union{Cf{T},Nothing},
    B::_ConvGF{T}, Bcc::_ConvGF{T},
    I::Integrator,
    beta::F64,
    h::F64,
    alpha::Number,
    ws::_ConvWorkspace{T}
) where {T}
    # Sanity check
    n₁ = max(tstp - 1, I.k)
    @assert 1 ≤ tstp ≤ I.k
    @assert length(buf) == n₁ + 1
    _check_incr(tstp, A, A, Acc, f, B, Bcc; target = ())
    @assert getntime(A) ≥ n₁ + 1
    @assert h > 0.0

    _incr_less_adv!(buf, 1:n₁+1, tstp, A, Acc, f, B, Bcc, I, beta, h, alpha, ws)
    return buf
end


"""
    _incr_convolution_lmix_rmix!(
        buf::AbstractVector{<:AbstractMatrix},
        tstp::I64,
        A::_ConvGF{T}, Acc::_ConvGF{T},
        f::Union{Cf{T},Nothing},
        B::_ConvGF{T}, Bcc::_ConvGF{T},
        I::Integrator,
        beta::F64,
        h::F64,
        alpha::Number,
        ws::_ConvWorkspace{T}
    ) where {T}

Scratch-target version of [`incr_convolution_lmix_rmix!`](@ref) for the
start-up time steps `1 ≤ tstp ≤ k`. Let `n₁ = max(tstp - 1, k)`, then
`buf[j]` (`j = 1, ⋯, n₁ + 1`) is incremented by
`alpha * [A^⌉ ∗ f₀ ∗ B^⌈](tⱼ,t)`. It is a private function.

### Returns
* buf -> The same vector.
"""
function _incr_convolution_lmix_rmix!(
    buf::AbstractVector{<:AbstractMatrix},
    tstp::I64,
    A::_ConvGF{T}, Acc::_ConvGF{T},
    f::Union{Cf{T},Nothing},
    B::_ConvGF{T}, Bcc::_ConvGF{T},
    I::Integrator,
    beta::F64,
    h::F64,
    alpha::Number,
    ws::_ConvWorkspace{T}
) where {T}
    # Sanity check
    n₁ = max(tstp - 1, I.k)
    @assert 1 ≤ tstp ≤ I.k
    @assert length(buf) == n₁ + 1
    _check_incr(tstp, A, A, Acc, f, B, Bcc; target = ())
    @assert getntime(A) ≥ n₁ + 1
    @assert beta > 0.0

    _incr_lmix_rmix!(buf, 1:n₁+1, tstp, A, Acc, f, B, Bcc, I, beta, h, alpha, ws)
    return buf
end


"""
    incr_convolution_les!(
        tstp::I64,
        C::_ConvTarget{T},
        A::_ConvGF{T}, Acc::_ConvGF{T},
        f::Union{Cf{T},Nothing},
        B::_ConvGF{T}, Bcc::_ConvGF{T},
        I::Integrator,
        beta::F64,
        h::F64,
        alpha::Number
    ) where {T}

Increment the lesser component `C^<(t',t)` (`t' ≤ t`, `t` is given by
`tstp`) by `alpha * [A ∗ f ∗ B]^<(t',t)`. It is the sum of the three
partial lesser convolutions.

### Arguments
See [`incr_convolution_tv!`](@ref).

### Returns
N/A

See also: [`incr_convolution_ret_less!`](@ref),
[`incr_convolution_less_adv!`](@ref), [`incr_convolution_lmix_rmix!`](@ref).
"""
function incr_convolution_les!(
    tstp::I64,
    C::_ConvTarget{T},
    A::_ConvGF{T}, Acc::_ConvGF{T},
    f::Union{Cf{T},Nothing},
    B::_ConvGF{T}, Bcc::_ConvGF{T},
    I::Integrator,
    beta::F64,
    h::F64,
    alpha::Number
) where {T}
    # Sanity check
    _check_incr(tstp, C, A, Acc, f, B, Bcc; target = (:less,))
    @assert tstp ≥ 1
    @assert beta > 0.0
    @assert h > 0.0

    ws = _ConvWorkspace(A, B)
    _incr_convolution_les!(tstp, C, A, Acc, f, B, Bcc, I, beta, h, alpha, ws)
end

#
# Workspace-bearing version of incr_convolution_les!(). It is a private
# function.
#

function _incr_convolution_les!(tstp::I64, C, A::_ConvGF{T}, Acc::_ConvGF{T}, f,
                                B::_ConvGF{T}, Bcc::_ConvGF{T}, I::Integrator,
                                beta::F64, h::F64,
                                alpha::Number, ws::_ConvWorkspace{T}) where {T}
    tgt = _less_target(C, tstp)
    _incr_ret_less!(tgt, 1:tstp, tstp, A, Acc, f, B, Bcc, I, beta, h, alpha, ws)
    _incr_less_adv!(tgt, 1:tstp, tstp, A, Acc, f, B, Bcc, I, beta, h, alpha, ws;
                    diagcc = _incr_les_diagcc(tstp, I))
    _incr_lmix_rmix!(tgt, 1:tstp, tstp, A, Acc, f, B, Bcc, I, beta, h, alpha, ws)
end


"""
    incr_convolution!(
        tstp::I64,
        C::Union{ℱ{T},𝒻{T}},
        A::ℱ{T}, Acc::ℱ{T},
        f::Union{Cf{T},Nothing},
        B::ℱ{T}, Bcc::ℱ{T},
        I::Integrator,
        beta::F64,
        h::F64,
        alpha::Number
    ) where {T}

Increment `C` at time step `tstp` by `alpha * [A ∗ f ∗ B]`. If `tstp = 0`,
only the Matsubara component is changed. Otherwise, the retarded,
left-mixing, and lesser components at time step `tstp` are changed.

### Arguments
* tstp  -> Index of time step.
* C     -> Target, a `ℱ` or `𝒻` object.
* A     -> Contour-ordered Green's function, A(t,t'').
* Acc   -> Hermitian conjugate partner of A.
* f     -> Contour function, f(t''), or `nothing`.
* B     -> Contour-ordered Green's function, B(t'',t').
* Bcc   -> Hermitian conjugate partner of B.
* I     -> Struct for numerical integration.
* beta  -> Inverse temperature, β.
* h     -> Time step interval.
* alpha -> Weight.

### Returns
N/A
"""
function incr_convolution!(
    tstp::I64,
    C::Union{ℱ{T},𝒻{T}},
    A::ℱ{T}, Acc::ℱ{T},
    f::Union{Cf{T},Nothing},
    B::ℱ{T}, Bcc::ℱ{T},
    I::Integrator,
    beta::F64,
    h::F64,
    alpha::Number
) where {T}
    # Sanity check
    # Only the Matsubara component is written if tstp = 0, which is
    # accumulated from a temporary.
    _check_incr(tstp, C, A, Acc, f, B, Bcc;
                target = tstp == 0 ? () : (:ret, :lmix, :less))
    @assert beta > 0.0
    @assert h > 0.0

    if tstp == 0
        _check_tstp(C, 0)
        incr_convolution_mat!(C, A, f, B, I, beta, alpha)
    else
        ws = _ConvWorkspace(A, B)
        _incr_convolution_ret!(tstp, C, A, Acc, f, B, Bcc, I, h, alpha, ws)
        _incr_convolution_tv!(tstp, C, A, Acc, f, B, Bcc, I, beta, h, alpha, ws)
        _incr_convolution_les!(tstp, C, A, Acc, f, B, Bcc, I, beta, h, alpha, ws)
    end
end


"""
    incr_convolution!(
        tstp::I64,
        C::Union{ℱ{T},𝒻{T}},
        A::ℱ{T}, f::Union{Cf{T},Nothing}, B::ℱ{T},
        I::Integrator,
        beta::F64,
        h::F64,
        alpha::Number
    ) where {T}

Increment `C` at time step `tstp` by `alpha * [A ∗ f ∗ B]`. Here `A` and
`B` are assumed to be hermitian, i.e., `Acc = A` and `Bcc = B`.
"""
function incr_convolution!(
    tstp::I64,
    C::Union{ℱ{T},𝒻{T}},
    A::ℱ{T}, f::Union{Cf{T},Nothing}, B::ℱ{T},
    I::Integrator,
    beta::F64,
    h::F64,
    alpha::Number
) where {T}
    incr_convolution!(tstp, C, A, A, f, B, B, I, beta, h, alpha)
end


"""
    incr_convolution!(
        tstp::I64,
        C::Union{ℱ{T},𝒻{T}},
        A::ℱ{T}, B::ℱ{T},
        I::Integrator,
        beta::F64,
        h::F64,
        alpha::Number
    ) where {T}

Increment `C` at time step `tstp` by `alpha * [A ∗ B]`. Here `A` and `B`
are assumed to be hermitian, i.e., `Acc = A` and `Bcc = B`.
"""
function incr_convolution!(
    tstp::I64,
    C::Union{ℱ{T},𝒻{T}},
    A::ℱ{T}, B::ℱ{T},
    I::Integrator,
    beta::F64,
    h::F64,
    alpha::Number
) where {T}
    incr_convolution!(tstp, C, A, A, nothing, B, B, I, beta, h, alpha)
end


