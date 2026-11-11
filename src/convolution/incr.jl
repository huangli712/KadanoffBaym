#
# Project : Lavender
# Source  : incr.jl
# Author  : Li Huang (huangli@caep.cn)
# Status  : Unstable
#
# Last modified: 2026/11/10
#

#=
### *Public API* : *Incremental Convolution*
=#

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
        _incr_convolution_lmix!(tstp, C, A, Acc, f, B, Bcc, I, beta, h, alpha, ws)
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
