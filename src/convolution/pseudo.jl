#
# Project : Lavender
# Source  : pseudo.jl
# Author  : Li Huang (huangli@caep.cn)
# Status  : Unstable
#
# Last modified: 2026/11/10
#

#=
*Remarks* : *Pseudo-Particle Convolution*

For the pseudo-particle Green's functions, only the Matsubara and the
left-mixing components of the convolution ``C = A \ast f \ast B`` differ
from the standard ones. The Matsubara component is a causal (restricted)
integral,

```math
\begin{equation}
C^{M}(\tau) = \int^{\tau}_{0} d\bar{\tau}
              A^{M}(\tau - \bar{\tau}) f_0 B^{M}(\bar{\tau}),
\end{equation}
```

without the periodic (or anti-periodic) contribution from ``\tau`` to
``\beta``. In the left-mixing component, the imaginary time integral is
restricted to ``\tau \cdots \beta``:

```math
\begin{equation}
C^{\rceil}(t,\tau) = \int^{t}_{0} d\bar{t} A^{>}(t,\bar{t}) f(\bar{t})
                     B^{\rceil}(\bar{t},\tau)
                   + \int^{\beta}_{\tau} d\bar{\tau} A^{\rceil}(t,\bar{\tau})
                     f_0 B^{M}(\bar{\tau} - \tau).
\end{equation}
```

Here the stored `ret` component of `A` (i.e., ``A^{>}``) is used, and the
weights are ``h`` and ``\delta\tau``, respectively. No extra factor ``-i``
or statistical sign is involved. The retarded and lesser components are
evaluated by the standard kernels, see `convolution.jl`.

*References* :

Please see [`NESSi`] `cntr_pseudo_convolution_impl.hpp`, and
`cntr_convolution_impl.hpp`, `matsubara_integral_1_1` and
`matsubara_integral_2_2`.
=#

#=
### *Pseudo-Particle Convolution* : *Private Kernels*
=#

"""
    _pseudo_mat_integral!(c, m::I64, A::Gᵐᵃᵗ, Bt, I::Integrator)

Evaluate the restricted Matsubara integral
`c = ∫₀^τ dx Aᴹ(τ - x) Bt(x)` at the imaginary time point `m` (KB index,
`τ = (m - 1) δτ`) without the factor `δτ`. Here `Bt[m]` returns the block
`f₀ Bᴹ(τₘ)`. It is a private function.

*References* :

Please see [`NESSi`] `cntr_convolution_impl.hpp`, `matsubara_integral_1_1`.
"""
function _pseudo_mat_integral!(c, m::I64, A::Gᵐᵃᵗ, Bt, I::Integrator)
    k = I.k
    n = m - 1 # NESSi index of τ
    fill!(c, zero(eltype(c)))

    if n ≥ 2 * k + 1 # Usual Gregory integration with end corrections
        for j = 0:k
            mul!(c, _block(A, n - j + 1), Bt[j+1], I.GIW.ω[j+1], true)
        end
        for j = k+1:n-k-1
            mul!(c, _block(A, n - j + 1), Bt[j+1], true, true)
        end
        for j = n-k:n
            mul!(c, _block(A, n - j + 1), Bt[j+1], I.GIW.ω[n-j+1], true)
        end
    elseif n ≥ k # Gregory integration
        for j = 0:n
            mul!(c, _block(A, n - j + 1), Bt[j+1], I.GIW[n,j], true)
        end
    elseif n > 0 # Strange boundary correction
        for l = 0:k
            for j = 0:k
                mul!(c, _block(A, l + 1), Bt[j+1], I.BCW[n-1,l,j], true)
            end
        end
    end

    return c
end

"""
    _pseudo_lmix_mat_loop!(tgt, n::I64, A::Gˡᵐⁱˣ, Bt, I::Integrator, s::Number)

Accumulate `s * ∫_τ^β dx A^⌉(tₙ,x) Bt(x - τ)` into `tgt[m]` for all `m`.
Here `Bt[m]` returns the block `f₀ Bᴹ(τₘ)`. It is the second half of
[`_conv_lmix_mat_loop!`](@ref). It is a private function.

*References* :

Please see [`NESSi`] `cntr_convolution_impl.hpp`, `matsubara_integral_2_2`.
"""
function _pseudo_lmix_mat_loop!(tgt, n::I64, A::Gˡᵐⁱˣ, Bt, I::Integrator, s::Number)
    k = I.k
    ntau = A.ntau

    for m = 1:ntau
        c = tgt[m]
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

"""
    _check_pseudo_mask(mask, n::I64)

Check the optional mask. If `mask` is `nothing`, all the points are
selected. It is a private function.
"""
_check_pseudo_mask(::Nothing, n::I64) = nothing

_check_pseudo_mask(mask::AbstractVector{Bool}, n::I64) = @assert length(mask) == n
#

_selected(::Nothing, m::I64) = true

_selected(mask::AbstractVector{Bool}, m::I64) = mask[m]

#=
### *Pseudo-Particle Convolution* : *Incremental Convolution*
=#

"""
    incr_pseudo_convolution_mat!(
        C::Union{PseudoGF{T},PseudoGFStep{T}},
        A::PseudoGF{T}, f::Union{Cf{T},Nothing}, B::PseudoGF{T},
        I::Integrator,
        beta::F64,
        alpha::Number;
        mask::Union{Nothing,AbstractVector{Bool}} = nothing
    ) where {T}

Increment the Matsubara component of the pseudo-particle Green's function
`C` by `alpha * δτ ∫₀^τ dx Aᴹ(τ - x) f₀ Bᴹ(x)`. Here `f[0]` is used. If
`f = nothing`, then `f = 1`. Only the imaginary time points selected by
`mask` (of length `ntau`) are changed.

### Arguments
* C     -> Target, a `PseudoGF` or `PseudoGFStep` object.
* A     -> Pseudo-particle Green's function.
* f     -> Contour function, or `nothing`.
* B     -> Pseudo-particle Green's function.
* I     -> Struct for numerical integration.
* beta  -> Inverse temperature, β.
* alpha -> Weight.
* mask  -> Selected imaginary time points, or `nothing` (all points).

### Returns
N/A

*References* :

Please see [`NESSi`] `cntr_pseudo_convolution_impl.hpp`,
`incr_pseudo_convolution_mat`.
"""
function incr_pseudo_convolution_mat!(
    C::Union{PseudoGF{T},PseudoGFStep{T}},
    A::PseudoGF{T}, f::Union{Cf{T},Nothing}, B::PseudoGF{T},
    I::Integrator,
    beta::F64,
    alpha::Number;
    mask::Union{Nothing,AbstractVector{Bool}} = nothing
) where {T}
    # Sanity check
    _check_incr(0, C, A, A, f, B, B; target = (:mat,))
    @assert beta > 0.0
    ntau = getntau(A)
    _check_pseudo_mask(mask, ntau)

    δτ = beta / (ntau - 1)
    ws = _ConvWorkspace(A, B)
    Bt = _fbmat!(ws, _f0(f), B.mat)
    tgt = _mat_target(C, 0)
    c = similar(ws.af)
    #
    for m = 1:ntau
        _selected(mask, m) || continue
        _pseudo_mat_integral!(c, m, A.mat, Bt, I)
        tgt[m] .= tgt[m] .+ (alpha * δτ) .* c
    end
end

"""
    incr_pseudo_convolution_tv!(
        tstp::I64,
        C::Union{PseudoGF{T},PseudoGFStep{T}},
        A::PseudoGF{T}, Acc::PseudoGF{T},
        f::Union{Cf{T},Nothing},
        B::PseudoGF{T}, Bcc::PseudoGF{T},
        I::Integrator,
        beta::F64,
        h::F64,
        alpha::Number;
        mask::Union{Nothing,AbstractVector{Bool}} = nothing
    ) where {T}

Increment the left-mixing component of the pseudo-particle Green's
function `C` at time step `tstp` by `alpha * [A ∗ f ∗ B]^⌉(t,τ)`, where
the imaginary time integral is restricted to `τ ⋯ β`. Only the imaginary
time points selected by `mask` (of length `ntau`) are changed.

### Arguments
See [`incr_convolution_lmix!`](@ref).

### Returns
N/A

*References* :

Please see [`NESSi`] `cntr_pseudo_convolution_impl.hpp`,
`incr_pseudo_convolution_tv`.
"""
function incr_pseudo_convolution_tv!(
    tstp::I64,
    C::Union{PseudoGF{T},PseudoGFStep{T}},
    A::PseudoGF{T}, Acc::PseudoGF{T},
    f::Union{Cf{T},Nothing},
    B::PseudoGF{T}, Bcc::PseudoGF{T},
    I::Integrator,
    beta::F64,
    h::F64,
    alpha::Number;
    mask::Union{Nothing,AbstractVector{Bool}} = nothing
) where {T}
    # Sanity check
    _check_incr(tstp, C, A, Acc, f, B, Bcc; target = ())
    @assert tstp ≥ 1
    @assert beta > 0.0
    @assert h > 0.0
    _check_pseudo_mask(mask, getntau(A))

    ws = _ConvWorkspace(A, B)
    _incr_pseudo_convolution_tv!(tstp, C, A, Acc, f, B, Bcc, I, beta, h, alpha, ws, mask)
end

#
# Workspace-bearing version of incr_pseudo_convolution_tv!(). The result
# is accumulated into a scratch row first, so that the target may be read
# by the kernels. It is a private function.
#

function _incr_pseudo_convolution_tv!(tstp::I64, C, A::PseudoGF{T}, Acc::PseudoGF{T}, f,
                                      B::PseudoGF{T}, Bcc::PseudoGF{T}, I::Integrator,
                                      beta::F64, h::F64, alpha::Number,
                                      ws::_ConvWorkspace{T}, mask) where {T}
    ntau = getntau(A)
    δτ = beta / (ntau - 1)
    row = [zeros(T, size(ws.af)) for _ = 1:ntau]
    #
    # Imaginary time part, A^⌉ ⋆ f₀ Bᴹ restricted to τ ⋯ β
    _pseudo_lmix_mat_loop!(row, tstp, A.lmix, _fbmat!(ws, _f0(f), B.mat), I, alpha * δτ)
    #
    # Real time part, A^{>} ∗ f ∗ B^⌉, the same as the standard one
    _conv_ret_lmix_kernel!(row, tstp, A.ret, Acc.ret, f, B.lmix, I, alpha * h, ws)
    #
    tgt = _lmix_target(C, tstp)
    for m = 1:ntau
        _selected(mask, m) || continue
        tgt[m] .= tgt[m] .+ row[m]
    end
end

"""
    incr_pseudo_convolution!(
        tstp::I64,
        C::Union{PseudoGF{T},PseudoGFStep{T}},
        A::PseudoGF{T}, Acc::PseudoGF{T},
        f::Union{Cf{T},Nothing},
        B::PseudoGF{T}, Bcc::PseudoGF{T},
        I::Integrator,
        beta::F64,
        h::F64,
        alpha::Number
    ) where {T}

Increment the pseudo-particle Green's function `C` at time step `tstp` by
`alpha * [A ∗ f ∗ B]`. If `tstp = 0`, the restricted Matsubara component
is changed. Otherwise, the greater (`ret`), left-mixing, and lesser
components are changed. The greater and lesser components are evaluated
by the standard kernels.

### Arguments
See [`incr_convolution!`](@ref).

### Returns
N/A

*References* :

Please see [`NESSi`] `cntr_pseudo_convolution_impl.hpp`,
`incr_pseudo_convolution`.
"""
function incr_pseudo_convolution!(
    tstp::I64,
    C::Union{PseudoGF{T},PseudoGFStep{T}},
    A::PseudoGF{T}, Acc::PseudoGF{T},
    f::Union{Cf{T},Nothing},
    B::PseudoGF{T}, Bcc::PseudoGF{T},
    I::Integrator,
    beta::F64,
    h::F64,
    alpha::Number
) where {T}
    if tstp == 0
        _check_tstp(C, 0)
        incr_pseudo_convolution_mat!(C, A, f, B, I, beta, alpha)
    else
        # Sanity check
        _check_incr(tstp, C, A, Acc, f, B, Bcc; target = (:ret, :less))
        @assert beta > 0.0
        @assert h > 0.0
        #
        ws = _ConvWorkspace(A, B)
        _incr_convolution_ret!(tstp, C, A, Acc, f, B, Bcc, I, h, alpha, ws)
        _incr_pseudo_convolution_tv!(tstp, C, A, Acc, f, B, Bcc, I, beta, h, alpha, ws, nothing)
        _incr_convolution_les!(tstp, C, A, Acc, f, B, Bcc, I, beta, h, alpha, ws)
    end
end

#=
### *Pseudo-Particle Convolution* : *Public API*
=#

"""
    pseudo_convolution_timestep!(
        tstp::I64,
        C::PseudoGF{T},
        A::PseudoGF{T}, Acc::PseudoGF{T},
        f::Union{Cf{T},Nothing},
        B::PseudoGF{T}, Bcc::PseudoGF{T},
        I::Integrator,
        beta::F64,
        h::F64
    ) where {T}

Calculate the pseudo-particle convolution `C = A ∗ f ∗ B` at time step
`tstp`. The old values of `C` at this time step are discarded. If
`tstp = 0`, only the Matsubara component is evaluated.

### Arguments
* tstp -> Index of time step.
* C    -> Pseudo-particle Green's function, the result.
* A    -> Pseudo-particle Green's function.
* Acc  -> Hermitian conjugate partner of A.
* f    -> Contour function, or `nothing`.
* B    -> Pseudo-particle Green's function.
* Bcc  -> Hermitian conjugate partner of B.
* I    -> Struct for numerical integration.
* beta -> Inverse temperature, β.
* h    -> Time step interval.

### Returns
N/A

*References* :

Please see [`NESSi`] `cntr_pseudo_convolution_impl.hpp`,
`pseudo_convolution_timestep`.
"""
function pseudo_convolution_timestep!(
    tstp::I64,
    C::PseudoGF{T},
    A::PseudoGF{T}, Acc::PseudoGF{T},
    f::Union{Cf{T},Nothing},
    B::PseudoGF{T}, Bcc::PseudoGF{T},
    I::Integrator,
    beta::F64,
    h::F64
) where {T}
    # Sanity check
    @assert 0 ≤ tstp ≤ getntime(C)
    @assert 1 ≤ I.k ≤ 5
    if tstp ≥ 1
        # The start-up time steps need the data at the first k + 1 steps
        nmin = max(tstp, I.k + 1)
        @assert getntime(A) ≥ nmin && getntime(Acc) ≥ nmin
        @assert getntime(B) ≥ nmin && getntime(Bcc) ≥ nmin
        f isa Cf && @assert f.ntime ≥ nmin
    end

    zeros!(C, tstp)
    incr_pseudo_convolution!(tstp, C, A, Acc, f, B, Bcc, I, beta, h, one(T))
end

"""
    pseudo_convolution_timestep!(
        tstp::I64,
        C::PseudoGF{T},
        A::PseudoGF{T}, Acc::PseudoGF{T},
        B::PseudoGF{T}, Bcc::PseudoGF{T},
        I::Integrator,
        beta::F64,
        h::F64
    ) where {T}

Calculate the pseudo-particle convolution `C = A ∗ B` at time step `tstp`.
"""
function pseudo_convolution_timestep!(
    tstp::I64,
    C::PseudoGF{T},
    A::PseudoGF{T}, Acc::PseudoGF{T},
    B::PseudoGF{T}, Bcc::PseudoGF{T},
    I::Integrator,
    beta::F64,
    h::F64
) where {T}
    pseudo_convolution_timestep!(tstp, C, A, Acc, nothing, B, Bcc, I, beta, h)
end

"""
    pseudo_convolution_timestep!(
        tstp::I64,
        C::PseudoGF{T},
        A::PseudoGF{T}, f::Union{Cf{T},Nothing}, B::PseudoGF{T},
        I::Integrator,
        beta::F64,
        h::F64
    ) where {T}

Hermitian version (`Acc = A`, `Bcc = B`) of the pseudo-particle
convolution at time step `tstp`.
"""
function pseudo_convolution_timestep!(
    tstp::I64,
    C::PseudoGF{T},
    A::PseudoGF{T}, f::Union{Cf{T},Nothing}, B::PseudoGF{T},
    I::Integrator,
    beta::F64,
    h::F64
) where {T}
    pseudo_convolution_timestep!(tstp, C, A, A, f, B, B, I, beta, h)
end

"""
    pseudo_convolution_timestep!(
        tstp::I64,
        C::PseudoGF{T},
        A::PseudoGF{T}, B::PseudoGF{T},
        I::Integrator,
        beta::F64,
        h::F64
    ) where {T}

Hermitian version (`Acc = A`, `Bcc = B`, `f = 1`) of the pseudo-particle
convolution at time step `tstp`.
"""
function pseudo_convolution_timestep!(
    tstp::I64,
    C::PseudoGF{T},
    A::PseudoGF{T}, B::PseudoGF{T},
    I::Integrator,
    beta::F64,
    h::F64
) where {T}
    pseudo_convolution_timestep!(tstp, C, A, A, nothing, B, B, I, beta, h)
end

"""
    pseudo_convolution_matsubara!(
        C::PseudoGF{T},
        A::PseudoGF{T}, f::Union{Cf{T},Nothing}, B::PseudoGF{T},
        I::Integrator,
        beta::F64
    ) where {T}

Calculate the Matsubara component of the pseudo-particle convolution
`C = A ∗ f ∗ B`.

*References* :

Please see [`NESSi`] `cntr_pseudo_convolution_impl.hpp`,
`pseudo_convolution_matsubara`.
"""
function pseudo_convolution_matsubara!(
    C::PseudoGF{T},
    A::PseudoGF{T}, f::Union{Cf{T},Nothing}, B::PseudoGF{T},
    I::Integrator,
    beta::F64
) where {T}
    zeros!(C, 0)
    incr_pseudo_convolution_mat!(C, A, f, B, I, beta, one(T))
end

"""
    pseudo_convolution_matsubara!(
        C::PseudoGF{T},
        A::PseudoGF{T}, B::PseudoGF{T},
        I::Integrator,
        beta::F64
    ) where {T}

Calculate the Matsubara component of the pseudo-particle convolution
`C = A ∗ B`.
"""
function pseudo_convolution_matsubara!(
    C::PseudoGF{T},
    A::PseudoGF{T}, B::PseudoGF{T},
    I::Integrator,
    beta::F64
) where {T}
    pseudo_convolution_matsubara!(C, A, nothing, B, I, beta)
end

"""
    pseudo_convolution!(
        C::PseudoGF{T},
        A::PseudoGF{T}, Acc::PseudoGF{T},
        f::Union{Cf{T},Nothing},
        B::PseudoGF{T}, Bcc::PseudoGF{T},
        I::Integrator,
        beta::F64,
        h::F64
    ) where {T}

Calculate the pseudo-particle convolution `C = A ∗ f ∗ B` at all the time
steps (`tstp = 0, 1, ⋯, ntime`).

*References* :

Please see [`NESSi`] `cntr_pseudo_convolution_impl.hpp`,
`pseudo_convolution`.
"""
function pseudo_convolution!(
    C::PseudoGF{T},
    A::PseudoGF{T}, Acc::PseudoGF{T},
    f::Union{Cf{T},Nothing},
    B::PseudoGF{T}, Bcc::PseudoGF{T},
    I::Integrator,
    beta::F64,
    h::F64
) where {T}
    for tstp = 0:getntime(C)
        pseudo_convolution_timestep!(tstp, C, A, Acc, f, B, Bcc, I, beta, h)
    end
end

"""
    pseudo_convolution!(
        C::PseudoGF{T},
        A::PseudoGF{T}, Acc::PseudoGF{T},
        B::PseudoGF{T}, Bcc::PseudoGF{T},
        I::Integrator,
        beta::F64,
        h::F64
    ) where {T}

Calculate the pseudo-particle convolution `C = A ∗ B` at all the time
steps.
"""
function pseudo_convolution!(
    C::PseudoGF{T},
    A::PseudoGF{T}, Acc::PseudoGF{T},
    B::PseudoGF{T}, Bcc::PseudoGF{T},
    I::Integrator,
    beta::F64,
    h::F64
) where {T}
    pseudo_convolution!(C, A, Acc, nothing, B, Bcc, I, beta, h)
end

"""
    pseudo_convolution!(
        C::PseudoGF{T},
        A::PseudoGF{T}, f::Union{Cf{T},Nothing}, B::PseudoGF{T},
        I::Integrator,
        beta::F64,
        h::F64
    ) where {T}

Hermitian version (`Acc = A`, `Bcc = B`) of the pseudo-particle
convolution at all the time steps.
"""
function pseudo_convolution!(
    C::PseudoGF{T},
    A::PseudoGF{T}, f::Union{Cf{T},Nothing}, B::PseudoGF{T},
    I::Integrator,
    beta::F64,
    h::F64
) where {T}
    pseudo_convolution!(C, A, A, f, B, B, I, beta, h)
end

