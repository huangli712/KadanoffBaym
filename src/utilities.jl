#
# Project : Lavender
# Source  : utilities.jl
# Author  : Li Huang (huangli@caep.cn)
# Status  : Unstable
#
# Last modified: 2026/11/09
#

#=
*Remarks* : *Polynomial Extrapolation*

Suppose that the contour function is known at the time steps ``t_{n-l}``
(``l = 0, \cdots, k``). The ``k``-th order polynomial approximant through
these points is evaluated at ``t_{n+1}``, which yields

```math
\begin{equation}
y_{n+1} \approx \sum^{k}_{l=0} p^{(k)}_l y_{n-l},
\end{equation}
```

with

```math
\begin{equation}
p^{(k)}_l = \sum^{k}_{a=0} (-1)^a P^{(k)}_{al}.
\end{equation}
```

Here ``P^{(k)}_{al}`` denotes the polynomial interpolation weights.

For the retarded component, the extrapolation is done along the lines
of fixed relative time ``t - t'``. For the lesser component, it is done
along the stored lines of fixed right time. The hermitian conjugate
relations are used if the needed elements are not in the stored triangles.

*References* :

Please see [`NESSi`] `cntr_utilities_impl.hpp`, `extrapolate_timestep`.
=#

#=
### *Private Service Functions* : *Extrapolation*
=#

"""
    _extrapolation_weights(I::Integrator)

Return the extrapolation weights ``p^{(k)}_l`` (``l = 0, \\cdots, k``).
It is a private function.

### Arguments
* I -> An Integrator struct.

### Returns
* p -> Vector of weights, `p[l+1]` is ``p^{(k)}_l``.
"""
function _extrapolation_weights(I::Integrator)
    k = I.k
    p = zeros(F64, k + 1)
    for m = 0:k
        for l = 0:k
            p[m + 1] = p[m + 1] + (1 - 2 * (l % 2)) * I.PIW[l, m]
        end
    end
    return p
end

"""
    _extrapolate_copy!(n::I64, G::ℱ{T})

Zeroth order extrapolation of the `ret`, `lmix`, and `less` components
of `G` to the time step `n + 1`. The values at time step `n` are simply
copied. It is a private function.

See also: [`extrapolate_timestep!`](@ref).
"""
function _extrapolate_copy!(n::I64, G::ℱ{T}) where {T}
    # Sanity check
    @assert 1 ≤ n < getntime(G)

    n₁ = n + 1
    ntau = getntau(G)

    # lmix component
    for m = 1:ntau
        copyto!(_block(G.lmix, n₁, m), _block(G.lmix, n, m))
    end

    # ret component
    for j = 0:n-1
        copyto!(_block(G.ret, n₁, n₁ - j), _block(G.ret, n, n - j))
    end
    copyto!(_block(G.ret, n₁, 1), _block(G.ret, n, 1))

    # less component
    for j = 2:n₁
        copyto!(_block(G.less, j, n₁), _block(G.less, j - 1, n))
    end
    copyto!(_block(G.less, 1, n₁), _block(G.less, 1, n))
end

#=
### *Public API* : *Extrapolation*
=#

"""
    extrapolate_timestep!(n::I64, G::ℱ{T}, I::Integrator)

Polynomial extrapolation of the `ret`, `lmix`, and `less` components of
the contour-ordered Green's function `G` to the time step `n + 1`. The
information at the time steps `n, n-1, ..., n-k` is used. Here `k` is
the order of the integrator `I`. The Matsubara component is not touched.

### Arguments
* n -> Index of the last known time step, `k + 1 ≤ n < ntime`.
* G -> A contour-ordered Green's function.
* I -> An Integrator struct.

### Returns
N/A

See also: [`Integrator`](@ref).
"""
function extrapolate_timestep!(n::I64, G::ℱ{T}, I::Integrator) where {T}
    # Get key parameters
    k = I.k
    n₁ = n + 1
    ntau = getntau(G)

    # Sanity check
    @assert n ≥ k + 1
    @assert n₁ ≤ getntime(G)

    # Get extrapolation weights
    p = _extrapolation_weights(I)

    # Temporary block
    ndim1, ndim2 = getdims(G)
    gtmp = zeros(T, ndim1, ndim2)

    # lmix component
    for m = 1:ntau
        fill!(gtmp, zero(T))
        for l = 0:k
            gtmp .= gtmp .+ p[l + 1] .* _block(G.lmix, n - l, m)
        end
        copyto!(_block(G.lmix, n₁, m), gtmp)
    end

    # ret component, along the lines of fixed relative time
    for j = 0:n-1-k
        fill!(gtmp, zero(T))
        for l = 0:k
            gtmp .= gtmp .+ p[l + 1] .* _block(G.ret, n - l, n - l - j)
        end
        copyto!(_block(G.ret, n₁, n₁ - j), gtmp)
    end
    #
    # ret component, the first k + 1 columns
    for j = 1:k+1
        fill!(gtmp, zero(T))
        for l = 0:k
            if n - l ≤ j
                gtmp .= gtmp .- p[l + 1] .* adjoint(_block(G.ret, j, n - l))
            else
                gtmp .= gtmp .+ p[l + 1] .* _block(G.ret, n - l, j)
            end
        end
        copyto!(_block(G.ret, n₁, j), gtmp)
    end

    # less component, the first k + 1 rows
    for j = 1:k+1
        fill!(gtmp, zero(T))
        for l = 0:k
            if n - l < j
                gtmp .= gtmp .- p[l + 1] .* adjoint(_block(G.less, n - l, j))
            else
                gtmp .= gtmp .+ p[l + 1] .* _block(G.less, j, n - l)
            end
        end
        copyto!(_block(G.less, j, n₁), gtmp)
    end
    #
    # less component, along the lines of fixed relative time
    for j = k+2:n₁
        fill!(gtmp, zero(T))
        for l = 0:k
            gtmp .= gtmp .+ p[l + 1] .* _block(G.less, j - l - 1, n - l)
        end
        copyto!(_block(G.less, j, n₁), gtmp)
    end
end

"""
    extrapolate_timestep!(n::I64, G::ℱ{T}, order::I64)

Polynomial extrapolation of the `ret`, `lmix`, and `less` components of
the contour-ordered Green's function `G` to the time step `n + 1`. If
`order = 0`, the values at time step `n` are copied.

### Arguments
* n     -> Index of the last known time step.
* G     -> A contour-ordered Green's function.
* order -> Order of extrapolation, 0 ≤ order ≤ 5.

### Returns
N/A
"""
function extrapolate_timestep!(n::I64, G::ℱ{T}, order::I64) where {T}
    @assert 0 ≤ order ≤ 5
    if order == 0
        _extrapolate_copy!(n, G)
    else
        extrapolate_timestep!(n, G, Integrator(order))
    end
end

"""
    extrapolate_timestep!(n::I64, f::Cf{T}, I::Integrator)

Polynomial extrapolation of the contour function `f` to the time step
`n + 1`. The information at the time steps `n, n-1, ..., n-k` is used.
The value at the Matsubara branch `f[0]` is not touched.

### Arguments
* n -> Index of the last known time step, `k + 1 ≤ n < ntime`.
* f -> A contour function.
* I -> An Integrator struct.

### Returns
N/A
"""
function extrapolate_timestep!(n::I64, f::Cf{T}, I::Integrator) where {T}
    # Get key parameters
    k = I.k

    # Sanity check
    @assert n ≥ k + 1
    @assert n + 1 ≤ f.ntime

    # Get extrapolation weights
    p = _extrapolation_weights(I)

    # Evaluate y(n+1) = ∑ₗ pₗ y(n-l)
    ftmp = zeros(T, getdims(f))
    for l = 0:k
        ftmp .= ftmp .+ p[l + 1] .* f[n - l]
    end
    f[n + 1] = ftmp
end

"""
    extrapolate_timestep!(n::I64, f::Cf{T}, order::I64)

Polynomial extrapolation of the contour function `f` to the time step
`n + 1`. If `order = 0`, the value at time step `n` is copied.

### Arguments
* n     -> Index of the last known time step.
* f     -> A contour function.
* order -> Order of extrapolation, 0 ≤ order ≤ 5.

### Returns
N/A
"""
function extrapolate_timestep!(n::I64, f::Cf{T}, order::I64) where {T}
    @assert 0 ≤ order ≤ 5
    if order == 0
        @assert 1 ≤ n < f.ntime
        f[n + 1] = f[n]
    else
        extrapolate_timestep!(n, f, Integrator(order))
    end
end

#=
### *Public API* : *Interpolation*
=#

#=
*Remarks* : *Polynomial Interpolation*

The ``k``-th order polynomial approximant through the function values
``(x_l, f_l)``, ``l = 0, \cdots, k``, is given by

```math
\begin{equation}
\mathcal{P}(x) = \sum^{k}_{p,l=0} x^p P^{(k)}_{pl} f_l.
\end{equation}
```

Here ``x`` is measured in units of the time step, from the first grid
point ``t_{tstp-k}`` of the interpolation interval.

*References* :

Please see [`NESSi`] `cntr_utilities_impl.hpp`, `interpolation`.
=#

"""
    interpolation(tstp::I64, tinter::Real, f::Cf{T}, I::Integrator)

Polynomial interpolation of the contour function `f` at an arbitrary
point `tinter` of the time interval `[tstp - k, tstp]`. Here `tinter` is
a continuous grid coordinate, i.e., the physical time is
`(tinter - 1) * dt`.

### Arguments
* tstp   -> Index of the last grid point of the interval, `tstp ≥ k + 1`.
* tinter -> Interpolation point.
* f      -> A contour function.
* I      -> An Integrator struct.

### Returns
* x -> Interpolated value, a new `Element{T}`.
"""
function interpolation(tstp::I64, tinter::Real, f::Cf{T}, I::Integrator) where {T}
    # Get key parameters
    k = I.k

    # Sanity check
    @assert tstp ≥ k + 1
    @assert tstp ≤ f.ntime
    @assert tstp - k ≤ tinter ≤ tstp

    # Evaluate the polynomial
    x = zeros(T, getdims(f))
    t₀ = F64(tstp - k)
    for l = 0:k
        t₁ = 1.0
        weight = I.PIW[0, l]
        for p = 1:k
            t₁ = t₁ * (tinter - t₀)
            weight = weight + t₁ * I.PIW[p, l]
        end
        x .= x .+ f[tstp - k + l] .* weight
    end

    return x
end

"""
    interpolation(tstp::I64, tinter::Real, f::Cf{T}, order::I64)

Polynomial interpolation of the contour function `f` at an arbitrary
point `tinter` of the time interval `[tstp - order, tstp]`. If
`order = 0`, `tinter` must be equal to `tstp`, and `f[tstp]` is copied.

### Arguments
* tstp   -> Index of the last grid point of the interval.
* tinter -> Interpolation point.
* f      -> A contour function.
* order  -> Order of interpolation, 0 ≤ order ≤ 5.

### Returns
* x -> Interpolated value, a new `Element{T}`.
"""
function interpolation(tstp::I64, tinter::Real, f::Cf{T}, order::I64) where {T}
    @assert 0 ≤ order ≤ 5
    if order == 0
        @assert 1 ≤ tstp ≤ f.ntime
        @assert tinter == tstp
        return copy(f[tstp])
    else
        return interpolation(tstp, tinter, f, Integrator(order))
    end
end

#=
### *Public API* : *Initial State*
=#

#=
*Remarks* : *Boundary Values From Matsubara Component*

At ``t = 0``, the real-time components are determined by the Matsubara
component:

```math
\begin{equation}
G^{\rceil}(0,\tau) = i \xi G^{M}(\beta - \tau),
\end{equation}
```

```math
\begin{equation}
G^{<}(0,0) = G^{\rceil}(0,0),
\end{equation}
```

```math
\begin{equation}
G^{R}(0,0) = i G^{M}(0) - G^{<}(0,0).
\end{equation}
```

Here ``\xi = -1`` (``+1``) for fermions (bosons).

*References* :

Please see [`NESSi`] `cntr_utilities_impl.hpp`, `set_t0_from_mat`.
=#

"""
    _set_tn_from_mat!(n::I64, G::ℱ{T})

Setup the `lmix` component at time step `n`, and the diagonal elements
of the `ret` and `less` components at `(n, n)`, by using the Matsubara
component. It is a private function.

See also: [`set_t0_from_mat!`](@ref), [`set_tk_from_mat!`](@ref).
"""
function _set_tn_from_mat!(n::I64, G::ℱ{T}) where {T}
    ntau = getntau(G)
    sign = getsign(G)

    # G^{⌉}(tₙ,τₘ) = i ξ G^{M}(β - τₘ)
    for m = 1:ntau
        _block(G.lmix, n, m) .= (im * sign) .* _block(G.mat, ntau - m + 1)
    end

    # G^{<}(tₙ,tₙ) = G^{⌉}(0,0)
    copyto!(_block(G.less, n, n), _block(G.lmix, 1, 1))

    # G^{R}(tₙ,tₙ) = i G^{M}(0) - G^{<}(0,0)
    _block(G.ret, n, n) .= im .* _block(G.mat, 1) .- _block(G.less, 1, 1)
end

"""
    set_t0_from_mat!(G::ℱ{T})

Setup the first real time step (`t = 0`, i.e., `tstp = 1`) of the `lmix`,
`ret`, and `less` components of `G` by using its Matsubara component.

### Arguments
* G -> A contour-ordered Green's function.

### Returns
N/A
"""
function set_t0_from_mat!(G::ℱ{T}) where {T}
    _set_tn_from_mat!(1, G)
end

"""
    set_tk_from_mat!(G::ℱ{T}, k::I64)

Setup the time steps `1, 2, ..., k + 1` of the `lmix` component, and the
diagonal elements of the `ret` and `less` components, by using the
Matsubara component of `G`. The values are the same as those at `t = 0`.
The off-diagonal elements of `ret` and `less` are not touched.

### Arguments
* G -> A contour-ordered Green's function.
* k -> The last time step to be set is `k + 1`.

### Returns
N/A
"""
function set_tk_from_mat!(G::ℱ{T}, k::I64) where {T}
    # Sanity check
    @assert k ≥ 0
    @assert getntime(G) ≥ k + 1

    for n = 1:k+1
        _set_tn_from_mat!(n, G)
    end
end

#=
### *Public API* : *Hermitian Matsubara Component*
=#

"""
    force_matsubara_hermitian!(G::ℱ{T})

Force the Matsubara component of `G` to be hermitian, i.e.,
``G^{M}(\\tau) \\to [G^{M}(\\tau) + G^{M}(\\tau)^{\\dagger}] / 2``.

### Arguments
* G -> A contour-ordered Green's function.

### Returns
N/A
"""
function force_matsubara_hermitian!(G::ℱ{T}) where {T}
    # Sanity check
    @assert equaldims(G)

    for m = 1:getntau(G)
        M = _block(G.mat, m)
        M .= (M .+ copy(M')) ./ 2
    end
end

"""
    force_matsubara_hermitian!(G::𝒻{S})

Force the Matsubara component of `G` to be hermitian, i.e.,
``G^{M}(\\tau) \\to [G^{M}(\\tau) + G^{M}(\\tau)^{\\dagger}] / 2``.

### Arguments
* G -> A contour-ordered Green's function at given time step.

### Returns
N/A
"""
function force_matsubara_hermitian!(G::𝒻{S}) where {S}
    # Sanity check
    @assert equaldims(G)

    for m = 1:getntau(G)
        M = G.mat.data[m]
        M .= (M .+ copy(M')) ./ 2
    end
end
