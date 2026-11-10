#
# Project : Lavender
# Source  : interpolation.jl
# Author  : Li Huang (huangli@caep.cn)
# Status  : Unstable
#
# Last modified: 2026/11/10
#

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
