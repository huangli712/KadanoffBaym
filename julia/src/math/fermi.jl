#
# Project : Lavender
# Source  : fermi.jl
# Author  : Li Huang (huangli@caep.cn)
# Status  : Unstable
#
# Last modified: 2026/11/10
#

#=
*Remarks* : *Distribution Functions*

**Fermionic System**

For fermionic system, the basic distribution function reads:

```math
\begin{equation}
f_1(\beta,\omega) = \frac{1}{1 + e^{\beta\omega}}.
\end{equation}
```

We further define ``f_2(\beta,\tau,\omega)``:

```math
\begin{equation}
f_2(\beta,\tau,\omega) = f_1(\beta,\omega) e^{\tau\omega}.
\end{equation}
```

For numerical stability, we should use

```math
\begin{equation}
f_2(\beta,\tau,\omega) = \frac{e^{\tau\omega}}{ 1 + e^{\beta\omega} },
\end{equation}
```

for ``\omega < 0`` and

```math
\begin{equation}
f_2(\beta,\tau,\omega) = \frac{e^{(\tau-\beta)\omega}}{ 1 + e^{-\beta\omega} },
\end{equation}
```

for ``\omega > 0``.
=#

#=
### *Fermi-Dirac Distribution Function*
=#

"""
    fermi(β::T, ω::T) where {T}

Calculate the basic Fermi-Dirac distribution function f₁(β,ω).

### Arguments
* β -> Inverse temperature (β = 1/T).
* ω -> Energy (scalar value).

### Returns
* Value of the Fermi-Dirac distribution function: f₁(β,ω).

### Notes
For numerical stability, when |βω| > 100, the function returns:

- 0 for βω > 0

- 1 for βω < 0

This prevents overflow/underflow in the exponential calculation.

See also: [`FERMI`](@ref).
"""
function fermi(β::T, ω::T) where {T}
    arg = ω * β
    if abs(arg) > 100
        arg > 0 ? zero(T) : one(T)
    else
        one(T) / ( one(T) + exp(arg) )
    end
end

"""
    fermi(β::T, τ::T, ω::T) where {T}

Calculate the extended Fermi-Dirac distribution function f₂(β,τ,ω).

### Arguments
* β -> Inverse temperature (β = 1/T).
* τ -> Imaginary time.
* ω -> Energy (scalar value).

### Returns
* Value of the extended Fermi-Dirac distribution function: f₂(β,τ,ω).

### Notes
For numerical stability, different formulations are used depending on
the sign of ω:

- For ω < 0: f₂(β,τ,ω) = exp(τω) / (1 + exp(βω))

- For ω > 0: f₂(β,τ,ω) = exp((τ-β)ω) / (1 + exp(-βω))

This prevents overflow/underflow in the exponential calculation.

See also: [`FERMI`](@ref).
"""
function fermi(β::T, τ::T, ω::T) where {T}
    if ω < 0
        exp(ω*τ) * fermi(β, ω)
    else
        exp((τ - β) * ω) * fermi(β, -ω)
    end
end

"""
    fermi(β::T, ω::Vector{N}) where {T,N}

Calculate the basic Fermi-Dirac distribution function f₁(β,ω) for
multiple energies.

### Arguments
* β -> Inverse temperature (β = 1/T).
* ω -> Array of energies.

### Returns
* Vector of Fermi-Dirac distribution function values for each energy in ω.

### Notes
This function applies the scalar `fermi(β, ω)` function element-wise to
the input vector. Type conversion is performed automatically if T and
N differ.

See also: [`FERMI`](@ref).
"""
function fermi(β::T, ω::Vector{N}) where {T,N}
    if T == N
        [fermi(β, x) for x in ω]
    else
        [fermi(β, convert(T, x)) for x in ω]
    end
end

"""
    fermi(β::T, τ::T, ω::Vector{N}) where {T,N}

Calculate the extended Fermi-Dirac distribution function f₂(β,τ,ω) for
multiple energies.

### Arguments
* β -> Inverse temperature (β = 1/T).
* τ -> Imaginary time.
* ω -> Array of energies.

### Returns
* Vector of Fermi-Dirac distribution function values for each energy in ω.

### Notes
This function applies the scalar `fermi(β, τ, ω)` function element-wise
to the input vector. Type conversion is performed automatically if T and
N differ.

See also: [`FERMI`](@ref).
"""
function fermi(β::T, τ::T, ω::Vector{N}) where {T,N}
    if T == N
        [fermi(β, τ, x) for x in ω]
    else
        [fermi(β, τ, convert(T, x)) for x in ω]
    end
end

"""
    fermi_exp(β::T, τ::T, ω::T) where {T}

Calculate ``e^{\\tau\\omega} n_F(\\omega)`` in an overflow-safe way. It
is a wrapper of the extended Fermi-Dirac distribution function
`fermi(β, τ, ω)`.

### Arguments
* β -> Inverse temperature (β = 1/T).
* τ -> Imaginary time (0 ≤ τ ≤ β).
* ω -> Energy (scalar value).

### Returns
* Value of ``e^{\\tau\\omega} / (1 + e^{\\beta\\omega})``.

*References* :

Please see [`NESSi`] `cntr_equilibrium_impl.hpp`, `fermi_exp`.

See also: [`fermi`](@ref).
"""
fermi_exp(β::T, τ::T, ω::T) where {T} = fermi(β, τ, ω)

"""
    fermi_exp(β::T, τ::T, ω::Vector{N}) where {T,N}

Calculate `fermi_exp(β, τ, ω)` for multiple energies.

See also: [`fermi`](@ref).
"""
fermi_exp(β::T, τ::T, ω::Vector{N}) where {T,N} = fermi(β, τ, ω)
