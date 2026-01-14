#
# Project : Lavender
# Source  : math.jl
# Author  : Li Huang (huangli@caep.cn)
# Status  : Unstable
#
# Last modified: 2026/01/14
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

**Bosonic System**

The basic Bose-Einstein distribution reads:

```math
\begin{equation}
b_1(\beta,\omega) = \frac{1}{e^{\beta\omega} - 1}.
\end{equation}
```

We should define ``b_2(\beta,\tau,\omega)`` as well:

```math
\begin{equation}
b_2(\beta,\tau,\omega) = b_1(\beta,\omega) e^{\tau\omega}.
\end{equation}
```

For numerical stability, we should use

```math
\begin{equation}
b_2(\beta,\tau,\omega) = \frac{e^{\tau\omega}}{ e^{\beta\omega} - 1 },
\end{equation}
```

for ``\omega < 0`` and

```math
\begin{equation}
b_2(\beta,\tau,\omega) = \frac{e^{(\tau-\beta)\omega}}{ 1 - e^{-\beta\omega}},
\end{equation}
```

for ``\omega > 0``.
=#

#=
### *Basic Physical Constants*
=#

"""
    FERMI

Basic physical constant. It is used to denote the fermionic system.

### Notes

This constant is used throughout the codebase to identify the fermionic
systems and to setup correct sign in calculations.

See also: [`BOSE`](@ref).
"""
const FERMI = -1

"""
    BOSE

Basic physical constant. It is used to denote the bosonic system.

### Notes

This constant is used throughout the codebase to identify the bosonic
systems and to setup correct sign in calculations.

See also: [`FERMI`](@ref).
"""
const BOSE = 1

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

#=
### *Bose-Einstein Distribution Function*
=#

"""
    bose(β::T, ω::T) where {T}

Calculate the basic Bose-Einstein distribution function b₁(β,ω).

### Arguments
* β -> Inverse temperature (β = 1/kᵦT).
* ω -> Energy (scalar value).

### Returns
* Value of the Bose-Einstein distribution function: b₁(β,ω) = 1/(exp(βω) - 1).

### Notes

For numerical stability and to handle negative energies:
- For ω < 0: Uses the relation b₁(β,ω) = -1 - b₁(β,-ω)
- For |βω| > 100: Returns 0 to prevent overflow
- For βω < 1.0e-10: Uses the approximation 1/(βω) to avoid division by zero

This implementation carefully handles the singularity at ω = 0.

See also: [`fermi`](@ref), [`BOSE`](@ref).
"""
function bose(β::T, ω::T) where {T}
    arg = ω * β
    if arg < 0
        return -one(T) - bose(β, -ω)
    end

    if abs(arg) > 100
        return zero(T)
    elseif arg < 1.0e-10
        return one(T) / arg
    else
        return one(T) / ( exp(arg) - one(T) )
    end
end

"""
    bose(β::T, τ::T, ω::T) where {T}

Calculate the extended Bose-Einstein distribution function b₂(β,τ,ω).

### Arguments
* β -> Inverse temperature (β = 1/kᵦT).
* τ -> Imaginary time.
* ω -> Energy (scalar value).

### Returns
* Value of the extended Bose-Einstein distribution function: b₂(β,τ,ω) = b₁(β,ω)exp(τω).

### Notes

For numerical stability, different formulations are used depending on the sign of ω:
- For ω < 0: b₂(β,τ,ω) = exp(τω) / (exp(βω) - 1)
- For ω > 0: b₂(β,τ,ω) = -exp((τ-β)ω) / (1 - exp(-βω))

This prevents overflow/underflow in the exponential calculation.

See also: [`fermi`](@ref), [`BOSE`](@ref).
"""
function bose(β::T, τ::T, ω::T) where {T}
    if ω < 0
        exp(τ * ω) * bose(β, ω)
    else
        -exp((τ - β) * ω) * bose(β, -ω)
    end
end

"""
    bose(β::T, ω::Vector{N}) where {T,N}

Calculate the basic Bose-Einstein distribution function b₁(β,ω) for multiple energies.

### Arguments
* β -> Inverse temperature (β = 1/kᵦT).
* ω -> Array of energies.

### Returns
* Vector of Bose-Einstein distribution function values for each energy in ω.

### Notes

This function applies the scalar `bose(β, ω)` function element-wise to the
input vector. Type conversion is performed automatically if T and N differ.

See also: [`fermi`](@ref), [`BOSE`](@ref).
"""
function bose(β::T, ω::Vector{N}) where {T,N}
    if T == N
        [bose(β, x) for x in ω]
    else
        [bose(β, convert(T, x)) for x in ω]
    end
end

"""
    bose(β::T, τ::T, ω::Vector{N}) where {T,N}

Calculate the extended Bose-Einstein distribution function b₂(β,τ,ω) for multiple energies.

### Arguments
* β -> Inverse temperature (β = 1/kᵦT).
* τ -> Imaginary time.
* ω -> Array of energies.

### Returns
* Vector of extended Bose-Einstein distribution function values for each energy in ω.

### Notes

This function applies the scalar `bose(β, τ, ω)` function element-wise to the
input vector. Type conversion is performed automatically if T and N differ.

See also: [`fermi`](@ref), [`BOSE`](@ref).
"""
function bose(β::T, τ::T, ω::Vector{N}) where {T,N}
    if T == N
        [bose(β, τ, x) for x in ω]
    else
        [bose(β, τ, convert(T, x)) for x in ω]
    end
end
