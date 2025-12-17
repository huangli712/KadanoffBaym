#
# Project : Lavender
# Source  : math.jl
# Author  : Li Huang (huangli@caep.cn)
# Status  : Unstable
#
# Last modified: 2025/12/17
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
"""
const FERMI = -1

"""
    BOSE

Basic physical constant. It is used to denote the bosonic system.
"""
const BOSE = 1

#=
### *Fermi-Dirac Distribution Function*
=#

"""
    fermi(β::T, ω::T)

Try to calculate basic Fermi-Dirac distribution function: f₁(β,ω). Here,
ω is a scalar number.
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
    fermi(β::T, τ::T, ω::T)

Try to calculate extended Fermi-Dirac distribution function: f₂(β,τ,ω).
Actually, f₂(β,τ,ω) ≡ f₁(β,ω)exp(τω). Here, ω is a scalar number.
"""
function fermi(β::T, τ::T, ω::T) where {T}
    if ω < 0
        exp(ω*τ) * fermi(β, ω)
    else
        exp((τ - β) * ω) * fermi(β, -ω)
    end
end

"""
    fermi(β::T, ω::Vector{N})

Try to calculate basic Fermi-Dirac distribution function: f₁(β,ω). Here,
ω is a vector.
"""
function fermi(β::T, ω::Vector{N}) where {T,N}
    if T == N
        [fermi(β, x) for x in ω]
    else
        [fermi(β, convert(T, x)) for x in ω]
    end
end

"""
    fermi(β::T, τ::T, ω::Vector{N})

Try to calculate extended Fermi-Dirac distribution function: f₂(β,τ,ω).
Actually, f₂(β,τ,ω) ≡ f₁(β,ω)exp(τω). Here, ω is a vector.
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
    bose(β::T, ω::T)

Try to calculate basic Bose-Einstein distribution function: b₁(β,ω). Here,
ω is a scalar number.
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
    bose(β::T, τ::T, ω::T)

Try to calculate extended Bose-Einstein distribution function: b₂(β,τ,ω).
Actually, b₂(β,τ,ω) ≡ b₁(β,ω)exp(τω). Here, ω is a scalar number.
"""
function bose(β::T, τ::T, ω::T) where {T}
    if ω < 0
        exp(τ * ω) * bose(β, ω)
    else
        -exp((τ - β) * ω) * bose(β, -ω)
    end
end

"""
    bose(β::T, ω::Vector{N})

Try to calculate basic Bose-Einstein distribution function: b₁(β,ω). Here,
ω is a vector.
"""
function bose(β::T, ω::Vector{N}) where {T,N}
    if T == N
        [bose(β, x) for x in ω]
    else
        [bose(β, convert(T, x)) for x in ω]
    end
end

"""
    bose(β::T, τ::T, ω::Vector{N})

Try to calculate extended Bose-Einstein distribution function: b₂(β,τ,ω).
Actually, b₂(β,τ,ω) ≡ b₁(β,ω)exp(τω). Here, ω is a vector.
"""
function bose(β::T, τ::T, ω::Vector{N}) where {T,N}
    if T == N
        [bose(β, τ, x) for x in ω]
    else
        [bose(β, τ, convert(T, x)) for x in ω]
    end
end
