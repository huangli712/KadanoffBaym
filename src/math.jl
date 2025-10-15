#
# Project : Lavender
# Source  : KadanoffBaym.jl
# Author  : Li Huang (huangli@caep.cn)
# Status  : Unstable
#
# Last modified: 2025/10/15
#

#=
### *Fermi-Dirac Statistics*
=#

function fermi(β::T, ω::T) where {T}
    arg = ω * β
    if abs(arg) > 100
        arg > 0 ? zero(T) : one(T)
    else
        one(T) / ( one(T) + exp(arg) )
    end
end

function fermi(β::T, τ::T, ω::T) where {T}
    if ω < 0
        exp(ω*τ) * fermi(β, ω)
    else
        exp((τ - β) * ω) * fermi(β, -ω)
    end
end

function fermi(β::T, ω::Vector{N}) where {T,N}
    if T == N
        [fermi(β, x) for x in ω]
    else
        [fermi(β, convert(T, x)) for x in ω]
    end
end

function fermi(β::T, τ::T, ω::Vector{N}) where {T,N}
    if T == N
        [fermi(β, τ, x) for x in ω]
    else
        [fermi(β, τ, convert(T, x)) for x in ω]
    end
end

#=
### *Bose Statistics*
=#

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

function bose(β::T, τ::T, ω::T) where {T}
    if ω < 0
        exp(τ * ω) * bose(β, ω)
    else
        -exp((τ - β) * ω) * bose(β, -ω)
    end
end

function bose(β::T, ω::Vector{N}) where {T,N}
    if T == N
        [bose(β, x) for x in ω]
    else
        [bose(β, convert(T, x)) for x in ω]
    end
end

function bose(β::T, τ::T, ω::Vector{N}) where {T,N}
    if T == N
        [bose(β, τ, x) for x in ω]
    else
        [bose(β, τ, convert(T, x)) for x in ω]
    end
end
