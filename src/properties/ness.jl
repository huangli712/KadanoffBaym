#
# Project : Lavender
# Source  : ness.jl
# Author  : Li Huang (huangli@caep.cn)
# Status  : Unstable
#
# Last modified: 2026/11/10
#

#=
### *Properties* : *NESS2 Grid*
=#

"""
    _time_mode(g::NessGrid, m::Integer)

Relative time of the signed mode `m` (0-based, periodic), see *Remarks* :
*NESS2 Grid*. It is a private function.
"""
function _time_mode(g::NessGrid, m::Integer)
    n = mod(m, g.Nft)
    return n < g.Nft ÷ 2 ? n * g.h : (n - g.Nft) * g.h
end

"""
    _freq_mode(g::NessGrid, m::Integer)

Frequency of the signed mode `m` (0-based, periodic), see *Remarks* :
*NESS2 Grid*. It is a private function.
"""
function _freq_mode(g::NessGrid, m::Integer)
    n = mod(m, g.Nft)
    return n < g.Nft ÷ 2 ? n * g.dw : (n - g.Nft) * g.dw
end

"""
    time_at(g::NessGrid, n::Integer)

Relative time of the 1-based storage index `n` (1 ≤ n ≤ Nft). The second
half of the grid represents the negative times.

See also: [`NessGrid`](@ref), [`freq_at`](@ref).
"""
function time_at(g::NessGrid, n::Integer)
    1 ≤ n ≤ g.Nft || throw(BoundsError(1:g.Nft, n))
    return _time_mode(g, n - 1)
end

"""
    freq_at(g::NessGrid, n::Integer)

Frequency of the 1-based storage index `n` (1 ≤ n ≤ Nft). The second
half of the grid represents the negative frequencies.

See also: [`NessGrid`](@ref), [`time_at`](@ref).
"""
function freq_at(g::NessGrid, n::Integer)
    1 ≤ n ≤ g.Nft || throw(BoundsError(1:g.Nft, n))
    return _freq_mode(g, n - 1)
end

#=
### *Properties* : *FFT Array*
=#

"""
    getnft(A::NessArray)

Return the number of the grid points of an NessArray struct.
"""
getnft(A::NessArray) = size(A.time, 3)

#=
### *Properties* : *Steady-state Green's Function*
=#

"""
    getnft(G::NessHermMat)

Return the number of the grid points of a NessHermMat struct.
"""
getnft(G::NessHermMat) = G.grid.Nft
