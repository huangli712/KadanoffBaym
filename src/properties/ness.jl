#
# Project : Lavender
# Source  : ness.jl
# Author  : Li Huang (huangli@caep.cn)
# Status  : Unstable
#
# Last modified: 2026/11/10
#

#=
### *Indexing* : *Steady-state Green's Function*
=#

"""
    get_ret(G::HermMatrixNESS, n::I64, dom::FFTDomain)

Return a copy of the retarded matrix at the 1-based storage index `n`.
"""
get_ret(G::HermMatrixNESS, n::I64, dom::FFTDomain) = G.ret[n, dom]


"""
    get_les(G::HermMatrixNESS, n::I64, dom::FFTDomain)

Return a copy of the lesser matrix at the 1-based storage index `n`.
"""
get_les(G::HermMatrixNESS, n::I64, dom::FFTDomain) = G.les[n, dom]


"""
    set_ret!(G::HermMatrixNESS, n::I64, M, dom::FFTDomain)

Copy the matrix (or the scalar for 1 × 1 matrices) `M` to the retarded
component at the 1-based storage index `n`.
"""
function set_ret!(G::HermMatrixNESS, n::I64, M, dom::FFTDomain)
    G.ret[n, dom] = M
    return G
end


"""
    set_les!(G::HermMatrixNESS, n::I64, M, dom::FFTDomain)

Copy the matrix (or the scalar for 1 × 1 matrices) `M` to the lesser
component at the 1-based storage index `n`.
"""
function set_les!(G::HermMatrixNESS, n::I64, M, dom::FFTDomain)
    G.les[n, dom] = M
    return G
end

#=
### *Properties* : *NESS2 Grid*
=#

"""
    _time_mode(g::GridInfo, m::Integer)

Relative time of the signed mode `m` (0-based, periodic), see *Remarks* :
*NESS2 Grid*. It is a private function.
"""
function _time_mode(g::GridInfo, m::Integer)
    n = mod(m, g.Nft)
    return n < g.Nft ÷ 2 ? n * g.h : (n - g.Nft) * g.h
end


"""
    _freq_mode(g::GridInfo, m::Integer)

Frequency of the signed mode `m` (0-based, periodic), see *Remarks* :
*NESS2 Grid*. It is a private function.
"""
function _freq_mode(g::GridInfo, m::Integer)
    n = mod(m, g.Nft)
    return n < g.Nft ÷ 2 ? n * g.dw : (n - g.Nft) * g.dw
end


"""
    time_at(g::GridInfo, n::Integer)

Relative time of the 1-based storage index `n` (1 ≤ n ≤ Nft). The second
half of the grid represents the negative times.

See also: [`GridInfo`](@ref), [`freq_at`](@ref).
"""
function time_at(g::GridInfo, n::Integer)
    1 ≤ n ≤ g.Nft || throw(BoundsError(1:g.Nft, n))
    return _time_mode(g, n - 1)
end


"""
    freq_at(g::GridInfo, n::Integer)

Frequency of the 1-based storage index `n` (1 ≤ n ≤ Nft). The second
half of the grid represents the negative frequencies.

See also: [`GridInfo`](@ref), [`time_at`](@ref).
"""
function freq_at(g::GridInfo, n::Integer)
    1 ≤ n ≤ g.Nft || throw(BoundsError(1:g.Nft, n))
    return _freq_mode(g, n - 1)
end


"""
    set_h(g::GridInfo, h::Real)

Return a new GridInfo struct with the time step `h`, the frequency step
is updated accordingly.
"""
function set_h(g::GridInfo, h::Real)
    h > 0 || throw(ArgumentError("h must be positive, got $h"))
    return GridInfo(g.Nft, F64(h), π / ((g.Nft ÷ 2) * F64(h)))
end


"""
    set_dw(g::GridInfo, dw::Real)

Return a new GridInfo struct with the frequency step `dw`, the time step
is updated accordingly.
"""
function set_dw(g::GridInfo, dw::Real)
    dw > 0 || throw(ArgumentError("dw must be positive, got $dw"))
    return GridInfo(g.Nft, π / ((g.Nft ÷ 2) * F64(dw)), F64(dw))
end


#=
### *Properties* : *FFT Array*
=#

"""
    getnft(A::FFTArray)

Return the number of the grid points of an FFTArray struct.
"""
getnft(A::FFTArray) = size(A.time, 3)


"""
    _fft_data(A::FFTArray, dom::FFTDomain)

Return the time or frequency data of an FFTArray struct. It is a private
function.
"""
_fft_data(A::FFTArray, dom::FFTDomain) = dom == FFT_TIME ? A.time : A.freq


#=
### *Properties* : *Steady-state Green's Function*
=#

"""
    getnft(G::HermMatrixNESS)

Return the number of the grid points of a HermMatrixNESS struct.
"""
getnft(G::HermMatrixNESS) = G.grid.Nft


