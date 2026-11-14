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
    get_ret(G::NessHermMat, n::I64, dom::FFTDomain)

Return a copy of the retarded matrix at the 1-based storage index `n`.
"""
get_ret(G::NessHermMat, n::I64, dom::FFTDomain) = G.ret[n, dom]

"""
    get_les(G::NessHermMat, n::I64, dom::FFTDomain)

Return a copy of the lesser matrix at the 1-based storage index `n`.
"""
get_les(G::NessHermMat, n::I64, dom::FFTDomain) = G.les[n, dom]

"""
    set_ret!(G::NessHermMat, n::I64, M, dom::FFTDomain)

Copy the matrix (or the scalar for 1 × 1 matrices) `M` to the retarded
component at the 1-based storage index `n`.
"""
function set_ret!(G::NessHermMat, n::I64, M, dom::FFTDomain)
    G.ret[n, dom] = M
    return G
end

"""
    set_les!(G::NessHermMat, n::I64, M, dom::FFTDomain)

Copy the matrix (or the scalar for 1 × 1 matrices) `M` to the lesser
component at the 1-based storage index `n`.
"""
function set_les!(G::NessHermMat, n::I64, M, dom::FFTDomain)
    G.les[n, dom] = M
    return G
end
