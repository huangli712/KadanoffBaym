#
# Project : Lavender
# Source  : gw_lattice.jl
# Author  : Li Huang (huangli@caep.cn)
# Status  : Unstable
#
# Last modified: 2026/11/10
#

#
# One-dimensional single band lattice of the GW example: k grid,
# dispersion with the Peierls phase, velocities, interaction vertex, and
# the vector potential of the electric field pulse.
#
# Port of NESSi examples/programs/gw_latt_impl.cpp. Time steps: the KB
# time step 0 is the Matsubara branch, the KB time step n ≥ 1 is the
# NESSi time step n - 1, i.e. t = (n - 1) h.
#

using KadanoffBaym

"""
    GWLattice

Lattice of the GW example. `A[n + 1]` is the vector potential at the KB
time step `n` (`A[1] = 0` on the Matsubara branch).
"""
struct GWLattice
    nk       :: Int
    G        :: Int
    kpoints  :: Vector{Float64}
    kweights :: Vector{Float64}
    th       :: Float64
    U        :: Float64
    V        :: Float64
    A        :: Vector{Float64}
end

"""
    efield_to_afield(E, h, k)

Vector potential ``A(t) = -\\int_0^t E(t') dt'`` by Gregory integration of
order `min(k, Nt)`. `E[n + 1]` is the field at the KB time step `n`
(`E[1]` on the Matsubara branch is not used). Return `A` with the same
indexing, `A[1] = 0`.

*References* : NESSi `lattice_1d_1b::efield_to_afield`.
"""
function efield_to_afield(E::Vector{Float64}, h::Float64, k::Int)
    ntime = length(E) - 1
    Nt = ntime - 1
    Nt ≥ 1 || throw(ArgumentError("at least two real time points are needed"))
    kt₁ = Nt ≥ k ? k : Nt
    I = Integrator(kt₁)
    A = zeros(Float64, ntime + 1)
    for tstp = 0:Nt
        n₁ = tstp < kt₁ ? kt₁ : tstp
        a = 0.0
        for n = 0:n₁
            a += I.GIW[tstp,n] * E[n+2]
        end
        A[tstp+2] = -h * a
    end
    return A
end

"""
    build_kgrid(nk)

Full Brillouin zone ``k_j = (j - G) 2\\pi / N_k`` (`j = 0, ⋯, N_k - 1`,
`G = N_k ÷ 2`) and the weights ``1/N_k``.
"""
function build_kgrid(nk::Int)
    nk > 1 || throw(ArgumentError("nk must be > 1"))
    G = nk ÷ 2
    kpoints = [(j - G) * 2π / nk for j = 0:nk-1]
    return G, kpoints, fill(1.0 / nk, nk)
end

"""
    GWLattice(nk, th, U, V, E, h, k)

Construct the lattice from the number of k points, the hopping, the
interactions, the electric field `E` (see `efield_to_afield`), the time
step, and the integration order.
"""
function GWLattice(nk::Int, th::Float64, U::Float64, V::Float64,
                   E::Vector{Float64}, h::Float64, k::Int)
    G, kp, kw = build_kgrid(nk)
    return GWLattice(nk, G, kp, kw, th, U, V, efield_to_afield(E, h, k))
end

"""
    add_kpoints(lat, k1, s1, k2, s2)

Index (1-based) of the momentum ``s_1 k_1 + s_2 k_2`` folded back to the
Brillouin zone.

*References* : NESSi `lattice_1d_1b::add_kpoints`.
"""
function add_kpoints(lat::GWLattice, k1::Int, s1::Int, k2::Int, s2::Int)
    k12 = s1 * (k1 - 1 - lat.G) + s2 * (k2 - 1 - lat.G) + lat.G
    return mod(k12, lat.nk) + 1
end

"""
    dispersion(lat, k, n)

``\\epsilon_k(t) = -2 t_h \\cos(k - A(t))`` at the KB time step `n`.
"""
dispersion(lat::GWLattice, k::Float64, n::Int) = -2.0 * lat.th * cos(k - lat.A[n+1])

"""
    velocity_nessi(lat, k)

Velocity of NESSi, ``2 t_h \\sin(k)``, without the vector potential.
"""
velocity_nessi(lat::GWLattice, k::Float64) = 2.0 * lat.th * sin(k)

"""
    velocity_peierls(lat, k, n)

Gauge invariant velocity ``\\partial\\epsilon_k/\\partial k = 2 t_h \\sin(k - A(t))``
at the KB time step `n` (opt-in, `current_mode = "peierls_corrected"`).
"""
velocity_peierls(lat::GWLattice, k::Float64, n::Int) = 2.0 * lat.th * sin(k - lat.A[n+1])

"""
    interaction_vertex(lat, q)

``V_q = U + 2 V \\cos(q)``.
"""
interaction_vertex(lat::GWLattice, q::Float64) = lat.U + 2.0 * lat.V * cos(q)
