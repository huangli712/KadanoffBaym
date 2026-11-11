#
# Project : Lavender
# Source  : gw_kpoints.jl
# Author  : Li Huang (huangli@caep.cn)
# Status  : Unstable
#
# Last modified: 2026/11/10
#

#
# Propagators of one k point of the GW example and the solution of the
# Dyson equation (G) and of the VIE2 with singular kernel (W) at a time
# step. Port of NESSi examples/programs/gw_kpoints_impl.cpp. KB time step
# n = NESSi time step + 1 (n = 0 is the Matsubara branch).
#

using LinearAlgebra
using KadanoffBaym

"""
    GWKPoint

Contour functions of one k point: `G`, `Σ` (GW part), `P`, `PV = -P V`,
`VP = -V P`, `W` (regular part), the singular part `Wsin` of W, the
single-time functions `SH`, `SF` (Hartree, Fock), `rho` (density after
the Dyson step), `hk`, `hkeff = hk + SH + SF`, `vertex`, and the
workspace of the VIE2 with singular kernel.
"""
struct GWKPoint
    k      :: Float64
    G      :: ℱ{ComplexF64}
    Σ      :: ℱ{ComplexF64}
    P      :: ℱ{ComplexF64}
    PV     :: ℱ{ComplexF64}
    VP     :: ℱ{ComplexF64}
    W      :: ℱ{ComplexF64}
    Wsin   :: Cf{ComplexF64}
    SH     :: Cf{ComplexF64}
    SF     :: Cf{ComplexF64}
    rho    :: Cf{ComplexF64}
    hk     :: Cf{ComplexF64}
    hkeff  :: Cf{ComplexF64}
    vertex :: Cf{ComplexF64}
    ws     :: VIE2SinWorkspace
end

"""
    GWKPoint(C, lat, j)

Construct the k point `j` (1-based) of the lattice `lat` on the contour `C`.
"""
function GWKPoint(C::Cn, lat::GWLattice, j::Int)
    ntime = C.ntime
    k = lat.kpoints[j]
    cf() = Cf(ntime, 1, 1, zero(ComplexF64))
    hk = cf()
    vertex = cf()
    for n = 0:ntime
        hk[n] = fill(ComplexF64(dispersion(lat, k, n)), 1, 1)
        vertex[n] = fill(ComplexF64(interaction_vertex(lat, k)), 1, 1)
    end
    hkeff = cf()
    memcpy!(hk, hkeff)
    W = ℱ(C, BOSE)
    return GWKPoint(k, ℱ(C, FERMI), ℱ(C, FERMI), ℱ(C, BOSE), ℱ(C, BOSE), ℱ(C, BOSE),
                    W, cf(), cf(), cf(), cf(), hk, hkeff, vertex, VIE2SinWorkspace(W))
end

# Time steps touched by a solver call at the KB time step n (the start-up
# steps 1 ⋯ order + 1 are solved together)
_steprange(n::Int, order::Int) = (n == 0 || n > order + 1) ? (n:n) : (1:order+1)

"""
    step_w!(kp, n, order, β, h, Fsin, Q)

Solve ``W = V + V P W`` at the KB time step `n` as the VIE2 with singular
kernel ``[1 - V P] W = V``, i.e. `F = VP = -V P`, `Fcc = PV = -P V`,
`Qsin = V`, the regular source `Q = 0`, and `Fsin = 0`. `P` must be known
at the touched time steps.

*References* : NESSi `gw::kpoint::step_W`.
"""
function step_w!(kp::GWKPoint, n::Int, order::Int, β::Float64, h::Float64,
                 Fsin::Cf, Q::ℱ)
    for m in _steprange(n, order)
        memcpy!(kp.P, kp.PV, m)
        smul!(kp.PV, kp.vertex, m)
        smul!(kp.PV, m, -1.0)
        memcpy!(kp.P, kp.VP, m)
        smul!(kp.vertex, kp.VP, m)
        smul!(kp.VP, m, -1.0)
    end
    vie2_timestep_sin!(n, kp.W, kp.Wsin, kp.VP, kp.PV, Fsin, Q, kp.vertex,
                       Integrator(order), β, h; ws = kp.ws)
    return kp
end

"""
    step_dyson!(kp, n, iter, lat, μ, β, h, order; threaded = false)

Solve the Dyson equation ``G = [i\\partial_t + \\mu - h_k - \\Sigma^H -
\\Sigma^F - \\Sigma]^{-1}`` at the KB time step `n` and store the density
matrices of the touched time steps in `kp.rho`. As in NESSi, the second
Matsubara iteration (`iter = 1`) uses the noninteracting Green's function
of `hkeff`.

*References* : NESSi `gw::kpoint::step_dyson`.
"""
function step_dyson!(kp::GWKPoint, n::Int, iter::Int, lat::GWLattice, μ::Float64,
                     β::Float64, h::Float64, order::Int; threaded::Bool = false)
    r = _steprange(n, order)
    for m in r
        kp.hk[m] = fill(ComplexF64(dispersion(lat, kp.k, m)), 1, 1)
        kp.hkeff[m] = kp.SH[m] + kp.SF[m] + kp.hk[m]
    end
    if n == 0 && iter == 1
        # Matsubara component of the noninteracting Green's function
        C₁ = Cn(2, getntau(kp.G), 1, 1, h, β)
        T = ℱ(C₁, FERMI)
        from_ham!(T, μ, kp.hkeff[0], β, h)
        memcpy!(T.mat, kp.G.mat)
    elseif n == 0
        dyson_mat!(kp.G, kp.Σ, μ, kp.hkeff, β; solve_order = order)
    elseif n ≤ order + 1
        dyson_start!(kp.G, μ, kp.hkeff, kp.Σ, β, h; solve_order = order)
    else
        dyson_timestep!(n, kp.G, μ, kp.hkeff, kp.Σ, β, h; solve_order = order,
                        threaded = threaded)
    end
    for m in r
        kp.rho[m] = density_matrix(kp.G, m)
    end
    return kp
end

"""
    step_dyson_with_error!(kp, n, iter, lat, μ, β, h, order; threaded)

`step_dyson!` and the distance of the time step `n` before and after.
"""
function step_dyson_with_error!(kp::GWKPoint, n::Int, iter::Int, lat::GWLattice,
                                μ::Float64, β::Float64, h::Float64, order::Int;
                                threaded::Bool = false)
    g = 𝒻(n, getntau(kp.G), 1, 1, FERMI)
    memcpy!(kp.G, g, n)
    step_dyson!(kp, n, iter, lat, μ, β, h, order; threaded = threaded)
    return distance(g, kp.G, n)
end

"""
    step_w_with_error!(kp, n, order, β, h, Fsin, Q)

`step_w!` and the distance of the time step `n` before and after.
"""
function step_w_with_error!(kp::GWKPoint, n::Int, order::Int, β::Float64, h::Float64,
                            Fsin::Cf, Q::ℱ)
    w = 𝒻(n, getntau(kp.W), 1, 1, BOSE)
    memcpy!(kp.W, w, n)
    step_w!(kp, n, order, β, h, Fsin, Q)
    return distance(w, kp.W, n)
end
