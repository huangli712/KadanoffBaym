#
# Project : Lavender
# Source  : test_nonequilibrium.jl
# Author  : Li Huang (huangli@caep.cn)
# Status  : Unstable
#
# Last modified: 2026/11/09
#

#
# Usage:
#
#   julia --project=. tutor/test_nonequilibrium/test_nonequilibrium.jl \
#         tutor/test_nonequilibrium/test_nonequilibrium.toml out.dat
#
# A single level `eps1` coupled to a second level `eps2` (hopping
# `i lambda`) is solved by embedding with Σ = lambda² g2, where g2 is the
# free Green's function of `eps2`. Two methods are compared with the
# (1,1) element of the exact 2 × 2 Green's function:
#
#   * the Dyson equation (integro-differential form), dyson_mat!(),
#     dyson_start!(), dyson_timestep!();
#   * the VIE2 (integral form) [1 + F] * G = G0 with F = -G0 * Σ and
#     Fcc = -Σ * G0, vie2_mat!(), vie2_start!(), vie2_timestep!().
#
# Output: one line `err_dyson err_vie2`. The error is the normalized
# squared norm of NESSi examples/programs/test_nonequilibrium.cpp:
#
#   err = Σ_t [2 d_les(t) / Nt² + 2 d_ret(t) / Nt² + d_tv(t) / (Nt Ntau)],
#
# where d_X are the block Frobenius distances of the components.
#
# Port of NESSi examples/programs/test_nonequilibrium.cpp.
#

using Printf
using KadanoffBaym

"""
    main(cfgfile::String, outfile::String)

Read the parameters from `cfgfile` (TOML) and write the errors of the
Dyson and VIE2 solutions into `outfile` (overwritten).
"""
function main(cfgfile::String, outfile::String)
    p = inp_toml(cfgfile, true)
    Nt = p["Nt"]::Int
    Ntau = p["Ntau"]::Int
    Tmax = Float64(p["Tmax"])
    k = p["SolveOrder"]::Int
    β = Float64(p["beta"])
    ε₁ = Float64(p["eps1"])
    ε₂ = Float64(p["eps2"])
    λ = Float64(p["lambda"])
    μ = Float64(p["mu"])
    @assert iseven(Ntau) && Ntau ≥ 4
    @assert 1 ≤ k ≤ 5 && Nt ≥ k

    # Nt and Ntau are the numbers of intervals
    h = Tmax / Nt
    nt = Nt + 1
    ntau = Ntau + 1
    C1 = Cn(nt, ntau, 1, 1, Tmax, β)
    C2 = Cn(nt, ntau, 2, 2, Tmax, β)

    # Exact Green's function of the 2 × 2 Hamiltonian
    G2 = ℱ(C2, FERMI)
    from_ham!(G2, μ, ComplexF64[ε₁ im*λ; -im*λ ε₂], β, h)
    Gx = ℱ(C1, FERMI)
    for m = 1:ntau
        Gx.mat[m] = G2.mat[m][1:1, 1:1]
    end
    for i = 1:nt
        for m = 1:ntau
            Gx.lmix[i, m] = G2.lmix[i, m][1:1, 1:1]
        end
        for j = 1:i
            Gx.ret[i, j] = G2.ret[i, j][1:1, 1:1]
            Gx.less[j, i] = G2.less[j, i][1:1, 1:1]
        end
    end

    # Embedding self-energy and Hamiltonian
    Σ = ℱ(C1, FERMI)
    from_ham!(Σ, μ, fill(ComplexF64(ε₂), 1, 1), β, h)
    for t = 0:nt
        smul!(Σ, t, λ^2)
    end
    H = Cf(nt, 1, 1)
    for i = 0:nt
        H[i] = fill(ComplexF64(ε₁), 1, 1)
    end

    error_of(G) = sum(2.0 * distance(Gx, G, t, Val(:les)) / Nt^2 +
                      2.0 * distance(Gx, G, t, Val(:ret)) / Nt^2 +
                      distance(Gx, G, t, Val(:tv)) / (Nt * Ntau) for t = 1:nt)

    # Dyson equation
    G = ℱ(C1, FERMI)
    dyson_mat!(G, Σ, μ, H, β; solve_order = k)
    dyson_start!(G, μ, H, Σ, β, h; solve_order = k)
    for tstp = k+2:nt
        dyson_timestep!(tstp, G, μ, H, Σ, β, h; solve_order = k)
    end
    err_dyson = error_of(G)

    # VIE2, the kernels are built step by step as in NESSi
    I = Integrator(k)
    G0 = ℱ(C1, FERMI)
    from_ham!(G0, μ, fill(ComplexF64(ε₁), 1, 1), β, h)
    F = ℱ(C1, FERMI)
    Fcc = ℱ(C1, FERMI)
    function kernel!(tstp)
        if tstp == 0
            conv_mat(F.mat, G0.mat, Σ.mat, I, β, FERMI)
            conv_mat(Fcc.mat, Σ.mat, G0.mat, I, β, FERMI)
        else
            convolution_time_step(tstp, F, G0, G0, Σ, Σ, I, β, h)
            convolution_time_step(tstp, Fcc, Σ, Σ, G0, G0, I, β, h)
        end
        smul!(F, tstp, -1.0)
        smul!(Fcc, tstp, -1.0)
    end
    G = ℱ(C1, FERMI)
    kernel!(0)
    vie2_mat!(G, F, Fcc, G0, β, I)
    for tstp = 1:k+1
        kernel!(tstp)
    end
    vie2_start!(G, F, Fcc, G0, I, β, h)
    for tstp = k+2:nt
        kernel!(tstp)
        vie2_timestep!(tstp, G, F, Fcc, G0, I, β, h)
    end
    err_vie2 = error_of(G)

    open(outfile, "w") do io
        println(io, "# err_dyson err_vie2")
        @printf(io, "%.5e %.5e\n", err_dyson, err_vie2)
    end
    @printf("Nt = %d, Ntau = %d, SolveOrder = %d: err_dyson = %.5e, err_vie2 = %.5e\n",
            Nt, Ntau, k, err_dyson, err_vie2)

    return err_dyson, err_vie2
end

if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) == 2 || error("usage: test_nonequilibrium.jl <config.toml> <output.dat>")
    main(ARGS[1], ARGS[2])
end
