#
# t_p5_holstein.jl
#
# Unit tests of the Holstein helpers (tutor/holstein_bethe/holstein_impl.jl,
# holstein_nambu_impl.jl, holstein_energy.jl): zero coupling, D = D0 for
# Pi = 0, the phonon displacement, the bare phonon energy, and the normal
# limit of the Nambu self-energies.
#

using Test
using LinearAlgebra
using KadanoffBaym

module P5Holstein
    include(joinpath(@__DIR__, "..", "..", "tutor", "holstein_bethe", "holstein_nambu_impl.jl"))
    include(joinpath(@__DIR__, "..", "..", "tutor", "holstein_bethe", "holstein_energy.jl"))
end

using .P5Holstein: holstein_phonon, holstein_scalar_cf, sigma_mig!, sigma_umig!,
                   sigma_mig_start!, sigma_umig_start!, sigma_mig_sc!, sigma_umig_sc!,
                   sigma_mig_sc_start!, sigma_umig_sc_start!, step_D!, start_D!,
                   gregory_integrate, phonon_displacement!, phonon_energy_quantum

const P5H = (ntime = 8, ntau = 21, k = 3, β = 4.0, h = 0.05, w0 = 1.0)

p5h_cn(size) = Cn(P5H.ntime, P5H.ntau, size, size, (P5H.ntime - 1) * P5H.h, P5H.β)

# Free Green's function of a level ε (size 1) or of diag(ε, ε) (size 2)
function p5h_green(ε, size)
    G = ℱ(p5h_cn(size), FERMI)
    green_from_H!(G, 0.0, Matrix{ComplexF64}(ε * I, size, size), P5H.β, P5H.h)
    return G
end

p5h_phonon(size) = holstein_phonon(P5H.ntime, P5H.ntau, size, P5H.w0, P5H.β, P5H.h, P5H.k)

# Maximal difference of two contour functions over the time steps ns
p5h_dist(A, B, ns) = maximum(distance(A, B, n) for n in ns)

@testset verbose = true "Phase 5: Holstein helpers" begin
    @testset "zero coupling gives zero self-energy" begin
        G = p5h_green(0.3, 1)
        g = holstein_scalar_cf(P5H.ntime, zeros(P5H.ntime + 1))
        for f! in (sigma_mig!, sigma_umig!)
            Σ = ℱ(p5h_cn(1), FERMI)
            ph = p5h_phonon(1)
            f!(0, G, Σ, g, ph)
            (f! === sigma_mig! ? sigma_mig_start! : sigma_umig_start!)(G, Σ, g, ph)
            @test p5h_dist(Σ, ℱ(p5h_cn(1), FERMI), 0:P5H.k+1) == 0.0
        end
    end

    @testset "D = D0 for Pi = 0" begin
        ph = p5h_phonon(1)
        step_D!(0, ph)
        start_D!(ph)
        @test p5h_dist(ph.D, ph.D0, 0:P5H.k+1) < 1e-10
        step_D!(P5H.k + 2, ph)
        @test distance(ph.D, ph.D0, P5H.k + 2) < 1e-10
    end

    @testset "phonon Dyson equation with a nonzero kernel" begin
        # The real-time VIE2 (start-up collocation, retarded time stepping)
        # and the convolution weights agree up to the discretization error,
        # O(h^(k+1)); with h = 0.01 the residual is below 1e-10 (8e-10 for
        # h = 0.05)
        h = 0.01
        ph = holstein_phonon(P5H.ntime, P5H.ntau, 1, P5H.w0, P5H.β, h, P5H.k)
        # Small bosonic Pi: 0.05 times a single-pole XX propagator
        green_single_pole!(ph.Pi, 1.5, P5H.β, h)
        foreach(n -> smul!(ph.Pi, n, 0.05), 0:P5H.ntime)
        step_D!(0, ph)
        start_D!(ph)
        foreach(n -> step_D!(n, ph), P5H.k+2:P5H.ntime)
        # Residual of D - D0 * Pi * D = D0 (D0_Pi = -D0 * Pi, Pi_D0 = -Pi * D0)
        C = ℱ(Cn(P5H.ntime, P5H.ntau, 1, 1, (P5H.ntime - 1) * h, P5H.β), BOSE)
        convolution(C, ph.D0_Pi, ph.Pi_D0, ph.D, ph.D, ph.I, P5H.β, h)
        rm = maximum(abs(ph.D.mat[m][1,1] + C.mat[m][1,1] - ph.D0.mat[m][1,1]) for m = 1:P5H.ntau)
        rt = 0.0
        for n = 1:P5H.ntime, j = 1:n
            rt = max(rt, abs(ph.D.ret[n, j][1,1] + C.ret[n, j][1,1] - ph.D0.ret[n, j][1,1]))
            rt = max(rt, abs(ph.D.less[j, n][1,1] + C.less[j, n][1,1] - ph.D0.less[j, n][1,1]))
        end
        @test rm < 1e-10
        @test rt < 1e-10
        # D differs from D0
        @test p5h_dist(ph.D, ph.D0, 0:P5H.ntime) > 1e-3
    end

    @testset "Gregory integration and phonon displacement" begin
        ph = p5h_phonon(1)
        # Polynomials of degree ≤ k are integrated exactly
        for n = 1:7
            f = [ComplexF64((j * P5H.h)^3 - 2(j * P5H.h)) for j = 0:max(n, P5H.k)]
            @test abs(gregory_integrate(ph.I, f, n) * P5H.h -
                      ((n * P5H.h)^4 / 4 - (n * P5H.h)^2)) < 1e-12
        end
        # Constant density: constant displacement X0 = -2 g n / w0
        nt = P5H.ntime
        ntot = fill(ComplexF64(0.8), nt + 1)
        gv = fill(ComplexF64(0.5), nt + 1)
        X = zeros(ComplexF64, nt + 1)
        # The direct formula is used at n = 0 and n > k (NESSi tstp ≥ k),
        # the start-up version at n = 1, ⋯, k + 1
        steps = [0; P5H.k+1:nt]
        for n in steps
            phonon_displacement!(n, X, ntot, gv, ph)
        end
        P5Holstein.phonon_displacement_start!(X, ntot, gv, ph)
        @test maximum(abs.(X .+ 2 * 0.5 * 0.8 / P5H.w0)) < 1e-14
        # Linear density: direct quadrature of D0^R(t,t') [g n(t') - g n(0)]
        ntot = ComplexF64[0.8 + 0.1 * max(n - 1, 0) * P5H.h for n = 0:nt]
        for n in steps
            phonon_displacement!(n, X, ntot, gv, ph)
        end
        for n = P5H.k+1:nt
            f = [ph.D0.ret[n, j+1][1,1] * (gv[j+2] * ntot[j+2] - gv[1] * ntot[1]) for j = 0:n-1]
            s = sum(ph.I.GIW[n-1, j] * f[j+1] for j = 0:n-1)
            @test abs(X[n+1] - (s * P5H.h - 2 * 0.5 * 0.8 / P5H.w0)) < 1e-12
        end
    end

    @testset "bare phonon energy" begin
        ph = p5h_phonon(1)
        Pi = ℱ(p5h_cn(1), BOSE)
        E = phonon_energy_quantum(ph.D0, Pi, ph)
        @test maximum(abs.(E .- P5H.w0 * (bose(P5H.β, P5H.w0) + 0.5))) < 1e-10
    end

    @testset "Nambu self-energies in the normal limit" begin
        G1 = p5h_green(0.3, 1)
        G2 = p5h_green(0.3, 2)
        g1 = holstein_scalar_cf(P5H.ntime, fill(0.4, P5H.ntime + 1))
        g2 = Cf(P5H.ntime, 2, 2, zero(ComplexF64))
        for n = 0:P5H.ntime
            g2[n] = ComplexF64[0.4 0; 0 -0.4]
        end
        for (fn!, fs!, fns!, fss!) in ((sigma_umig!, sigma_umig_sc!, sigma_umig_start!, sigma_umig_sc_start!),
                                        (sigma_mig!, sigma_mig_sc!, sigma_mig_start!, sigma_mig_sc_start!))
            Σ1 = ℱ(p5h_cn(1), FERMI)
            Σ2 = ℱ(p5h_cn(2), FERMI)
            ph1, ph2 = p5h_phonon(1), p5h_phonon(2)
            fn!(0, G1, Σ1, g1, ph1)
            fs!(0, G2, Σ2, g2, ph2)
            fns!(G1, Σ1, g1, ph1)
            fss!(G2, Σ2, g2, ph2)
            err = 0.0
            for n = 0:P5H.k+1, a = 1:2
                if n == 0
                    for m = 1:P5H.ntau
                        err = max(err, abs(Σ2.mat[m][a,a] - Σ1.mat[m][1,1]))
                    end
                else
                    for j = 1:n
                        err = max(err, abs(Σ2.ret[n, j][a,a] - Σ1.ret[n, j][1,1]))
                        err = max(err, abs(Σ2.less[j, n][a,a] - Σ1.less[j, n][1,1]))
                    end
                    for m = 1:P5H.ntau
                        err = max(err, abs(Σ2.lmix[n, m][a,a] - Σ1.lmix[n, m][1,1]))
                    end
                end
            end
            @test err < 1e-12
            # No anomalous self-energy without an anomalous Green's function
            @test maximum(abs(Σ2.mat[m][1,2]) for m = 1:P5H.ntau) < 1e-14
        end
    end
end
