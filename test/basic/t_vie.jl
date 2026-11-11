#
# t_vie.jl
#
# To test the VIE2 solvers, vie2_mat!(), vie2_start!(), vie2_timestep!(),
# vie2!(), and vie2_timestep_sin!().
#
# Model: the 2 × 2 block of the free 4 × 4 Hamiltonian [Ha V; V' Hb]
# solves G = G0 + G0 * Σ * G, where G0 is the free Green's function of Ha
# and Σ = V gb V' is the embedding self-energy. Hence the VIE2
# [1 + F] * G = Q with F = -G0 * Σ, Fcc = -Σ * G0, and Q = G0 is solved.
# The blocks do not commute and the couplings are complex.
#
# The point-wise comparison with NESSi is in test/nessi/t_phase4.jl.
#

isdefined(Main, :kb_test_profile) ||
    include(joinpath(@__DIR__, "..", "common", "profile.jl"))
isdefined(Main, :test_order_gate) ||
    include(joinpath(@__DIR__, "..", "common", "order_gate.jl"))

using Test
using LinearAlgebra
using KadanoffBaym

const VT_HA = ComplexF64[-1.0 0.3im; -0.3im 0.5]
const VT_HB = ComplexF64[1.0 0.2+0.1im; 0.2-0.1im 1.5]
const VT_V = ComplexF64[0.4 0.1im; -0.2 0.3]

#
# Exact solution G, kernel F, partner Fcc, and source Q. The kernels are
# evaluated by the convolution of order ks.
#
function vt_setup(Nt, ks; ntau = 201, β = 1.0, Tmax = 2.0, sign = FERMI)
    μ = sign == FERMI ? 0.1 : -2.5
    h = Tmax / Nt
    nt = Nt + 1
    C4 = Cn(nt, ntau, 4, 4, Tmax, β)
    C2 = Cn(nt, ntau, 2, 2, Tmax, β)
    Gf = ℱ(C4, sign)
    from_ham!(Gf, μ, [VT_HA VT_V; VT_V' VT_HB], β, h)
    Gx = ℱ(C2, sign)
    for m = 1:ntau
        Gx.mat[m] = Gf.mat[m][1:2, 1:2]
    end
    for i = 1:nt, m = 1:ntau
        Gx.lmix[i, m] = Gf.lmix[i, m][1:2, 1:2]
    end
    for i = 1:nt, j = 1:i
        Gx.ret[i, j] = Gf.ret[i, j][1:2, 1:2]
        Gx.less[j, i] = Gf.less[j, i][1:2, 1:2]
    end
    gb = ℱ(C2, sign)
    from_ham!(gb, μ, VT_HB, β, h)
    Σ = ℱ(C2, sign)
    for m = 1:ntau
        Σ.mat[m] = VT_V * gb.mat[m] * VT_V'
    end
    for i = 1:nt, m = 1:ntau
        Σ.lmix[i, m] = VT_V * gb.lmix[i, m] * VT_V'
    end
    for i = 1:nt, j = 1:i
        Σ.ret[i, j] = VT_V * gb.ret[i, j] * VT_V'
        Σ.less[j, i] = VT_V * gb.less[j, i] * VT_V'
    end
    Q = ℱ(C2, sign)
    from_ham!(Q, μ, VT_HA, β, h)
    F = ℱ(C2, sign)
    Fcc = ℱ(C2, sign)
    convolution(F, Q, Σ, ks, β, h)
    convolution(Fcc, Σ, Q, ks, β, h)
    for t = 0:nt
        smul!(F, t, -1.0)
        smul!(Fcc, t, -1.0)
    end
    return (G = Gx, F = F, Fcc = Fcc, Q = Q, C = C2, β = β, h = h, nt = nt, sign = sign)
end

#
# Errors of the ret, tv, and les components per element. The Matsubara
# component is taken from the exact solution, such that only the real
# time discretization is measured.
#
function vt_errors(ks, Nt; kw...)
    S = vt_setup(Nt, ks; kw...)
    G = ℱ(S.C, S.sign)
    G.mat.data .= S.G.mat.data
    I = Integrator(ks)
    vie2_start!(G, S.F, S.Fcc, S.Q, I, S.β, S.h)
    for tstp = ks+2:S.nt
        vie2_timestep!(tstp, G, S.F, S.Fcc, S.Q, I, S.β, S.h)
    end
    ntau = getntau(G)
    return (maximum(distance(G, S.G, n, Val(:ret)) / n for n = 1:S.nt),
            maximum(distance(G, S.G, n, Val(:tv)) / ntau for n = 1:S.nt),
            maximum(distance(G, S.G, n, Val(:les)) / n for n = 1:S.nt))
end

#
# Reference implementation of vie2_timestep_sin!, a literal port of NESSi
# vie2_timestep_sin (cntr_vie2_impl.hpp:1778-1844): full copies of F, Fcc,
# and Q are transformed at the time steps 0, ⋯, n₂ in every call.
#
function vt_sin_reference!(tstp, G, Gsin, F, Fcc, Fsin, Q, Qsin, I, β, h)
    k = I.k
    n₂ = (1 ≤ tstp ≤ k + 1) ? k + 1 : tstp
    Id = Matrix{ComplexF64}(LinearAlgebra.I, 2, 2)
    Finv = Cf(n₂, 2, 2)
    for i = 0:n₂
        Dinv = inv(Id + Fsin[i])
        Finv[i] = Dinv
        Gsin[i] = Dinv * Qsin[i]
    end
    tF = deepcopy(F)
    tFcc = deepcopy(Fcc)
    tQ = deepcopy(Q)
    for i = 0:n₂
        smul!(Finv, tF, i)
        smul!(tFcc, Finv, i)
        smul!(Finv, tQ, i)
    end
    tF₁ = deepcopy(tF)
    for i = 0:n₂
        smul!(tF₁, Gsin, i)
        incr!(tQ, tF₁, i, -one(ComplexF64))
    end
    vie2_timestep!(tstp, G, tF, tFcc, tQ, I, β, h)
    return G
end

#
# GW-like calling pattern: repeated calls at tstp = 0, repeated start-up
# with modified inputs at tstp = 1, ⋯, k + 1, and repeated calls at each
# later time step with modified inputs at that time step only. `call` has
# the arguments of vt_sin_reference!.
#
function vt_sin_pattern(call, k, sign; Nt = 12)
    S = vt_setup(Nt, k; sign = sign)
    nt = S.nt
    F, Fcc, Q = deepcopy(S.F), deepcopy(S.Fcc), deepcopy(S.Q)
    Fs, Qs, Gs = Cf(nt, 2, 2), Cf(nt, 2, 2), Cf(nt, 2, 2)
    for i = 0:nt
        Fs[i] = ComplexF64[0.3+0.01i 0.1im; -0.1im -0.2]
        Qs[i] = ComplexF64[0.5 0.2+0.01im*i; -0.3 -0.4]
    end
    G = ℱ(S.C, sign)
    I = Integrator(k)
    Id = Matrix{ComplexF64}(LinearAlgebra.I, 2, 2)
    for it = 1:2
        foreach(X -> smul!(X, 0, 1.0 + 0.01 * it), (F, Fcc, Q))
        call(0, G, Gs, F, Fcc, Fs, Q, Qs, I, S.β, S.h)
    end
    for it = 1:2
        for t = 1:k+1
            foreach(X -> smul!(X, t, 1.0 + 0.01 * it), (F, Fcc, Q))
        end
        Fs[k+1] = Fs[k+1] + 0.001 * it * Id
        for tstp = 1:k+1
            call(tstp, G, Gs, F, Fcc, Fs, Q, Qs, I, S.β, S.h)
        end
    end
    for tstp = k+2:nt, it = 1:2
        foreach(X -> smul!(X, tstp, 1.0 + 0.01 * it), (F, Fcc, Q))
        Fs[tstp] = Fs[tstp] + 0.001 * it * Id
        call(tstp, G, Gs, F, Fcc, Fs, Q, Qs, I, S.β, S.h)
    end
    return G, Gs
end

vt_same(G1, G2) = G1.mat.data == G2.mat.data && G1.ret.data == G2.ret.data &&
                  G1.lmix.data == G2.lmix.data && G1.less.data == G2.less.data

@testset "VIE2 ($(kb_test_profile()))" begin
    @testset "Matsubara: fixpoint and cg" begin
        for sign in (FERMI, BOSE)
            S = vt_setup(10, 5; sign = sign)
            I = Integrator(5)
            ntau = getntau(S.G)
            # FIXPOINT
            G1 = ℱ(S.C, sign)
            vie2_mat!(G1, S.F, S.Fcc, S.Q, S.β, I)
            # CG
            G2 = ℱ(S.C, sign)
            vie2_mat!(G2, S.F, S.Fcc, S.Q, S.β, I; method = :cg)
            @test distance(G1, S.G, 0) / ntau < 1e-12
            @test distance(G2, S.G, 0) / ntau < 1e-12
            # Residual of the discrete equation G + F * G - Q
            for G in (G1, G2)
                R = KadanoffBaym.Gᵐᵃᵗ(ntau, 2, 2)
                KadanoffBaym.conv_mat(R, S.F.mat, G.mat, I, S.β, sign)
                R.data .= R.data .+ G.mat.data .- S.Q.mat.data
                @test maximum(abs, R.data) < 1e-10
            end
            # Only the Matsubara component is written
            @test all(iszero, G1.ret.data) && all(iszero, G1.less.data)
        end
        # The inner product of CG is the sum of all the elements of
        # X' Y, not the trace
        X = KadanoffBaym.Gᵐᵃᵗ(11, 2, 2)
        for m = 1:11
            KadanoffBaym._block(X, m) .= ComplexF64[1 2im; 0.5 -1]
        end
        I = Integrator(2)
        w = [I.GIW.ω[1:3]; ones(5); reverse(I.GIW.ω[1:3])]
        P = ComplexF64[1 2im; 0.5 -1]' * ComplexF64[1 2im; 0.5 -1]
        @test KadanoffBaym._mat_inner(X, X, I) ≈ sum(w) * sum(real, P)
        @test !(KadanoffBaym._mat_inner(X, X, I) ≈ sum(w) * real(tr(P)))
    end

    @testset "Matsubara: dispatch and errors" begin
        S = vt_setup(6, 2)
        I = Integrator(2)
        G = ℱ(S.C, FERMI)
        # :fourier is implemented in Phase 7, see t_matsubara_solvers_fourier.jl
        @test all(isfinite, vie2_mat!(G, S.F, S.Fcc, S.Q, S.β, I; method = :fourier).mat.data)
        @test_throws ArgumentError vie2_mat!(G, S.F, S.Fcc, S.Q, S.β, I; method = :steep)
        @test_throws ArgumentError vie2_mat!(G, S.F, S.Fcc, S.Q, -1.0, I)
        @test_throws ArgumentError vie2_mat!(G, S.F, S.Fcc, S.Q, S.β, Integrator(6))
        @test_throws ArgumentError vie2_mat!(S.F, S.F, S.Fcc, S.Q, S.β, I)
        @test_throws ArgumentError vie2_mat!(ℱ(S.C, BOSE), S.F, S.Fcc, S.Q, S.β, I)
        @test_throws DimensionMismatch vie2_mat!(G, ℱ(Cn(7, 11, 2, 2, 1.0, 1.0), FERMI),
                                                 S.Fcc, S.Q, S.β, I)
    end

    @testset "real time: routing and drivers (k = $k)" for k in (1, 3, 5)
        for sign in (FERMI, BOSE)
            S = vt_setup(12, k; sign = sign)
            I = Integrator(k)
            # Full driver
            G1 = ℱ(S.C, sign)
            vie2!(G1, S.F, S.Fcc, S.Q, I, S.β, S.h)
            # Step by step, including the reduced orders at the start
            G2 = ℱ(S.C, sign)
            for tstp = 0:S.nt
                vie2_timestep!(tstp, G2, S.F, S.Fcc, S.Q, I, S.β, S.h)
                if tstp == 1
                    @test G2.ret[1, 1] == S.Q.ret[1, 1]
                elseif 2 ≤ tstp ≤ k + 1
                    G3 = deepcopy(G2)
                    vie2_start!(G3, S.F, S.Fcc, S.Q, Integrator(tstp - 1), S.β, S.h)
                    @test all(n -> distance(G2, G3, n) == 0.0, 0:tstp)
                end
            end
            @test all(n -> distance(G1, G2, n) == 0.0, 0:S.nt)
            # Accuracy, the sum of the block norms at each time step
            @test maximum(distance(G1, S.G, n) for n = 0:S.nt) < (k == 1 ? 1e-1 : 2e-2)
            # Both F and Fcc are used: exchanging them changes the result
            G4 = ℱ(S.C, sign)
            vie2!(G4, S.Fcc, S.F, S.Q, I, S.β, S.h)
            @test maximum(distance(G1, G4, n) for n = 1:S.nt) > 1e-6
        end
    end

    @testset "real time: sentinels of the time stepping" begin
        k = 3
        S = vt_setup(12, k)
        I = Integrator(k)
        G = ℱ(S.C, FERMI)
        for tstp = 0:S.nt-1
            vie2_timestep!(tstp, G, S.F, S.Fcc, S.Q, I, S.β, S.h)
        end
        tstp = S.nt
        Gref = deepcopy(G)
        vie2_timestep!(tstp, Gref, S.F, S.Fcc, S.Q, I, S.β, S.h)
        ntau = getntau(G)
        # The current left-mixing row is cleared before the convolution
        for val in (1e6, NaN)
            X = deepcopy(G)
            KadanoffBaym._vie2_timestep_ret!(tstp, X, S.Fcc, S.F, S.Q, S.h, I)
            for m = 1:ntau
                X.lmix[tstp, m] = fill(ComplexF64(val), 2, 2)
            end
            KadanoffBaym._vie2_timestep_lmix!(tstp, X, S.F, S.Fcc, S.Q, S.β, S.h, I)
            @test all(m -> X.lmix[tstp, m] == Gref.lmix[tstp, m], 1:ntau)
        end
        # The current lesser column does not enter the source term, i.e.,
        # the term F^R * G^< is not part of the explicit convolution
        for val in (1e6, NaN)
            X = deepcopy(G)
            KadanoffBaym._vie2_timestep_ret!(tstp, X, S.Fcc, S.F, S.Q, S.h, I)
            KadanoffBaym._vie2_timestep_lmix!(tstp, X, S.F, S.Fcc, S.Q, S.β, S.h, I)
            for j = 1:tstp
                X.less[j, tstp] = fill(ComplexF64(val), 2, 2)
            end
            KadanoffBaym._vie2_timestep_less!(tstp, X, S.F, S.Fcc, S.Q, S.β, S.h, I)
            @test all(j -> X.less[j, tstp] == Gref.less[j, tstp], 1:tstp)
        end
    end

    @testset "real time: errors" begin
        S = vt_setup(8, 2)
        I = Integrator(2)
        G = ℱ(S.C, FERMI)
        @test_throws BoundsError vie2_timestep!(S.nt + 1, G, S.F, S.Fcc, S.Q, I, S.β, S.h)
        @test_throws BoundsError vie2_timestep!(-1, G, S.F, S.Fcc, S.Q, I, S.β, S.h)
        @test_throws ArgumentError vie2_timestep!(1, G, S.F, S.Fcc, S.Q, I, S.β, 0.0)
        @test_throws ArgumentError vie2_start!(G, S.F, S.Fcc, S.Q, I, S.β, -0.1)
        @test_throws ArgumentError vie2_start!(G, S.F, S.Fcc, S.Q, Integrator(6), S.β, S.h)
        S2 = vt_setup(2, 2)
        @test_throws ArgumentError vie2_start!(ℱ(S2.C, FERMI), S2.F, S2.Fcc, S2.Q,
                                               Integrator(3), S2.β, S2.h)
    end

    @testset "singular kernel" begin
        k = 3
        S = vt_setup(12, k)
        I = Integrator(k)
        nt = S.nt
        Id = Matrix{ComplexF64}(LinearAlgebra.I, 2, 2)
        # Zero singular parts: the ordinary VIE2
        Gs = Cf(nt, 2, 2)
        Fs = Cf(nt, 2, 2)
        Qs = Cf(nt, 2, 2)
        for i = 0:nt
            Fs[i] = zeros(ComplexF64, 2, 2)
            Qs[i] = zeros(ComplexF64, 2, 2)
        end
        G1 = ℱ(S.C, FERMI)
        G2 = ℱ(S.C, FERMI)
        for tstp = 0:nt
            vie2_timestep!(tstp, G1, S.F, S.Fcc, S.Q, I, S.β, S.h)
            vie2_timestep_sin!(tstp, G2, Gs, S.F, S.Fcc, Fs, S.Q, Qs, I, S.β, S.h)
        end
        @test all(n -> distance(G1, G2, n) == 0.0, 0:nt)
        # Constructed problem: D = 1 + Fsin, F' = D F, Fcc' = Fcc D,
        # Q' = D Q + F' Gsin. The regular part is the solution G1 of the
        # ordinary VIE2 and the singular part is D⁻¹ Qsin.
        Dm = ComplexF64[1.3 0.1im; -0.1im 0.8]
        Dc = Cf(nt, 2, 2)
        Gsx = Cf(nt, 2, 2)
        for i = 0:nt
            Fs[i] = Dm - Id
            Qs[i] = ComplexF64[0.5 0.2; 0.2 -0.4] * (1 + 0.1 * i)
            Dc[i] = Dm
            Gsx[i] = Dm \ Qs[i]
        end
        Fd = deepcopy(S.F)
        Fccd = deepcopy(S.Fcc)
        Qd = deepcopy(S.Q)
        for t = 0:nt
            smul!(Dc, Fd, t)
            smul!(Fccd, Dc, t)
            smul!(Dc, Qd, t)
        end
        FG = deepcopy(Fd)
        for t = 0:nt
            smul!(FG, Gsx, t)
            incr!(Qd, FG, t, 1.0)
        end
        G3 = ℱ(S.C, FERMI)
        for tstp = 0:nt
            vie2_timestep_sin!(tstp, G3, Gs, Fd, Fccd, Fs, Qd, Qs, I, S.β, S.h)
        end
        @test maximum(distance(G1, G3, n) for n = 0:nt) < 1e-12
        @test maximum(norm(Gs[i] - Gsx[i]) for i = 0:nt) < 1e-13
        # Errors: singular D and non-hermitian Fsin
        for i = 0:nt
            Fs[i] = -Id
        end
        @test_throws SingularException vie2_timestep_sin!(1, G3, Gs, Fd, Fccd, Fs, Qd, Qs,
                                                           I, S.β, S.h)
        for i = 0:nt
            Fs[i] = ComplexF64[0.1 0.2; 0.0 0.1]
        end
        @test_throws ArgumentError vie2_timestep_sin!(1, G3, Gs, Fd, Fccd, Fs, Qd, Qs,
                                                      I, S.β, S.h)
    end

    @testset "singular kernel: workspace" begin
        cases = kb_release() ? [(s, k) for s in (FERMI, BOSE) for k = 1:5] :
                               [(FERMI, 1), (BOSE, 3), (FERMI, 5)]
        for (sign, k) in cases
            Gr, Gsr = vt_sin_pattern(vt_sin_reference!, k, sign)
            # Without workspace
            G1, Gs1 = vt_sin_pattern(k, sign) do args...
                vie2_timestep_sin!(args...)
            end
            @test vt_same(G1, Gr)
            @test all(i -> Gs1[i] == Gsr[i], 0:getntime(Gr))
            # One workspace for all the calls
            ws = Ref{Any}(nothing)
            G2, Gs2 = vt_sin_pattern(k, sign) do tstp, G, rest...
                isnothing(ws[]) && (ws[] = VIE2SinWorkspace(G))
                vie2_timestep_sin!(tstp, G, rest...; ws = ws[])
            end
            @test vt_same(G2, Gr)
            @test all(i -> Gs2[i] == Gsr[i], 0:getntime(Gr))
        end

        # Other input objects reset the workspace; going back to an earlier
        # time step invalidates the later ones
        k = 2
        S = vt_setup(10, k)
        nt = S.nt
        I = Integrator(k)
        Fs, Qs = Cf(nt, 2, 2), Cf(nt, 2, 2)
        for i = 0:nt
            Fs[i] = ComplexF64[0.2 0.05im; -0.05im -0.1] * (1 + 0.02i)
            Qs[i] = ComplexF64[0.4 0.1; 0.1 -0.3]
        end
        F2 = deepcopy(S.F)
        foreach(t -> smul!(F2, t, 0.9), 0:nt)
        Gr = ℱ(S.C, FERMI); Gsr = Cf(nt, 2, 2)
        Gw = ℱ(S.C, FERMI); Gsw = Cf(nt, 2, 2)
        ws = VIE2SinWorkspace(Gw)
        for tstp = 0:nt
            vt_sin_reference!(tstp, Gr, Gsr, S.F, S.Fcc, Fs, S.Q, Qs, I, S.β, S.h)
            vie2_timestep_sin!(tstp, Gw, Gsw, S.F, S.Fcc, Fs, S.Q, Qs, I, S.β, S.h; ws = ws)
        end
        @test vt_same(Gw, Gr) && ws.ndone == nt
        # Another kernel object at the last time step
        vt_sin_reference!(nt, Gr, Gsr, F2, S.Fcc, Fs, S.Q, Qs, I, S.β, S.h)
        vie2_timestep_sin!(nt, Gw, Gsw, F2, S.Fcc, Fs, S.Q, Qs, I, S.β, S.h; ws = ws)
        @test vt_same(Gw, Gr)
        # Back to an earlier time step with modified inputs, then forward
        t₀ = k + 3
        for t = t₀:nt
            Fs[t] = Fs[t] + 0.01 * Matrix{ComplexF64}(LinearAlgebra.I, 2, 2)
            smul!(F2, t, 1.1)
        end
        for tstp = t₀:nt
            vt_sin_reference!(tstp, Gr, Gsr, F2, S.Fcc, Fs, S.Q, Qs, I, S.β, S.h)
            vie2_timestep_sin!(tstp, Gw, Gsw, F2, S.Fcc, Fs, S.Q, Qs, I, S.β, S.h; ws = ws)
        end
        @test vt_same(Gw, Gr)
        @test all(i -> Gsw[i] == Gsr[i], 0:nt)
        # Another Gsin object at a later time step: the earlier values of
        # the new object are not known, so the workspace must be reset
        Gsn = Cf(nt, 2, 2)
        Gsr2 = Cf(nt, 2, 2)
        vt_sin_reference!(nt, Gr, Gsr2, F2, S.Fcc, Fs, S.Q, Qs, I, S.β, S.h)
        vie2_timestep_sin!(nt, Gw, Gsn, F2, S.Fcc, Fs, S.Q, Qs, I, S.β, S.h; ws = ws)
        @test vt_same(Gw, Gr)
        @test all(i -> Gsn[i] == Gsr2[i], 0:nt)
        # Incompatible workspace
        S2 = vt_setup(6, k)
        @test_throws DimensionMismatch vie2_timestep_sin!(1, Gw, Gsw, S.F, S.Fcc, Fs, S.Q, Qs,
                                                          I, S.β, S.h;
                                                          ws = VIE2SinWorkspace(ℱ(S2.C, FERMI)))
        @test_throws DimensionMismatch vie2_timestep_sin!(1, Gw, Gsw, S.F, S.Fcc, Fs, S.Q, Qs,
                                                          I, S.β, S.h;
                                                          ws = VIE2SinWorkspace(ℱ(S.C, BOSE)))
    end

    @testset "convergence order (k = $k)" for k = 1:5
        # The observed order is k + 2
        Ns, caps = kb_release() ?
            ([20, 40, 80, 160], (5e-7, 1e-8, 2e-10, 1e-11, 1e-12)) :
            ([8, 16, 32, 64], (5e-6, 2e-7, 1e-8, 1e-9, 1e-10))
        E = [vt_errors(k, N) for N in Ns]
        for c = 1:3
            test_order_gate(Ns, [e[c] for e in E], k + 2, caps[k]; order_tol = 0.35)
        end
        # Reduced-order mutant: the data of order 1 must fail the gate of
        # order 2
        if k == 2
            Em = [vt_errors(1, N) for N in Ns]
            for c = 1:3
                @test !check_order_gate(Ns, [e[c] for e in Em], k + 2, caps[k];
                                        order_tol = 0.35)[1]
            end
        end
    end

    if kb_release()
        @testset "release: allocation of one time step" begin
            @test kb_test_profile() == "release"
            S = vt_setup(64, 5; ntau = 129, Tmax = 3.2)
            I = Integrator(5)
            G = ℱ(S.C, FERMI)
            vie2!(G, S.F, S.Fcc, S.Q, I, S.β, S.h)
            ids = [objectid(getfield(X, c).data) for X in (G, S.F, S.Fcc, S.Q)
                   for c in (:mat, :ret, :lmix, :less)]
            vie2_timestep!(S.nt - 1, G, S.F, S.Fcc, S.Q, I, S.β, S.h)
            bytes = @allocated vie2_timestep!(S.nt, G, S.F, S.Fcc, S.Q, I, S.β, S.h)
            total = sum(sizeof(getfield(X, c).data) for X in (G, S.F, S.Fcc, S.Q)
                        for c in (:mat, :ret, :lmix, :less))
            println("    vie2_timestep! allocation: $bytes bytes, G/F/Fcc/Q storage: $total bytes")
            @test bytes ≤ 256 * 1024
            @test ids == [objectid(getfield(X, c).data) for X in (G, S.F, S.Fcc, S.Q)
                          for c in (:mat, :ret, :lmix, :less)]
            #
            # vie2_timestep_sin! with a workspace: no copy of a full contour
            # function, the extra allocation is that of one time slice
            nt = S.nt
            Fs, Qs, Gs = Cf(nt, 2, 2), Cf(nt, 2, 2), Cf(nt, 2, 2)
            for i = 0:nt
                Fs[i] = ComplexF64[0.3 0.1im; -0.1im -0.2]
                Qs[i] = ComplexF64[0.5 0.2; 0.2 -0.4]
            end
            Gx = ℱ(S.C, FERMI)
            ws = VIE2SinWorkspace(Gx)
            for tstp = 0:nt-1
                vie2_timestep_sin!(tstp, Gx, Gs, S.F, S.Fcc, Fs, S.Q, Qs, I, S.β, S.h; ws = ws)
            end
            bsin = @allocated vie2_timestep_sin!(nt, Gx, Gs, S.F, S.Fcc, Fs, S.Q, Qs,
                                                 I, S.β, S.h; ws = ws)
            oneℱ = sum(sizeof(getfield(S.F, c).data) for c in (:mat, :ret, :lmix, :less))
            println("    vie2_timestep_sin! allocation: $bsin bytes, one ℱ: $oneℱ bytes")
            @test bsin ≤ 2 * 256 * 1024
            @test bsin < oneℱ
        end
    end
end
