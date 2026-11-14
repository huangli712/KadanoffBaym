#
# t_equilibrium.jl
#
# To test the equilibrium Green's functions and the distribution
# functions.
#

using Test
using LinearAlgebra
using KadanoffBaym

@testset "equilibrium: fermi_exp and bose_exp" begin
    β = 5.0
    τs = [0.0, 0.3, 2.5, 5.0]
    #
    # Moderate arguments: compare with the naive formulas
    for ω in (-3.0, -0.7, 0.2, 1.9), τ in τs
        @test fermi_exp(β, τ, ω) ≈ exp(τ * ω) / (1.0 + exp(β * ω)) rtol = 1e-14
        @test bose_exp(β, τ, ω) ≈ exp(τ * ω) / (exp(β * ω) - 1.0) rtol = 1e-14
    end
    #
    # Wrappers of the extended distribution functions
    for ω in (-2.0, 0.5), τ in τs
        @test fermi_exp(β, τ, ω) == fermi(β, τ, ω)
        @test bose_exp(β, τ, ω) == bose(β, τ, ω)
    end
    #
    # Vector versions
    ωs = [-1.0, 0.3, 2.0]
    @test fermi_exp(β, 1.0, ωs) == [fermi_exp(β, 1.0, x) for x in ωs]
    @test bose_exp(β, 1.0, ωs) == [bose_exp(β, 1.0, x) for x in ωs]
    #
    # Extreme arguments, β ω = ± 1000, no overflow
    for ω in (-200.0, 200.0), τ in τs
        @test isfinite(fermi_exp(β, τ, ω))
        @test isfinite(bose_exp(β, τ, ω))
    end
    @test fermi_exp(β, 0.0, 200.0) == 0.0
    @test fermi_exp(β, β, -200.0) == 0.0
    @test fermi_exp(β, 0.0, -200.0) == 1.0
    @test fermi_exp(β, β, 200.0) == 1.0
    @test bose_exp(β, 0.0, -200.0) == -1.0
    @test bose_exp(β, β, 200.0) == 1.0
    #
    # Zero mode of the Bose function
    @test_throws DomainError bose_exp(β, 1.0, 0.0)
    @test_throws DomainError bose_exp(β, 1.0, [1.0, 0.0])
end

#
# Exact solutions ported from libcntr/test/equilibrium.cpp
#

# Propagator of H(t) = [Δ V e^{-2iωt}; V e^{2iωt} -Δ]
function eq_U(t, Δ, V, ω)
    Om = sqrt((Δ - ω)^2 + V^2)
    s, c = sin(Om * t), cos(Om * t)
    return ComplexF64[exp(-im*ω*t)*(c - im*(Δ-ω)*s/Om) -im*V*exp(-im*ω*t)*s/Om;
                      -im*V*exp(im*ω*t)*s/Om exp(im*ω*t)*(c + im*(Δ-ω)*s/Om)]
end

function eq_ham_time(nt, h, Δ, V, ω)
    H = Cf(nt, 2, 2)
    H[0] = ComplexF64[Δ V; V -Δ]
    for n = 1:nt
        t = (n - 1) * h
        H[n] = ComplexF64[Δ V*exp(-2im*ω*t); V*exp(2im*ω*t) -Δ]
    end
    return H
end

# exact_timeindependent(), H = √2 [1 -i; i -1], fermions
function eq_exact_timeindependent(C::Cn, β, dt)
    nt, ntau = getntime(C), getntau(C)
    δτ = β / (ntau - 1)
    G = ℱ(C, FERMI)
    for m = 1:ntau
        τ = (m - 1) * δτ
        mat = zeros(ComplexF64, 2, 2)
        mat[1,1] = exp(-2τ) * (-0.8535533905932737 * exp(2β) - 0.1464466094067262 * exp(4τ))
        mat[2,2] = exp(-2τ) * (-0.1464466094067262 * exp(2β) - 0.8535533905932737 * exp(4τ))
        mat[2,1] = -im * 0.35355339059327373 * (exp(2(β - τ)) - exp(2τ))
        mat[1,2] = im * 0.35355339059327373 * (exp(2(β - τ)) - exp(2τ))
        G.mat[m] = mat / (1.0 + exp(2β))
        for n = 1:nt
            t1 = (n - 1) * dt
            tv = zeros(ComplexF64, 2, 2)
            tv[1,1] = im*0.8535533905932737*exp(-2im*t1 + 2τ) + im*0.1464466094067262*exp(2im*t1 - 2τ + 2β)
            tv[1,2] = -0.35355339059327373 * (exp(2im*t1 + 2β - 2τ) - exp(-2im*t1 + 2τ))
            tv[2,1] = 0.35355339059327373 * (exp(2im*t1 + 2β - 2τ) - exp(-2im*t1 + 2τ))
            tv[2,2] = im*0.8535533905932737*exp(2im*t1 + 2β - 2τ) + 0.1464466094067262*im*exp(-2im*t1 + 2τ)
            G.lmix[n, m] = tv / (1.0 + exp(2β))
        end
    end
    for m = 1:nt, n = 1:m
        t1, t2 = (m - 1) * dt, (n - 1) * dt
        x = 2.0 * (t1 - t2)
        G.ret[m, n] = ComplexF64[-im*cos(x)-0.7071067811865475*sin(x) 0.7071067811865475im*sin(x);
                                 -0.7071067811865475im*sin(x) -im*cos(x)+0.7071067811865475*sin(x)]
        t1, t2 = (n - 1) * dt, (m - 1) * dt
        les = zeros(ComplexF64, 2, 2)
        les[1,1] = 0.8535533905932737im*exp(2im*(t2-t1)) + 0.1464466094067262im*exp(2im*(t1-t2) + 2β)
        les[2,1] = -0.35355339059327373*exp(2im*(t2-t1)) + 0.35355339059327373*exp(2im*(t1-t2) + 2β)
        les[1,2] = -0.35355339059327373*exp(2im*(t1-t2) + 2β) + 0.35355339059327373*exp(-2im*(t1-t2))
        les[2,2] = 0.8535533905932737im*exp(2im*(t1-t2) + 2β) + 0.1464466094067262im*exp(2im*(t2-t1))
        G.less[n, m] = les / (1.0 + exp(2β))
    end
    return G
end

# exact_time() and exact_time_boson(), Δ = 1, V = 0.5
function eq_exact_time(C::Cn, β, dt, Δ, V, ω, sign)
    nt, ntau = getntime(C), getntau(C)
    δτ = β / (ntau - 1)
    λ = 1.118033988749895
    vec = [0.22975292054736116 -0.9732489894677301; -0.9732489894677301 -0.22975292054736116]
    G = ℱ(C, sign)
    if sign == FERMI
        nd = Diagonal([1.0 / (1.0 + exp(-λ * β)), 1.0 / (1.0 + exp(λ * β))])
        pref = im
    else
        nd = Diagonal([1.0 / (-1.0 + exp(-λ * β)), 1.0 / (-1.0 + exp(λ * β))])
        pref = -im
    end
    for m = 1:ntau
        τ = (m - 1) * δτ
        mat = zeros(ComplexF64, 2, 2)
        if sign == FERMI
            mat[1,1] = exp(-λ*τ) * (-0.9472135954999577*exp(λ*β) - 0.05278640450004204*exp(2λ*τ))
            mat[1,2] = exp(-λ*τ) * (-0.22360679774997894*exp(λ*β) + 0.22360679774997894*exp(2λ*τ))
            mat[2,1] = mat[1,2]
            mat[2,2] = exp(-λ*τ) * (-0.052786404500042045*exp(λ*β) - 0.9472135954999577*exp(2λ*τ))
            G.mat[m] = mat / (1.0 + exp(λ * β))
        else
            mat[1,1] = exp(-λ*τ) * (-0.9472135954999577*exp(λ*β) + 0.05278640450004204*exp(2λ*τ))
            mat[1,2] = exp(-λ*τ) * (-0.22360679774997894*exp(λ*β) - 0.22360679774997894*exp(2λ*τ))
            mat[2,1] = mat[1,2]
            mat[2,2] = exp(-λ*τ) * (-0.052786404500042045*exp(λ*β) + 0.9472135954999577*exp(2λ*τ))
            G.mat[m] = mat / (-1.0 + exp(λ * β))
        end
        expon = Diagonal([exp(-λ * τ), exp(λ * τ)])
        for n = 1:nt
            G.lmix[n, m] = pref * eq_U((n - 1) * dt, Δ, V, ω) * transpose(vec) * nd * expon * conj(vec)
        end
    end
    for m = 1:nt, n = 1:m
        U1 = eq_U((m - 1) * dt, Δ, V, ω)
        U2 = eq_U((n - 1) * dt, Δ, V, ω)
        G.ret[m, n] = -im * U1 * U2'
        G.less[n, m] = pref * U2 * transpose(vec) * nd * conj(vec) * U1'
    end
    return G
end

@testset "equilibrium: NESSi Catch (equilibrium.cpp)" begin
    nt, ntau, β, h, μ, kt = 100, 500, 5.0, 0.01, 0.0, 5
    C = Cn(nt + 1, ntau + 1, 2, 2, nt * h, β)
    #
    # Time independent
    a = sqrt(2.0) * ComplexF64[1.0 -im; im -1.0]
    exact = eq_exact_timeindependent(C, β, h)
    A = ℱ(C, FERMI)
    from_ham!(A, μ, a, β, h)
    @test sum(distance(A, exact, tstp) for tstp = 0:nt) < 1e-6
    ϵ = Cf(nt + 1, 2, 2)
    for n = 0:nt+1
        ϵ[n] = a
    end
    A2 = ℱ(C, FERMI)
    from_ham!(A2, μ, ϵ, β, h; fix_hamiltonian = true)
    @test sum(distance(A2, exact, tstp) for tstp = 0:nt) < 1e-6
    #
    # Time-dependent propagators, fix_hamiltonian = true
    Δ, V, ω = 1.0, 0.5, 0.7
    H = eq_ham_time(nt + 1, h, Δ, V, ω)
    U4 = [Matrix{ComplexF64}(I, 2, 2) for _ = 1:nt+1]
    err = norm(eq_U(0.0, Δ, V, ω) - U4[1])
    for nkb = 2:nt
        KadanoffBaym._propagator_cf4!(U4, nkb, H, h, kt, true)
        err += norm(eq_U((nkb - 1) * h, Δ, V, ω) - U4[nkb])
    end
    @test err < 1e-6
    #
    # Time-dependent propagators, the Hamiltonian is extrapolated
    Htrun = Cf(nt + 1, 2, 2)
    U4 = [Matrix{ComplexF64}(I, 2, 2) for _ = 1:nt+1]
    err = 0.0
    for nkb = 2:nt
        tstp = nkb - 1
        n₁ = tstp ≤ kt ? kt : tstp - 1
        for trun = 0:n₁+1
            Htrun[trun] = H[trun]
        end
        KadanoffBaym._propagator_cf4!(U4, nkb, Htrun, h, kt, false)
        err += norm(eq_U((nkb - 1) * h, Δ, V, ω) - U4[nkb])
    end
    @test err < 1e-6
    #
    # Time-dependent fermions and bosons
    for sign in (FERMI, BOSE)
        A = ℱ(C, sign)
        from_ham!(A, μ, H, β, h; fix_hamiltonian = true)
        exact = eq_exact_time(C, β, h, Δ, V, ω, sign)
        @test sum(distance(A, exact, tstp) for tstp = 0:nt+1) < 1e-6
    end
end

@testset "equilibrium: from_ham! (constant H)" begin
    nt, ntau, β, h = 21, 41, 3.0, 0.05
    #
    # 1 × 1, analytic formulas
    for sign in (FERMI, BOSE)
        ε, μ = 0.4, sign == FERMI ? 0.1 : -0.5
        λ = μ - ε
        C = Cn(nt, ntau, 1, 1, (nt - 1) * h, β)
        G = ℱ(C, sign)
        from_ham!(G, μ, fill(ComplexF64(ε), 1, 1), β, h)
        δτ = β / (ntau - 1)
        nξ(x) = sign == FERMI ? 1.0 / (exp(β * x) + 1.0) : 1.0 / (exp(β * x) - 1.0)
        err = 0.0
        for m = 1:ntau
            τ = (m - 1) * δτ
            gm = sign == FERMI ? -exp(τ * λ) * nξ(λ) : exp(τ * λ) * nξ(λ)
            err = max(err, abs(G.mat[m][1,1] - gm))
            for n = 1:nt
                t = (n - 1) * h
                gtv = -sign * im * exp(im * λ * t) * exp(-τ * λ) * nξ(-λ)
                err = max(err, abs(G.lmix[n, m][1,1] - gtv))
            end
        end
        for n = 1:nt, j = 1:n
            t, t′ = (n - 1) * h, (j - 1) * h
            err = max(err, abs(G.ret[n, j][1,1] + im * exp(im * λ * (t - t′))))
            gl = -sign * im * exp(im * λ * (t′ - t)) * nξ(-λ)
            err = max(err, abs(G.less[j, n][1,1] - gl))
        end
        @test err < 1e-12
    end
    #
    # 2 × 2 with complex off-diagonal elements: KMS, retarded diagonal,
    # and anti-hermiticity of the lesser component
    H0 = ComplexF64[-0.3 0.4-0.2im; 0.4+0.2im 0.6]
    for sign in (FERMI, BOSE)
        μ = sign == FERMI ? 0.1 : -1.0
        C = Cn(nt, ntau, 2, 2, (nt - 1) * h, β)
        G = ℱ(C, sign)
        from_ham!(G, μ, H0, β, h)
        for m = 1:ntau
            @test norm(G.lmix[1, m] - (im * sign) * G.mat[ntau - m + 1]) < 1e-13
        end
        for n = 1:nt
            @test norm(G.ret[n, n] + im * I) < 1e-13
            @test norm(G.less[n, n] + G.less[n, n]') < 1e-13
            @test norm(G.less[1, n] + G.lmix[n, 1]') < 1e-13
        end
        #
        G2 = ℱ(C, sign)
        from_ham!(G2, μ, H0, β, h)
        @test isequal(G2, G)
        #
        # The time-dependent version with a constant Hamiltonian
        Hc = Cf(nt, 2, 2)
        for n = 0:nt
            Hc[n] = H0
        end
        for cf in (2, 4)
            G3 = ℱ(C, sign)
            from_ham!(G3, μ, Hc, β, h; cf_order = cf)
            @test maximum(maximum(abs.(getfield(G3, s).data .- getfield(G, s).data))
                          for s in (:mat, :ret, :lmix, :less)) < 1e-12
        end
    end
end

@testset "equilibrium: from_ham! (CF2 and CF4)" begin
    nt, ntau, β, h = 101, 51, 5.0, 0.02
    Δ, V, ω = 1.0, 0.5, 0.7
    C = Cn(nt, ntau, 2, 2, (nt - 1) * h, β)
    H = eq_ham_time(nt, h, Δ, V, ω)
    for sign in (FERMI, BOSE)
        exact = eq_exact_time(C, β, h, Δ, V, ω, sign)
        errs = Dict{Int,Float64}()
        for cf in (2, 4)
            G = ℱ(C, sign)
            from_ham!(G, 0.0, H, β, h; cf_order = cf)
            errs[cf] = maximum(distance(G, exact, n) for n = 0:nt)
        end
        @test errs[4] < 1e-6
        @test errs[4] ≤ errs[2]
    end
    #
    # The Hamiltonian is extrapolated if fix_hamiltonian = false. The
    # recursive extrapolation is only accurate for a few time steps.
    nts = 11
    Cs = Cn(nts, ntau, 2, 2, (nts - 1) * h, β)
    Htrun = Cf(nts, 2, 2)
    for n = 0:6
        Htrun[n] = H[n]
    end
    G = ℱ(Cs, FERMI)
    from_ham!(G, 0.0, Htrun, β, h; fix_hamiltonian = false)
    exact = eq_exact_time(Cs, β, h, Δ, V, ω, FERMI)
    @test maximum(distance(G, exact, n) for n = 0:nts) < 1e-6
    @test norm(Htrun[nts] - H[nts]) > 0.0
    @test norm(Htrun[nts] - H[nts]) < 1e-4
end

@testset "equilibrium: DOS integrals" begin
    KB = KadanoffBaym
    #
    # Panel control flow: limit = 1 returns one panel, otherwise the
    # number of panels is max(2, limit - 1)
    ρ, lo, hi = KB._bethe_dos(1.0, 0.0)
    for (limit, npanel) in ((1, 1), (2, 2), (3, 2), (4, 3), (10, 9), (50, 49))
        P = KB._adft_sample(x -> complex(ρ(x)), lo, hi, 20, limit)
        @test length(P.a) == npanel
        @test P.a[1] == lo && P.b[end] == hi
        @test all(P.a[2:end] .== P.b[1:end-1])
        @test all(length.(P.f) .== 21)
    end
    #
    # Corrected DFT of a cubic polynomial is exact
    P = KB._adft_sample(x -> complex(x^3), 0.0, 1.0, 20, 1)
    for w in (0.0, 0.3, 5.0)
        r, _ = KB._adft_eval(P, w)
        if w == 0.0
            ex = 0.25
        else
            z = im * w
            ex = cis(w) * (1 / z - 3 / z^2 + 6 / z^3 - 6 / z^4) + 6 / z^4
        end
        @test abs(r - ex) < 1e-12
    end
    #
    # Exponential function, fourth order accuracy
    P = KB._adft_sample(x -> complex(exp(-x)), 0.0, 2.0, 20, 50)
    for w in (0.0, 1.0, -3.0)
        r, e = KB._adft_eval(P, w)
        ex = (exp((im * w - 1) * 2.0) - 1) / (im * w - 1)
        @test abs(r - ex) < 1e-11
        @test isfinite(e)
    end
    #
    # Bethe DOS is normalized
    P = KB._adft_sample(x -> complex(ρ(x)), lo, hi, 20, 200)
    @test abs(KB._adft_eval(P, 0.0)[1] - 1.0) < 1e-11
    #
    # Error paths
    @test_throws ArgumentError KB._adft_sample(x -> 1.0 + 0im, 1.0, 1.0, 20, 5)
    @test_throws ArgumentError KB._adft_sample(x -> 1.0 + 0im, 0.0, 1.0, 21, 5)
    @test_throws ArgumentError KB._adft_sample(x -> 1.0 + 0im, 0.0, 1.0, 18, 5)
    @test_throws ArgumentError KB._adft_sample(x -> 1.0 + 0im, 0.0, 1.0, 20, 0)
    @test_throws DomainError KB._adft_eval(
        KB._adft_sample(x -> complex(1.0 / x), 0.0, 1.0, 20, 1), 0.0)
end

@testset "equilibrium: from_dos!" begin
    nt, ntau, β, h = 21, 41, 2.0, 0.05
    #
    # Narrow Gaussian DOS approaches the single level. Note that the real
    # time components of from_ham! contain the phase exp(iμ(t - t')),
    # which is absent in the DOS representation.
    for sign in (FERMI, BOSE)
        ε₀, μ = 0.5, sign == FERMI ? 0.2 : -0.5
        C = Cn(nt, ntau, 1, 1, (nt - 1) * h, β)
        G₀ = ℱ(C, sign)
        from_ham!(G₀, μ, fill(ComplexF64(ε₀), 1, 1), β, h)
        for n = 1:nt
            t = (n - 1) * h
            for m = 1:ntau
                G₀.lmix[n, m] = G₀.lmix[n, m] * cis(-μ * t)
            end
            for j = 1:n
                t′ = (j - 1) * h
                G₀.ret[n, j] = G₀.ret[n, j] * cis(-μ * (t - t′))
                G₀.less[j, n] = G₀.less[j, n] * cis(-μ * (t′ - t))
            end
        end
        errs = Float64[]
        for σ in (0.02, 0.01)
            dos(x) = exp(-(x - ε₀)^2 / (2σ^2)) / (sqrt(2π) * σ)
            G = ℱ(C, sign)
            from_dos!(G, dos, ε₀ - 10σ, ε₀ + 10σ, β, h; μ = μ)
            push!(errs, maximum(maximum(abs.(getfield(G, s).data .- getfield(G₀, s).data))
                                for s in (:mat, :ret, :lmix, :less)))
            #
            # Retarded component: G^R(t) = -i exp(-iε₀t - σ²t²/2)
            for n = 1:nt
                t = (n - 1) * h
                @test abs(G.ret[n, 1][1,1] + im * exp(-im * ε₀ * t - σ^2 * t^2 / 2)) < 1e-10
            end
        end
        @test errs[1] < 1e-2
        @test errs[2] < errs[1] / 3
    end
    #
    # Bethe lattice: KMS and boundary relations, diagonal matrices
    for sign in (FERMI, BOSE)
        μ = sign == FERMI ? 0.3 : -2.5
        C = Cn(nt, ntau, 2, 2, (nt - 1) * h, β)
        G = ℱ(C, sign)
        from_dos_bethe!(G, β, h; μ = μ)
        @test norm(G.ret[1, 1] + im * I) < 1e-10
        for m = 1:ntau
            @test norm(G.lmix[1, m] - (im * sign) * G.mat[ntau - m + 1]) < 1e-10
            @test G.mat[m][1,2] == 0.0 && G.mat[m][1,1] == G.mat[m][2,2]
        end
        @test norm(G.less[1, 1] + G.lmix[1, 1]') < 1e-10
        #
        # The Matsubara-only version
        Gm = ℱ(C, sign)
        from_dos_mat_bethe!(Gm, β; μ = μ)
        @test Gm.mat.data == G.mat.data
        @test all(iszero, Gm.ret.data)
        #
        # Shifted and rescaled Bethe DOS
        G2 = ℱ(C, sign)
        from_dos_bethe!(G2, β, h; V = 0.5, E0 = 0.3, μ = μ)
        @test norm(G2.ret[1, 1] + im * I) < 1e-10
        @test G2.ret[3, 1] != G.ret[3, 1]
    end
    #
    # Error paths
    C = Cn(nt, ntau, 1, 1, (nt - 1) * h, β)
    G = ℱ(C, FERMI)
    @test_throws ArgumentError from_dos_bethe!(G, β, h; V = 0.0)
    @test_throws ArgumentError from_dos_bethe!(G, β, h; V = -1.0)
    @test_throws ArgumentError from_dos_bethe!(G, β, h; nn = 21)
    @test_throws ArgumentError from_dos_bethe!(G, β, h; limit = 0)
    @test_throws ArgumentError from_dos!(G, x -> 1.0, 1.0, 0.0, β, h)
    @test_throws DomainError from_dos!(G, x -> -1.0, 0.0, 1.0, β, h)
    @test_throws DomainError from_dos!(G, x -> NaN, 0.0, 1.0, β, h)
    @test_throws DomainError from_dos!(ℱ(C, BOSE), x -> 1.0, -1.0, 1.0, β, h)
end

@testset "equilibrium: error paths of from_ham!" begin
    nt, ntau, β, h = 11, 21, 2.0, 0.05
    C = Cn(nt, ntau, 2, 2, (nt - 1) * h, β)
    G = ℱ(C, FERMI)
    H0 = ComplexF64[0.1 0.2; 0.2 -0.3]
    @test_throws ArgumentError from_ham!(G, 0.0, ComplexF64[0.1 0.2; 0.0 -0.3], β, h)
    @test_throws DimensionMismatch from_ham!(G, 0.0, zeros(ComplexF64, 3, 3), β, h)
    @test_throws DomainError from_ham!(G, 0.0, ComplexF64[NaN 0.0; 0.0 0.0], β, h)
    Hc = Cf(nt, 2, 2)
    for n = 0:nt
        Hc[n] = H0
    end
    @test_throws ArgumentError from_ham!(G, 0.0, Hc, β, h; cf_order = 3)
    @test_throws ArgumentError from_ham!(G, 0.0, Hc, β, h; solve_order = 0)
    @test_throws ArgumentError from_ham!(G, 0.0, Hc, β, h; solve_order = 6)
    @test_throws ArgumentError from_ham!(G, 0.0, Cf(nt - 2, 2, 2), β, h)
    Hn = deepcopy(Hc)
    Hn[5] = ComplexF64[0.1 0.2; 0.0 -0.3]
    @test_throws ArgumentError from_ham!(G, 0.0, Hn, β, h)
end
