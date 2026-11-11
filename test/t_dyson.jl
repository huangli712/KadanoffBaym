#
# t_dyson.jl
#
# To test the real time Dyson solvers dyson_start!, dyson_timestep!, and
# dyson!.
#
# NESSi Catch tests ported here:
#   libcntr/test/dyson.cpp:23-96, section Integro-differential form (the
#   section Integral form uses VIE2, which belongs to Phase 4)
#   libcntr/test/dyson_new.cpp: the start/timestep paths of the same
#   embedding model are covered by the start/timestep/full testsets below
#

using Test
using LinearAlgebra
using KadanoffBaym

const KB = KadanoffBaym

#
# Copy the block r × r of the full contour function Gf into Gs
#
function dy_block!(Gs::ℱ, Gf::ℱ, r)
    nt, ntau = getntime(Gf), getntau(Gf)
    for m = 1:ntau
        Gs.mat[m] = Gf.mat[m][r, r]
    end
    for i = 1:nt, m = 1:ntau
        Gs.lmix[i, m] = Gf.lmix[i, m][r, r]
    end
    for i = 1:nt, j = 1:i
        Gs.ret[i, j] = Gf.ret[i, j][r, r]
        Gs.less[j, i] = Gf.less[j, i][r, r]
    end
    return Gs
end

#
# Σ = V gb V', where gb is the free Green's function of Hb
#
function dy_sigma(C::Cn, Hb, V, μ, β, h, sign)
    nt, ntau = getntime(C), getntau(C)
    Cb = Cn(nt, ntau, size(Hb, 1), size(Hb, 1), (nt - 1) * h, β)
    gb = ℱ(Cb, sign)
    from_ham!(gb, μ, Hb, β, h)
    Σ = ℱ(C, sign)
    for m = 1:ntau
        Σ.mat[m] = V * gb.mat[m] * V'
    end
    for i = 1:nt, m = 1:ntau
        Σ.lmix[i, m] = V * gb.lmix[i, m] * V'
    end
    for i = 1:nt, j = 1:i
        Σ.ret[i, j] = V * gb.ret[i, j] * V'
        Σ.less[j, i] = V * gb.less[j, i] * V'
    end
    return Σ
end

#
# Constant Hamiltonian
#
function dy_ham(nt, H0)
    d = size(H0, 1)
    H = Cf(nt, d, d)
    for i = 0:nt
        H[i] = H0
    end
    return H
end

#
# Embedding model: the 2 × 2 block (1:2) of the free Green's function of
# [Ha V; V' Hb] solves the Dyson equation for Ha with Σ = V gb V'. The
# blocks do not commute, and the couplings are complex.
#
const DY_HA = ComplexF64[-1.0 0.3im; -0.3im 0.5]
const DY_HB = ComplexF64[1.0 0.2+0.1im; 0.2-0.1im 1.5]
const DY_V = ComplexF64[0.4 0.1im; -0.2 0.3]
#
function dy_embedding(nt, ntau, β, h, sign)
    μ = sign == FERMI ? 0.1 : -2.0
    C4 = Cn(nt, ntau, 4, 4, (nt - 1) * h, β)
    C2 = Cn(nt, ntau, 2, 2, (nt - 1) * h, β)
    Gf = ℱ(C4, sign)
    from_ham!(Gf, μ, [DY_HA DY_V; DY_V' DY_HB], β, h)
    G_exact = dy_block!(ℱ(C2, sign), Gf, 1:2)
    Σ = dy_sigma(C2, DY_HB, DY_V, μ, β, h, sign)
    H = dy_ham(nt, DY_HA)
    return C2, μ, H, Σ, G_exact
end

#
# Maximum absolute error of all elements of all components
#
function dy_maxabs(A::ℱ, B::ℱ)
    return maximum(maximum(abs.(getfield(A, s).data .- getfield(B, s).data))
                   for s in (:mat, :ret, :lmix, :less))
end

@testset "dyson: free model (sign = $sign)" for sign in (FERMI, BOSE)
    nt, ntau, β, h = 51, 41, 2.0, 0.005
    μ = sign == FERMI ? 0.2 : -1.5
    H0 = ComplexF64[-0.5 0.3im; -0.3im 0.4]
    C = Cn(nt, ntau, 2, 2, (nt - 1) * h, β)
    G_exact = ℱ(C, sign)
    from_ham!(G_exact, μ, H0, β, h)
    Σ = ℱ(C, sign)
    G = ℱ(C, sign)
    dyson!(G, μ, dy_ham(nt, H0), Σ, β, h; solve_order = 5)
    @test dy_maxabs(G, G_exact) < 1e-11
end

@testset "dyson: NESSi Catch (dyson.cpp)" begin
    eps1, eps2, lam, μ, β, dt = -1.0, 1.0, 0.2, -0.1, 10.0, 0.05
    Nt, Ntau = 200, 400
    h2x2 = ComplexF64[eps1 im*lam; -im*lam eps2]
    C2 = Cn(Nt + 1, Ntau + 1, 2, 2, Nt * dt, β)
    C1 = Cn(Nt + 1, Ntau + 1, 1, 1, Nt * dt, β)
    G2x2 = ℱ(C2, FERMI)
    from_ham!(G2x2, μ, h2x2, β, dt)
    G_exact = dy_block!(ℱ(C1, FERMI), G2x2, 1:1)
    Σ = dy_sigma(C1, fill(ComplexF64(eps2), 1, 1), fill(ComplexF64(lam), 1, 1),
                 μ, β, dt, FERMI)
    H = dy_ham(Nt + 1, fill(ComplexF64(eps1), 1, 1))
    #
    G = ℱ(C1, FERMI)
    dyson_mat!(G, Σ, μ, H, β; solve_order = 5)
    dyson_start!(G, μ, H, Σ, β, dt; solve_order = 5)
    for nkb = 7:Nt+1
        dyson_timestep!(nkb, G, μ, H, Σ, β, dt; solve_order = 5)
    end
    err = sum(distance(G_exact, G, nkb) for nkb = 1:Nt+1)
    @test err / Nt < 1e-3
    # The actual accuracy is much better
    @test err / Nt < 1e-5
end

@testset "dyson: embedding model (sign = $sign)" for sign in (FERMI, BOSE)
    nt, ntau, β, h = 41, 81, 2.0, 0.05
    C, μ, H, Σ, G_exact = dy_embedding(nt, ntau, β, h, sign)
    errs = Float64[]
    for k = 1:5
        G = ℱ(C, sign)
        dyson!(G, μ, H, Σ, β, h; solve_order = k)
        push!(errs, sum(distance(G_exact, G, nkb) for nkb = 1:nt) / (nt - 1))
    end
    # Normalized error of the default order, and the higher orders are
    # more accurate
    @test errs[5] < 1e-3
    @test issorted(errs; rev = true)
end

@testset "dyson: start, timestep, and full" begin
    nt, ntau, β, h = 21, 41, 2.0, 0.05
    sign = FERMI
    C, μ, H, Σ, G_exact = dy_embedding(nt, ntau, β, h, sign)
    for k in (1, 3, 5)
        # Full solution
        Gf = ℱ(C, sign)
        dyson!(Gf, μ, H, Σ, β, h; solve_order = k)
        #
        # Manual Matsubara, start, and time stepping
        G = ℱ(C, sign)
        dyson_mat!(G, Σ, μ, H, β; solve_order = k)
        Gm = deepcopy(G)
        dyson_start!(G, μ, H, Σ, β, h; solve_order = k)
        #
        # The Matsubara component is not modified by the start-up
        @test G.mat.data == Gm.mat.data
        #
        # Only the first k + 1 time steps are written
        for i = k+2:nt
            @test all(iszero, G.lmix.data[:, :, i, :])
            for j = 1:i
                @test all(iszero, KB._block(G.ret, i, j))
                @test all(iszero, KB._block(G.less, j, i))
            end
        end
        #
        # Boundary values
        for n = 1:k+1
            @test KB._block(G.ret, n, n) == -im * I
            @test norm(G.less[1, n] + G.lmix[n, 1]') < 1e-13
            # G^<(t,t) is anti-hermitian up to the discretization error
            @test norm(G.less[n, n] + G.less[n, n]') ≤
                  2 * norm(G.less[n, n] - G_exact.less[n, n]) + 1e-13
        end
        for m = 1:ntau
            @test norm(G.lmix[1, m] - (im * sign) * G.mat[ntau-m+1]) < 1e-13
        end
        #
        # Time stepping does not modify the previous time steps
        for nkb = k+2:nt
            Gp = deepcopy(G)
            dyson_timestep!(nkb, G, μ, H, Σ, β, h; solve_order = k)
            @test G.mat.data == Gp.mat.data
            @test G.lmix.data[:, :, 1:nkb-1, :] == Gp.lmix.data[:, :, 1:nkb-1, :]
            for i = 1:nkb-1, j = 1:i
                @test KB._block(G.ret, i, j) == KB._block(Gp.ret, i, j)
                @test KB._block(G.less, j, i) == KB._block(Gp.less, j, i)
            end
            @test KB._block(G.ret, nkb, nkb) == -im * I
            # The anti-hermiticity of G^<(t,t) is not imposed by the time
            # stepping, it holds up to the discretization error
            @test norm(G.less[nkb, nkb] + G.less[nkb, nkb]') ≤
                  2 * norm(G.less[nkb, nkb] - G_exact.less[nkb, nkb]) + 1e-13
            @test norm(G.less[1, nkb] + G.lmix[nkb, 1]') < 1e-13
        end
        #
        # Full solution and manual steps are identical
        @test G == Gf
    end
end

#
# Fixture with non-hermitian blocks and a nonzero Σ^⌉ for the directed
# regression tests. The self-energy is not required to be a physical one.
#
function dy_fixture(nt, ntau, k)
    β, h, sign = 2.0, 0.05, FERMI
    C, μ, H, Σ, _ = dy_embedding(nt, ntau, β, h, sign)
    # Non-hermitian perturbation of all the stored blocks
    P = ComplexF64[0.02 0.05im; -0.03 0.01+0.04im]
    for m = 1:ntau
        Σ.mat[m] = Σ.mat[m] + 0.1 * P
    end
    for i = 1:nt, m = 1:ntau
        Σ.lmix[i, m] = Σ.lmix[i, m] + (0.1 + 0.01i) * P
    end
    for i = 1:nt, j = 1:i
        Σ.ret[i, j] = Σ.ret[i, j] + 0.05 * P
        Σ.less[j, i] = Σ.less[j, i] + 0.05im * P'
    end
    G = ℱ(C, sign)
    dyson_mat!(G, Σ, μ, H, β; solve_order = k)
    return C, μ, H, Σ, G, β, h
end

@testset "dyson: directed regressions" begin
    nt, ntau, k = 16, 41, 4
    C, μ, H, Σ, G0, β, h = dy_fixture(nt, ntau, k)
    sign = getsign(G0)
    I4 = Integrator(k)
    δτ = β / (ntau - 1)
    #
    # (a) Sentinels in the unknown lesser blocks and the current
    #     left-mixing row do not change the results
    G = deepcopy(G0)
    dyson_start!(G, μ, H, Σ, β, h; solve_order = k)
    Gs = deepcopy(G0)
    for i = 1:k+1, j = 1:i
        KB._block(Gs.less, j, i) .= 7.0 + 3.0im
    end
    dyson_start!(Gs, μ, H, Σ, β, h; solve_order = k)
    @test Gs == G
    for nkb = k+2:nt
        Gs = deepcopy(G)
        dyson_timestep!(nkb, G, μ, H, Σ, β, h; solve_order = k)
        for j = 1:nkb
            KB._block(Gs.less, j, nkb) .= -5.0 + 2.0im
        end
        Gs.lmix.data[:, :, nkb, :] .= 3.0 - 1.0im
        dyson_timestep!(nkb, Gs, μ, H, Σ, β, h; solve_order = k)
        @test Gs == G
    end
    #
    # (b) Mixed Matsubara source: the start-up left-mixing component
    #     satisfies the discrete equation with the source term given by
    #     conv_lmix_mat (δτ ∫ Σ^⌉ G^M, no extra ±i). The same residual
    #     with ±i J is large.
    J = KB.Gˡᵐⁱˣ(nt, ntau, 2, 2)
    for n = 2:k+1
        conv_lmix_mat(n, J, Σ.lmix, G.mat, I4, β, sign)
    end
    function res_start_lmix(c)
        r = 0.0
        for n = 1:k, m = 1:ntau
            R = -c .* J[n+1, m]
            for j = 0:k
                R = R + (im / h * I4.PDW[n,j]) * G.lmix[j+1, m]
                R = R - h * I4.GIW[n,j] * Σ.ret[n+1, j+1] * G.lmix[j+1, m]
            end
            R = R + (μ * I - H[n+1]) * G.lmix[n+1, m]
            r = max(r, norm(R))
        end
        return r
    end
    @test res_start_lmix(1.0) < 1e-10
    @test res_start_lmix(im) > 1e-3
    @test res_start_lmix(-im) > 1e-3
    #
    #     Time stepping: the residual of the BDF equation with the full
    #     convolution [Σ * G]^⌉ vanishes.
    function res_step_lmix(nkb, c)
        # Σ^R * G^⌉ (including the implicit term) + c δτ ∫ Σ^⌉ G^M
        Jn = KB.Gˡᵐⁱˣ(nt, ntau, 2, 2)
        conv_lmix_mat(nkb, Jn, Σ.lmix, G.mat, I4, β, sign)
        Tn = KB.Gˡᵐⁱˣ(nt, ntau, 2, 2)
        conv_ret_lmix(nkb, Tn, Σ.ret, Σ.ret, G.lmix, G.lmix, I4, h)
        r = 0.0
        for m = 1:ntau
            R = -(Tn[nkb, m] + c * Jn[nkb, m])
            for p = 0:k+1
                R = R + (im / h * I4.BDW[p]) * G.lmix[nkb-p, m]
            end
            R = R + (μ * I - H[nkb]) * G.lmix[nkb, m]
            r = max(r, norm(R))
        end
        return r
    end
    @test res_step_lmix(nt, 1.0) < 1e-10
    @test res_step_lmix(nt, im) > 1e-3
    #
    # (c) Start-up lesser source: the blocks j > n of the scratch-target
    #     convolutions are needed. Truncating them changes the results.
    Gt = deepcopy(G0)
    KB._dyson_start_ret!(Gt, μ, H, Σ, h, I4)
    KB._dyson_start_lmix!(Gt, μ, H, Σ, β, h, I4)
    cws = KB._ConvWorkspace(Σ, Gt)
    diff = 0.0
    for n = 0:k-1
        buf = [zeros(ComplexF64, 2, 2) for _ = 1:k+1]
        KB._incr_convolution_lmix_rmix!(buf, n + 1, Σ, Σ, nothing, Gt, Gt, I4, β, h, 1.0, cws)
        KB._incr_convolution_less_adv!(buf, n + 1, Σ, Σ, nothing, Gt, Gt, I4, β, h, 1.0, cws)
        @test all(b -> norm(b) > 0.0, buf[n+2:k+1])
        tr = deepcopy(buf)
        for j = n+2:k+1
            fill!(tr[j], 0.0)
        end
        Ga = deepcopy(Gt)
        Gb = deepcopy(Gt)
        KB._dyson_less_core!(n, buf, Ga, μ, H, Σ, h, I4)
        KB._dyson_less_core!(n, tr, Gb, μ, H, Σ, h, I4)
        diff = max(diff, maximum(norm(Ga.less[j, n+1] - Gb.less[j, n+1]) for j = 1:n+1))
    end
    @test diff > 1e-6
    #
    #     Time stepping of the lesser component: the residual of the BDF
    #     equation with the full lesser convolution vanishes.
    nkb = nt
    W = 𝒻(nkb, ntau, 2, 2, sign)
    incr_convolution_les!(nkb, W, Σ, Σ, nothing, G, G, I4, β, h, 1.0)
    r = 0.0
    for j = k+2:nkb
        R = -W.less.data[j]
        for p = 0:k+1
            R = R + (im / h * I4.BDW[p]) * G.less[j-p, nkb]
        end
        R = R + (μ * I - H[j]) * G.less[j, nkb]
        r = max(r, norm(R))
    end
    @test r < 1e-10
end

@testset "dyson: allocations and private kernels" begin
    function alloc_step(nt)
        ntau, β, h = 129, 2.0, 0.02
        C, μ, H, Σ, _ = dy_embedding(nt, ntau, β, h, FERMI)
        G = ℱ(C, FERMI)
        dyson!(G, μ, H, Σ, β, h; solve_order = 5)
        dyson_timestep!(nt, G, μ, H, Σ, β, h; solve_order = 5)
        return @allocated dyson_timestep!(nt, G, μ, H, Σ, β, h; solve_order = 5)
    end
    a64 = alloc_step(64)
    a128 = alloc_step(128)
    @test a128 ≤ 512 * 1024
    @test a128 - a64 ≤ 64 * 1024
    #
    # The private kernels dispatch on CnAbstractFunction{C64}, there are no
    # methods specialized for ℱ, and H is a CnAbstractFunction{C64}.
    for f in (KB._dyson_start_ret!, KB._dyson_start_lmix!, KB._dyson_start_less!,
              KB._dyson_timestep_ret!, KB._dyson_timestep_lmix!, KB._dyson_timestep_less!)
        ms = collect(methods(f))
        @test length(ms) == 1
        sig = Base.unwrap_unionall(ms[1].sig).parameters
        gtypes = filter(t -> t isa Type && t <: CnAbstractFunction, collect(sig))
        @test length(gtypes) == 3
        @test all(t -> t == CnAbstractFunction{ComplexF64}, gtypes)
    end
end

@testset "dyson: error paths" begin
    nt, ntau, β, h = 11, 41, 2.0, 0.05
    C, μ, H, Σ, _ = dy_embedding(nt, ntau, β, h, FERMI)
    G = ℱ(C, FERMI)
    dyson_mat!(G, Σ, μ, H, β)
    #
    # Time step order
    @test_throws ArgumentError dyson_timestep!(6, G, μ, H, Σ, β, h; solve_order = 5)
    @test_throws ArgumentError dyson_timestep!(3, G, μ, H, Σ, β, h; solve_order = 2)
    @test_throws ArgumentError dyson_timestep!(nt + 1, G, μ, H, Σ, β, h; solve_order = 2)
    #
    # Short grids
    Cs = Cn(4, ntau, 2, 2, 3h, β)
    Gs = ℱ(Cs, FERMI)
    Σs = ℱ(Cs, FERMI)
    @test_throws ArgumentError dyson_start!(Gs, μ, dy_ham(4, DY_HA), Σs, β, h; solve_order = 5)
    @test_throws ArgumentError dyson!(Gs, μ, dy_ham(4, DY_HA), Σs, β, h; solve_order = 5)
    @test_throws ArgumentError dyson_start!(G, μ, dy_ham(3, DY_HA), Σ, β, h; solve_order = 5)
    #
    # Solve order
    @test_throws ArgumentError dyson_start!(G, μ, H, Σ, β, h; solve_order = 0)
    @test_throws ArgumentError dyson!(G, μ, H, Σ, β, h; solve_order = 6)
    #
    # Non-hermitian Hamiltonian at a real time step
    Hn = dy_ham(nt, DY_HA)
    Hn[4] = ComplexF64[-1.0 0.3; 0.0 0.5]
    @test_throws ArgumentError dyson_start!(G, μ, Hn, Σ, β, h; solve_order = 5)
    @test_throws ArgumentError dyson!(G, μ, Hn, Σ, β, h; solve_order = 2)
    #
    # Statistics and shapes
    @test_throws ArgumentError dyson_start!(ℱ(C, BOSE), μ, H, Σ, β, h)
    C1 = Cn(nt, ntau, 1, 1, (nt - 1) * h, β)
    @test_throws DimensionMismatch dyson_start!(ℱ(C1, FERMI), μ, H, Σ, β, h)
    C3 = Cn(nt + 1, ntau, 2, 2, nt * h, β)
    @test_throws DimensionMismatch dyson_start!(ℱ(C3, FERMI), μ, H, Σ, β, h)
end
