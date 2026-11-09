#
# t_utilities.jl
#
# To test the utility functions (extrapolation, interpolation, distance,
# boundary values from the Matsubara component, and hermitian Matsubara
# component).
#
# NESSi Catch tests ported here:
#   libcntr/test/utilities.cpp:213-488 (all sections)
#   libcntr/test/function.cpp:110-134 (Set/get for function)
#   libcntr/test/function.cpp:136-168 (Extrapolation)
#   libcntr/test/function.cpp:170-227 (Interpolation)
#   libcntr/test/function.cpp:228-340 -> delegated to t_gf_tools.jl
#   libcntr/test/function.cpp:342-372 -> skipped (HDF5 only)
#

isdefined(Main, :kb_test_profile) ||
    include(joinpath(@__DIR__, "..", "common", "profile.jl"))
isdefined(Main, :test_order_gate) ||
    include(joinpath(@__DIR__, "..", "common", "order_gate.jl"))

using LinearAlgebra

#
# Helpers ported from libcntr/test/utilities.cpp:24-33
#
function ut_lagrange_weights(p::Int, h::Float64, x::Float64)
    w = ones(Float64, p + 1)
    for i = 0:p
        for j = 0:p
            if i != j
                w[i+1] *= (x - h * j) / (i * h - h * j)
            end
        end
    end
    return w
end

#
# Smooth matrix-valued test function
#
ut_F(t) = [cos(-1.0 * t) -0.2im * sin(0.7 * t); 0.2im * sin(0.7 * t) cos(0.8 * t)] .* exp(0.3 * t)

#
# Errors of extrapolation / interpolation at a fixed physical time with
# step h = 1/N. `ord` is the order really used (for mutation fixtures).
#
function ut_extrap_error(k::Int, N::Int, ord::Int)
    h = 1.0 / N
    t⁺ = 1.3
    f = Cf(k + 2, 2)
    for i = 1:k+1
        f[i] = ut_F(t⁺ - (k + 2 - i) * h)
    end
    extrapolate_timestep!(k + 1, f, ord)
    return norm(f[k+2] - ut_F(t⁺))
end
#
function ut_interp_error(k::Int, N::Int, ord::Int)
    h = 1.0 / N
    t⁺ = 1.3
    f = Cf(k + 1, 2)
    for i = 1:k+1
        f[i] = ut_F(t⁺ - (k + 1 - i) * h)
    end
    x = interpolation(k + 1, k + 0.5, f, ord)
    return norm(x - ut_F(t⁺ - 0.5 * h))
end

# Grids N = 2^m, chosen per order so that at least 3 errors stay above
# the round-off plateau and the finest error is below 1e-8.
const UT_GRIDS = Dict(1 => 12:15, 2 => 6:9, 3 => 5:8, 4 => 4:7, 5 => 3:6)

@testset verbose = true "KadanoffBaym: utilities.jl" begin

    @testset "extrapolate Cf: polynomial exactness" begin
        h = 0.1
        for k = 0:5
            for dims in ((1, 1), (2, 2))
                Ms = [rand(ComplexF64, dims...) for _ = 0:k]
                P(t) = sum(Ms[r+1] .* t^r for r = 0:k)
                ntime = k + 4
                f = Cf(ntime, dims...)
                for i = 1:ntime
                    f[i] = P((i - 1) * h)
                end
                f[0] = fill(C64(7.0), dims...)
                g = deepcopy(f)
                n = ntime - 1
                g[n+1] = zeros(C64, dims...)
                extrapolate_timestep!(n, g, k)
                @test isapprox(g[n+1], f[n+1]; atol = 1e-13, rtol = 1e-12)
                @test g[0] == f[0]
                if k ≥ 1
                    g[n+1] = zeros(C64, dims...)
                    extrapolate_timestep!(n, g, Integrator(k))
                    @test isapprox(g[n+1], f[n+1]; atol = 1e-13, rtol = 1e-12)
                end
            end
        end
    end

    @testset "extrapolate ℱ: polynomial exactness" begin
        ntime = 12
        ntau = 7
        h = 0.1
        C = Cn(ntime, ntau, 2, 2, (ntime - 1) * h, 2.0)
        for k = 0:5
            Q = [rand(ComplexF64, 2, 2) for _ = 0:k, _ = 0:k]
            W = [rand(ComplexF64, 2, 2) for _ = 0:k, _ = 0:k]
            Qf(t, s) = sum(Q[a+1,b+1] .* t^a * s^b for a = 0:k, b = 0:k if a + b ≤ k)
            Wf(t, s) = sum(W[a+1,b+1] .* t^a * s^b for a = 0:k, b = 0:k if a + b ≤ k)
            R(t, s) = Qf(t, s) - Qf(s, t)'
            L(t, s) = Wf(t, s) - Wf(s, t)'
            G = ℱ(C)
            for i = 1:ntime, j = 1:i
                G.ret[i,j] = R((i - 1) * h, (j - 1) * h)
            end
            for j = 1:ntime, i = 1:j
                G.less[i,j] = L((i - 1) * h, (j - 1) * h)
            end
            for i = 1:ntime, m = 1:ntau
                G.lmix[i,m] = Qf((i - 1) * h, 0.1 * m)
            end
            for m = 1:ntau
                G.mat[m] = rand(ComplexF64, 2, 2)
            end
            Gref = deepcopy(G)
            n = ntime - 1
            zeros!(G, n + 1)
            extrapolate_timestep!(n, G, k)
            # Round-off scales with the magnitude of the data
            @test distance(G, Gref, n + 1) ≤ 1e-12 * distance(Gref, ℱ(C), n + 1)
            @test G.mat == Gref.mat
            for t = 1:n
                @test distance(G, Gref, t) == 0.0
            end
        end
    end

    @testset "extrapolate: preconditions" begin
        C = Cn(8, 5, 1, 1, 0.7, 1.0)
        G = ℱ(C)
        @test_throws AssertionError extrapolate_timestep!(3, G, 3)
        @test_throws AssertionError extrapolate_timestep!(8, G, 1)
        f = Cf(8, 1)
        @test_throws AssertionError extrapolate_timestep!(2, f, 2)
        @test_throws AssertionError extrapolate_timestep!(8, f, 0)
        @test_throws AssertionError extrapolate_timestep!(5, f, 6)
    end

    @testset "interpolation: polynomial exactness" begin
        h = 0.1
        for k = 1:5
            for dims in ((1, 1), (2, 2))
                Ms = [rand(ComplexF64, dims...) for _ = 0:k]
                P(t) = sum(Ms[r+1] .* t^r for r = 0:k)
                ntime = k + 3
                f = Cf(ntime, dims...)
                for i = 1:ntime
                    f[i] = P((i - 1) * h)
                end
                tstp = ntime
                for tinter in (tstp, tstp - k, tstp - 0.5, tstp - k + 0.3)
                    x = interpolation(tstp, tinter, f, k)
                    @test isapprox(x, P((tinter - 1) * h); atol = 1e-13, rtol = 1e-12)
                end
                @test_throws AssertionError interpolation(tstp, tstp + 0.1, f, k)
                @test_throws AssertionError interpolation(tstp, tstp - k - 0.1, f, k)
            end
        end
        f = Cf(4, 2)
        f[3] = rand(ComplexF64, 2, 2)
        @test interpolation(3, 3, f, 0) == f[3]
        @test_throws AssertionError interpolation(3, 2.5, f, 0)
        @test_throws AssertionError interpolation(2, 2, f, 2)
    end

    @testset "distance: block norms" begin
        C = Cn(6, 5, 2, 2, 0.5, 1.0)
        A = ℱ(C)
        B = ℱ(C)
        tstp = 4
        # Single block difference 3 + 4im -> 5
        A.ret[tstp,2] = C64[3+4im 0; 0 0]
        @test distance(A, B, tstp, Val(:ret)) == 5.0
        @test distance(A[tstp], B, tstp, Val(:ret)) == 5.0
        @test distance(A, B[tstp], tstp, Val(:ret)) == 5.0
        @test distance(A[tstp], B[tstp], tstp, Val(:ret)) == 5.0
        @test distance(A, B, tstp, Val(:tv)) == 0.0
        @test distance(A, B, tstp, Val(:les)) == 0.0
        # Elements with vanishing sum but non-zero norm
        A.lmix[tstp,3] = C64[1 -1; 0 0]
        A.less[2,tstp] = C64[0 2; 0 0]
        @test distance(A, B, tstp, Val(:tv)) ≈ sqrt(2.0)
        @test distance(A, B, tstp, Val(:les)) == 2.0
        for (a, b) in ((A, B), (A[tstp], B), (A, B[tstp]), (A[tstp], B[tstp]))
            @test distance(a, b, tstp) ≈ 5.0 + sqrt(2.0) + 2.0
            @test distance(a, b, tstp) ≈ distance(a, b, tstp, Val(:ret)) +
                                         distance(a, b, tstp, Val(:tv)) +
                                         distance(a, b, tstp, Val(:les))
        end
        # Matsubara branch
        A.mat[2] = C64[1 -1; 0 0]
        @test distance(A, B, 0) ≈ sqrt(2.0)
        @test distance(A[0], B, 0) ≈ sqrt(2.0)
        @test distance(A, B, 0, Val(:ret)) == 0.0
        @test distance(A, B, 0, Val(:tv)) == 0.0
        @test distance(A, B, 0, Val(:les)) == 0.0
        # Mismatched time step of 𝒻
        @test_throws AssertionError distance(A[3], B, tstp)
    end

    @testset "set_t0_from_mat! and set_tk_from_mat!" begin
        ntime = 9
        ntau = 41
        beta = 5.0
        h = 0.05
        H = C64[-1.0 -0.7im; 0.7im 0.8]
        for sign in (FERMI, BOSE)
            C = Cn(ntime, ntau, 2, 2, (ntime - 1) * h, beta)
            Gref = ℱ(C, sign)
            init_green!(Gref, sign == FERMI ? H : H .+ 2.0, 0.0, beta, h)
            #
            G = ℱ(C, sign)
            memcpy!(Gref, G, 0)
            set_t0_from_mat!(G)
            @test distance(G, Gref, 1) ≤ 1e-13
            @test G.mat == Gref.mat
            #
            k = 4
            G = ℱ(C, sign)
            memcpy!(Gref, G, 0)
            sentinel = C64[9 8; 7 6]
            for n = 2:k+1
                G.ret[n,1] = sentinel
                G.less[1,n] = sentinel
            end
            set_tk_from_mat!(G, k)
            for n = 1:k+1
                for m = 1:ntau
                    @test G.lmix[n,m] ≈ Gref.lmix[1,m] atol = 1e-13
                end
                @test G.less[n,n] ≈ Gref.less[1,1] atol = 1e-13
                @test G.ret[n,n] ≈ Gref.ret[1,1] atol = 1e-13
            end
            for n = 2:k+1
                @test G.ret[n,1] == sentinel
                @test G.less[1,n] == sentinel
            end
            @test distance(G, ℱ(C, sign), k + 2) == 0.0
            @test_throws AssertionError set_tk_from_mat!(G, ntime)
        end
    end

    @testset "force_matsubara_hermitian!" begin
        C = Cn(5, 6, 2, 2, 0.4, 1.0)
        G = ℱ(C)
        for m = 1:6
            G.mat[m] = rand(ComplexF64, 2, 2)
        end
        for i = 1:5, j = 1:i
            G.ret[i,j] = rand(ComplexF64, 2, 2)
        end
        Gref = deepcopy(G)
        force_matsubara_hermitian!(G)
        for m = 1:6
            @test norm(G.mat[m] - G.mat[m]') ≤ 1e-14
            @test G.mat[m] ≈ (Gref.mat[m] + Gref.mat[m]') / 2
        end
        for t = 1:5
            @test distance(G, Gref, t) == 0.0
        end
        #
        g = Gref[0]
        force_matsubara_hermitian!(g)
        for m = 1:6
            @test norm(g.mat[m] - g.mat[m]') ≤ 1e-14
            @test g.mat[m] ≈ G.mat[m]
        end
        #
        Cr = Cn(5, 6, 2, 3, 0.4, 1.0)
        @test_throws AssertionError force_matsubara_hermitian!(ℱ(Cr))
    end

    #
    # Port of libcntr/test/utilities.cpp:213-488
    #
    @testset "NESSi utilities.cpp" begin
        nt = 20
        ntau = 100
        kt = 5
        beta = 5.0
        h = 0.05
        eps = 1.0e-6
        ntime = nt + 1
        ntauk = ntau + 1
        tmax = nt * h
        w1, w2, w3 = -1.0, 0.8, 0.7
        nu1, nu2, nu3 = -2.0, 1.6, 1.4
        ham1 = C64[w1 -im*w3; im*w3 w2]
        ham2 = C64[nu1 -im*nu3; im*nu3 nu2]

        @testset "extrapolate timestep" begin
            C = Cn(ntime, ntauk, 1, 1, tmax, beta)
            G1 = ℱ(C, FERMI)
            init_green!(G1, fill(C64(1.0), 1, 1), 0.0, beta, h)
            G2 = deepcopy(G1)
            G3 = deepcopy(G1)
            p = kt
            wl = ut_lagrange_weights(p, h, (p + 1) * h)
            err = 0.0
            for tstp = kt:nt-1
                n = tstp + 1
                memcpy!(G1, G2, n)
                memcpy!(G1, G3, n)
                extrapolate_timestep!(n, G2, Integrator(kt))
                # tv
                for m = 1:ntauk
                    x = zeros(C64, 1, 1)
                    for q = 0:p
                        x .+= wl[p-q+1] .* G3.lmix[n-q,m]
                    end
                    G3.lmix[n+1,m] = x
                end
                # les
                for j = 1:n
                    x = zeros(C64, 1, 1)
                    for q = 0:p
                        x .+= wl[p-q+1] .* G3.less[j,n-q]
                    end
                    G3.less[j,n+1] = x
                end
                x = zeros(C64, 1, 1)
                for q = 0:p
                    x .+= wl[p-q+1] .* G3.less[n-q,n+1]
                end
                G3.less[n+1,n+1] = x
                # ret
                for j = 1:n
                    x = zeros(C64, 1, 1)
                    for q = 0:p
                        x .+= wl[p-q+1] .* G3.ret[n-q,j]
                    end
                    G3.ret[n+1,j] = x
                end
                err += distance(G2, G3, n + 1, Val(:les))
            end
            @test err < eps
        end

        @testset "Extrapolate function" begin
            fv(t) = C64[cos(w1*t) -im*0.2*sin(w3*t); im*0.2*sin(w3*t) cos(w2*t)]
            func1 = Cf(ntime, 2)
            func2 = Cf(ntime, 2)
            func3 = Cf(ntime, 2)
            for tstp = 0:nt
                for f in (func1, func2, func3)
                    f[tstp+1] = fv(tstp * h)
                end
            end
            p = kt
            wl = ut_lagrange_weights(p, h, (p + 1) * h)
            err = 0.0
            for tstp = kt:nt-1
                n = tstp + 1
                extrapolate_timestep!(n, func1, Integrator(kt))
                fref = sum(wl[p-q+1] .* func3[n-q] for q = 0:p)
                err += norm(func1[n+1] - fref)
                func1[n+1] = func2[n+1]
                func3[n+1] = func2[n+1]
            end
            @test err < eps
        end

        @testset "Interpolate function" begin
            fv(t) = C64[cos(w1*t) -im*0.2*sin(w3*t); im*0.2*sin(w3*t) cos(w2*t)]
            func1 = Cf(ntime, 2)
            for tstp = 0:nt
                func1[tstp+1] = fv(tstp * h)
            end
            p = kt
            err = 0.0
            for tstp = kt:nt-1
                tinter = tstp + 0.5
                f = interpolation(tstp + 2, tinter + 1, func1, Integrator(kt))
                # interpolate_function(tstp+1, kt, h, h*tinter, func1)
                ts = tstp + 1
                wl = ut_lagrange_weights(p, h, h * tinter - h * (ts - p))
                fref = sum(wl[p-q+1] .* func1[ts-q+1] for q = 0:p)
                err += norm(f - fref)
            end
            @test err < eps
        end

        @testset "set_t0_from_mat" begin
            C = Cn(ntime, ntauk, 2, 2, tmax, beta)
            G1 = ℱ(C, FERMI)
            G2 = ℱ(C, FERMI)
            init_green!(G1, ham1, 0.0, beta, h)
            memcpy!(G1, G2, 0)
            set_t0_from_mat!(G2)
            @test distance(G1, G2, 1) < eps
        end

        @testset "set_tk_from_mat" begin
            C = Cn(ntime, ntauk, 2, 2, tmax, beta)
            G1 = ℱ(C, FERMI)
            G2 = ℱ(C, FERMI)
            init_green!(G1, ham1, 0.0, beta, h)
            memcpy!(G1, G2, 0)
            set_tk_from_mat!(G2, kt)
            err = 0.0
            for n = 1:kt+1
                for m = 1:ntauk
                    err += norm(G1.lmix[1,m] - G2.lmix[n,m])
                end
                err += norm(G1.less[n,n] - G2.less[n,n])
                err += norm(G1.ret[n,n] - G2.ret[n,n])
            end
            @test err < eps
        end

        @testset "distance_norm2 (herm_matrix and timestep)" begin
            C = Cn(ntime, ntauk, 2, 2, tmax, beta)
            G1 = ℱ(C, FERMI)
            G2 = ℱ(C, FERMI)
            init_green!(G1, ham1, 0.0, beta, h)
            init_green!(G2, ham2, 0.0, beta, h)
            # Port of dist_green() and dist_green_step()
            function dist_green(tstp, a, b)
                d = 0.0
                if tstp == 0
                    for m = 1:ntauk
                        d += norm(a.mat[m] - b.mat[m])
                    end
                else
                    for m = 1:ntauk
                        d += norm(a.lmix[tstp,m] - b.lmix[tstp,m])
                    end
                    for j = 1:tstp
                        d += norm(a.less[j,tstp] - b.less[j,tstp])
                        d += norm(a.ret[tstp,j] - b.ret[tstp,j])
                    end
                end
                return d
            end
            err = 0.0
            errs = 0.0
            for tstp = 0:ntime
                err += abs(distance(G1, G2, tstp) - dist_green(tstp, G1, G2))
                errs += abs(distance(G1[tstp], G2[tstp], tstp) - dist_green(tstp, G1, G2))
            end
            @test err < eps
            @test errs < eps
        end

        @testset "force_matsubara_hermitian" begin
            ham3 = C64[w2 w3; w3 w1]
            C = Cn(ntime, ntauk, 2, 2, tmax, beta)
            G1 = ℱ(C, FERMI)
            G2 = ℱ(C, FERMI)
            init_green!(G1, ham1, 0.0, beta, h)
            init_green!(G2, ham3, 0.0, beta, h)
            incr!(G1, G2, 0, im)
            force_matsubara_hermitian!(G1)
            err = 0.0
            for m = 1:ntauk
                err += norm(G1.mat[m] - G1.mat[m]')
            end
            @test err < eps
        end
    end

    #
    # Port of libcntr/test/function.cpp:97-227 (non-HDF5 sections)
    #
    @testset "NESSi function.cpp" begin
        nt = 50
        size1 = 2
        size2 = 5
        eps = 1e-7
        eps2 = 1e-3
        h = 0.01
        kt = 5
        ntime = nt + 1
        A = Cf(ntime, size1)

        @testset "Set/get for function" begin
            function setget(F, a)
                toterr = 0.0
                # NESSi m = 0 .. nt-1 -> KB m+1
                for m = 0:nt-1
                    F[m+1] = a
                    toterr += norm(a - F[m+1])
                end
                return toterr
            end
            a = C64[sqrt(2.0) sqrt(2.0)*im; -sqrt(2.0)*im -sqrt(2.0)]
            b = C64[i + 2j for i = 0:size1-1, j = 0:size2-1]
            B = Cf(ntime, size1, size2)
            @test setget(A, a) < eps
            @test setget(B, b) < eps
        end

        cv(t) = C64[2cos(t) 0.5cos(t); 0.5cos(t) 3cos(t)]
        # NESSi tstp = -1 is the Matsubara value (t = 0)
        A[0] = cv(0.0)
        for tstp = 0:nt
            A[tstp+1] = cv(tstp * h)
        end

        @testset "Extrapolation" begin
            B = Cf(ntime, size1)
            toterr = 0.0
            for tstp = 0:nt-1
                n1 = (0 ≤ tstp ≤ kt) ? tstp : kt
                for m = -1:tstp
                    B[m+1] = A[m+1]
                end
                extrapolate_timestep!(tstp + 1, B, n1)
                toterr += norm(A[tstp+2] - B[tstp+2])
            end
            @test toterr < eps2
        end

        @testset "Interpolation" begin
            B = Cf(ntime, size1)
            toterr0 = 0.0
            toterr1 = 0.0
            for tstp = 1:nt-1
                n1 = (0 ≤ tstp ≤ kt) ? tstp : kt
                for m = -1:tstp
                    B[m+1] = A[m+1]
                end
                b0 = interpolation(tstp + 1, tstp + 1, B, n1)
                a0 = A[tstp+1]
                b1 = interpolation(tstp + 1, tstp + 1 - 0.5, B, n1)
                a1 = cv((tstp - 0.5) * h)
                toterr0 += norm(a0 - b0)
                toterr1 += norm(a1 - b1)
            end
            @test toterr0 < eps
            @test toterr1 < eps2
        end
    end

    #
    # Release only: convergence order gates and reduced-order mutants.
    #
    if kb_release()
        @testset "release: extrapolation / interpolation order gate" begin
            @test kb_test_profile() == "release"
            for k = 1:5
                Ns = [2^m for m in UT_GRIDS[k]]
                # Formal method: order k + 1
                errs = [ut_extrap_error(k, N, k) for N in Ns]
                test_order_gate(Ns, errs, k + 1, 1e-8)
                errs = [ut_interp_error(k, N, k) for N in Ns]
                test_order_gate(Ns, errs, k + 1, 1e-8)
                # Mutation fixture: one order lower must be rejected
                errs = [ut_extrap_error(k, N, k - 1) for N in Ns]
                @test !check_order_gate(Ns, errs, k + 1, 1e-8)[1]
                if k ≥ 2
                    errs = [ut_interp_error(k, N, k - 1) for N in Ns]
                    @test !check_order_gate(Ns, errs, k + 1, 1e-8)[1]
                end
            end
        end
    end
end
