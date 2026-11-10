#
# t_differentiation.jl
#
# To test the differentiation of contour-ordered Green's functions. There
# is no Catch test for differentiation in NESSi 2.0.0, so the functions
# are verified with analytic polynomials and exponentials.
#

isdefined(Main, :kb_test_profile) ||
    include(joinpath(@__DIR__, "..", "common", "profile.jl"))
isdefined(Main, :test_order_gate) ||
    include(joinpath(@__DIR__, "..", "common", "order_gate.jl"))

using LinearAlgebra

#
# A generic (non-hermitian) smooth two-time function F(t,s) and its
# partial derivatives are stored in `A` and `Acc`, such that the
# reconstructed elements of `A` are given by `Acc` via the hermitian
# conjugate relations.
#
function df_fill!(A::ℱ, Acc::ℱ, h, Fr, Fl, Fm, Fx)
    ntime = getntime(A)
    ntau = getntau(A)
    for i = 1:ntime, j = 1:i
        A.ret[i,j] = Fr((i - 1) * h, (j - 1) * h)
        Acc.ret[i,j] = -Matrix(Fr((j - 1) * h, (i - 1) * h)')
    end
    for j = 1:ntime, i = 1:j
        A.less[i,j] = Fl((i - 1) * h, (j - 1) * h)
        Acc.less[i,j] = -Matrix(Fl((j - 1) * h, (i - 1) * h)')
    end
    for i = 1:ntime, m = 1:ntau
        A.lmix[i,m] = Fm((i - 1) * h, m)
        Acc.lmix[i,m] = A.lmix[i,m]
    end
    for m = 1:ntau
        A.mat[m] = Fx(m)
        Acc.mat[m] = A.mat[m]
    end
end

#
# Polynomial of degree `deg` in two variables with random matrix
# coefficients, together with its partial derivatives.
#
function df_poly2(deg, dims)
    c = [rand(ComplexF64, dims...) for _ = 0:deg, _ = 0:deg]
    F(t, s)  = sum(c[a+1,b+1] .* (t^a * s^b) for a = 0:deg, b = 0:deg if a + b ≤ deg)
    D1(t, s) = sum(c[a+1,b+1] .* (a * t^(a-1) * s^b) for a = 1:deg, b = 0:deg if a + b ≤ deg; init = zeros(ComplexF64, dims...))
    D2(t, s) = sum(c[a+1,b+1] .* (b * t^a * s^(b-1)) for a = 0:deg, b = 1:deg if a + b ≤ deg; init = zeros(ComplexF64, dims...))
    return F, D1, D2
end

@testset verbose = true "KadanoffBaym: differentiation.jl" begin

    @testset "polynomial exactness" begin
        for k = 1:5
            for dims in ((1, 1), (2, 2))
                ntime = 2 * k + 5
                ntau = 2 * k + 4
                h = 0.1
                beta = 0.7
                hv = beta / (ntau - 1)
                C = Cn(ntime, ntau, dims..., (ntime - 1) * h, beta)
                A = ℱ(C)
                Acc = ℱ(C)
                Fr, Dr1, Dr2 = df_poly2(k, dims)
                Fl, Dl1, Dl2 = df_poly2(k, dims)
                Fm, Dm1, Dm2 = df_poly2(k, dims)
                Fx, Dx1, _ = df_poly2(k + 1, dims)
                df_fill!(A, Acc, h,
                         Fr, Fl,
                         (t, m) -> Fm(t, (m - 1) * hv),
                         m -> Fx((m - 1) * hv, 0.0))
                #
                # Left derivative, all time steps
                dA = ℱ(C)
                for tstp = 0:ntime
                    deriv1_timestep!(tstp, dA, A, Acc, Integrator(k), beta, h)
                end
                err = 0.0
                for i = 1:ntime, j = 1:i
                    err = max(err, maximum(abs.(dA.ret[i,j] - im * Dr1((i-1)*h, (j-1)*h))))
                end
                for j = 1:ntime, i = 1:j
                    err = max(err, maximum(abs.(dA.less[i,j] - im * Dl1((i-1)*h, (j-1)*h))))
                end
                for i = 1:ntime, m = 1:ntau
                    err = max(err, maximum(abs.(dA.lmix[i,m] - im * Dm1((i-1)*h, (m-1)*hv))))
                end
                for m = 1:ntau
                    err = max(err, maximum(abs.(dA.mat[m] + Dx1((m-1)*hv, 0.0))))
                end
                @test err ≤ 1e-10
                #
                # Right derivative, all time steps
                dB = ℱ(C)
                for tstp = 0:ntime
                    deriv2_timestep!(tstp, dB, A, Acc, Integrator(k), beta, h)
                end
                err = 0.0
                for i = 1:ntime, j = 1:i
                    err = max(err, maximum(abs.(dB.ret[i,j] + im * Dr2((i-1)*h, (j-1)*h))))
                end
                for j = 1:ntime, i = 1:j
                    err = max(err, maximum(abs.(dB.less[i,j] + im * Dl2((i-1)*h, (j-1)*h))))
                end
                for i = 1:ntime, m = 1:ntau
                    err = max(err, maximum(abs.(dB.lmix[i,m] - Dm2((i-1)*h, (m-1)*hv))))
                end
                @test err ≤ 1e-10
                @test dB.mat == dA.mat
            end
        end
    end

    @testset "hermitian GF: left and right derivatives" begin
        # G^R = -i U(t) U(t')†, ..., from init_green!. Then
        # i∂ₜ G = (H - μ) G and -i∂ₜ' G = G (H - μ).
        ntime = 40
        ntau = 30
        beta = 2.0
        h = 0.01
        μ = 0.1
        H = C64[-0.4 0.3im; -0.3im 0.6]
        Hμ = H - μ * I
        for sign in (FERMI, BOSE)
            C = Cn(ntime, ntau, 2, 2, (ntime - 1) * h, beta)
            G = ℱ(C, sign)
            init_green!(G, sign == FERMI ? H : H + 2.0 * I, μ, beta, h)
            Hs = sign == FERMI ? Hμ : Hμ + 2.0 * I
            k = 5
            dA = ℱ(C, sign)
            dB = ℱ(C, sign)
            for tstp = 1:ntime
                deriv1_timestep!(tstp, dA, G, beta, h, k)
                deriv2_timestep!(tstp, dB, G, beta, h, k)
            end
            err = 0.0
            for i = 1:ntime, j = 1:i
                err = max(err, maximum(abs.(dA.ret[i,j] - Hs * G.ret[i,j])))
                err = max(err, maximum(abs.(dB.ret[i,j] - G.ret[i,j] * Hs)))
            end
            for j = 1:ntime, i = 1:j
                err = max(err, maximum(abs.(dA.less[i,j] - Hs * G.less[i,j])))
                err = max(err, maximum(abs.(dB.less[i,j] - G.less[i,j] * Hs)))
            end
            for i = 1:ntime, m = 1:ntau
                err = max(err, maximum(abs.(dA.lmix[i,m] - Hs * G.lmix[i,m])))
            end
            # Discretization error of the 6th order BDF at h = 0.01
            @test err ≤ 1e-8
        end
    end

    @testset "timestep entry equals element calls" begin
        k = 3
        ntime = 10
        ntau = 9
        h = 0.1
        beta = 1.0
        C = Cn(ntime, ntau, 2, 2, (ntime - 1) * h, beta)
        A = ℱ(C)
        Acc = ℱ(C)
        Fr, _, _ = df_poly2(4, (2, 2))
        Fl, _, _ = df_poly2(4, (2, 2))
        Fm, _, _ = df_poly2(4, (2, 2))
        df_fill!(A, Acc, h, Fr, Fl, (t, m) -> Fm(t, 0.1m), m -> Fm(0.1m, 0.0))
        I = Integrator(k)
        for tstp in (2, 4, 7)
            sentinel = C64(3.0 - 1.0im)
            dA = ℱ(C, sentinel, FERMI)
            dE = ℱ(C, sentinel, FERMI)
            deriv1_timestep!(tstp, dA, A, Acc, I, beta, h)
            for j = 1:ntime
                deriv1_element!(tstp, j, dE, A, Acc, I, h)
            end
            deriv1_lmix!(tstp, dE, A, I, h)
            @test dA == dE
            # Only the target time step is touched
            for i = 1:ntime, j = 1:i
                i == tstp && continue
                @test all(dA.ret[i,j] .== sentinel)
            end
            for j = 1:ntime, i = 1:j
                i == tstp && continue
                @test all(dA.less[i,j] .== sentinel)
            end
            for i = 1:ntime, m = 1:ntau
                i == tstp && continue
                @test all(dA.lmix[i,m] .== sentinel)
            end
            @test all(dA.mat.data .== sentinel)
            #
            dB = ℱ(C, sentinel, FERMI)
            dE = ℱ(C, sentinel, FERMI)
            deriv2_timestep!(tstp, dB, A, Acc, I, beta, h)
            for j = 1:ntime
                deriv2_element!(tstp, j, dE, A, Acc, I, h)
            end
            deriv2_lmix!(tstp, dE, A, I, beta)
            @test dB == dE
        end
    end

    @testset "Matsubara grid lower bound" begin
        for k = 1:5
            I = Integrator(k)
            # ntau = 2k + 2 is the minimal valid grid
            C = Cn(k + 2, 2 * k + 2, 1, 1, 1.0, 1.0)
            A = ℱ(C)
            for m = 1:2*k+2
                A.mat[m] = fill(C64(m^2), 1, 1)
            end
            dA = ℱ(C)
            deriv1_matsubara!(dA, A, I, 1.0)
            @test all(isfinite, dA.mat.data)
            # One point less must be rejected
            C = Cn(k + 2, 2 * k + 1, 1, 1, 1.0, 1.0)
            @test_throws AssertionError deriv1_matsubara!(ℱ(C), ℱ(C), I, 1.0)
        end
        # Other preconditions
        C = Cn(6, 6, 1, 1, 1.0, 1.0)
        A = ℱ(C)
        @test_throws AssertionError deriv1_element!(0, 1, ℱ(C), A, A, Integrator(2), 0.1)
        @test_throws AssertionError deriv1_lmix!(7, ℱ(C), A, Integrator(2), 0.1)
        @test_throws AssertionError deriv1_timestep!(1, A, A, 1.0, 0.1, 2)
        @test_throws AssertionError deriv1_timestep!(1, ℱ(C), A, 1.0, 0.1, 6)
        # The output must not be the input Acc either
        Acc = deepcopy(A)
        @test_throws AssertionError deriv1_element!(2, 1, Acc, A, Acc, Integrator(2), 0.1)
        @test_throws AssertionError deriv2_element!(2, 1, Acc, A, Acc, Integrator(2), 0.1)
        @test_throws AssertionError deriv1_timestep!(2, Acc, A, Acc, 1.0, 0.1, 2)
        @test_throws AssertionError deriv2_timestep!(2, Acc, A, Acc, 1.0, 0.1, 2)
    end

    #
    # Release only: convergence order gates and reduced-order mutants.
    #
    # The local derivative error at a fixed grid index is measured on small
    # grids with step h -> 0. In the start-up region the polynomial
    # differentiation of order k is used (error ~ h^k); otherwise BDF of
    # order k + 1 is used (error ~ h^(k+1)).
    #
    if kb_release()
        @testset "release: differentiation order gate" begin
            @test kb_test_profile() == "release"
            H = C64[-0.4 0.3im; -0.3im 0.6]
            X = C64[0.3 0.1; 0.1 0.2]
            λ = 0.8
            U(t) = exp(-im * H * t)
            Fr(t, s) = -im * U(t + 0.3) * U(s + 0.3)'
            Fl(t, s) = im * U(t + 0.3) * X * U(s + 0.3)'
            Z = C64[0.5 0.2im; -0.2im 0.4]
            #
            # Error of the left/right derivatives at grid index i.
            function deriv_error(k, ord, h, i)
                ntime = max(i, k + 1) + 1
                ntau = 2 * k + 2
                C = Cn(ntime, ntau, 2, 2, (ntime - 1) * h, 1.0)
                A = ℱ(C)
                Acc = ℱ(C)
                df_fill!(A, Acc, h, Fr, Fl, (t, m) -> U(t) * Z, m -> Z)
                dA = ℱ(C)
                dB = ℱ(C)
                J = Integrator(ord)
                err = 0.0
                for j = 1:i
                    deriv1_element!(i, j, dA, A, Acc, J, h)
                    deriv2_element!(j, i, dB, A, Acc, J, h)
                    t = (i - 1) * h
                    s = (j - 1) * h
                    err = max(err, maximum(abs.(dA.ret[i,j] - H * Fr(t, s))))
                    err = max(err, maximum(abs.(dB.less[j,i] - Fl(s, t) * H)))
                end
                return err
            end
            #
            # Error of -∂τ A^M at the last imaginary time point.
            function mat_error(k, ord, hv)
                ntau = 2 * k + 3
                beta = hv * (ntau - 1)
                C = Cn(2, ntau, 2, 2, 1.0, beta)
                A = ℱ(C)
                for m = 1:ntau
                    A.mat[m] = Z .* exp(-λ * (m - 1) * hv)
                end
                dA = ℱ(C)
                deriv1_matsubara!(dA, A, Integrator(ord), beta)
                return maximum(abs.(dA.mat[1] - λ * Z))
            end
            #
            #
            # Grids N = 2^m, chosen per order and region so that the errors
            # stay in the asymptotic regime (above the round-off floor of
            # the finite differences) and the finest error is below 1e-8.
            grids = Dict(
                (1, :b) => 11:14, (1, :m) => 11:14,
                (2, :s) => 11:14, (2, :b) => 6:9, (2, :m) => 6:9,
                (3, :s) => 6:9,   (3, :b) => 4:7, (3, :m) => 4:7,
                (4, :s) => 4:7,   (4, :b) => 3:6, (4, :m) => 3:6,
                (5, :s) => 3:6,   (5, :b) => 2:5, (5, :m) => 2:5,
            )
            for k = 1:5
                if k ≥ 2
                    # Start-up region (i = 1): order k
                    Ns = [2^m for m in grids[(k, :s)]]
                    errs = [deriv_error(k, k, 1.0 / N, 1) for N in Ns]
                    test_order_gate(Ns, errs, k, 1e-8)
                    errs = [deriv_error(k, k - 1, 1.0 / N, 1) for N in Ns]
                    @test !check_order_gate(Ns, errs, k, 1e-8)[1]
                end
                # Stable region (i = k + 2): order k + 1
                Ns = [2^m for m in grids[(k, :b)]]
                errs = [deriv_error(k, k, 1.0 / N, k + 2) for N in Ns]
                test_order_gate(Ns, errs, k + 1, 1e-8)
                if k ≥ 2
                    errs = [deriv_error(k, k - 1, 1.0 / N, k + 2) for N in Ns]
                    @test !check_order_gate(Ns, errs, k + 1, 1e-8)[1]
                end
                # Matsubara: BDF of order k + 1
                Ns = [2^m for m in grids[(k, :m)]]
                errs = [mat_error(k, k, 1.0 / N) for N in Ns]
                test_order_gate(Ns, errs, k + 1, 1e-8)
                if k ≥ 2
                    errs = [mat_error(k, k - 1, 1.0 / N) for N in Ns]
                    @test !check_order_gate(Ns, errs, k + 1, 1e-8)[1]
                end
            end
        end
    end
end
