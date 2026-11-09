#
# t_convolution_utilities.jl
#
# To test the incremental convolutions, the partial lesser convolutions,
# the equal-time lesser convolution, and the density matrix convolution.
#
# NESSi Catch tests ported here:
#   libcntr/test/convolution.cpp:848-885 (time-diagonal convolution)
#

isdefined(Main, :kb_test_profile) ||
    include(joinpath(@__DIR__, "..", "common", "profile.jl"))
isdefined(Main, :test_order_gate) ||
    include(joinpath(@__DIR__, "..", "common", "order_gate.jl"))

using LinearAlgebra

#
# Smooth two-time function with random matrix coefficients.
#
function cu_poly(dims, deg = 2)
    c = [rand(ComplexF64, dims...) .- (0.5 + 0.5im) for _ = 0:deg, _ = 0:deg]
    return (t, s) -> sum(c[a+1,b+1] .* (t^a * s^b) for a = 0:deg, b = 0:deg if a + b ≤ deg)
end

#
# Build a general (non-hermitian) pair (A, Acc). The stored elements of
# `Acc` are chosen such that the reconstructed elements of `A` from `Acc`
# are smooth continuations of the stored elements of `A`.
#
function cu_make(C::Cn, sign, h, hv)
    dims = getdims(C)
    A = ℱ(C, sign)
    Acc = ℱ(C, sign)
    Fr = cu_poly(dims)
    Fl = cu_poly(dims)
    Fm = cu_poly(dims)
    Fc = cu_poly(dims)
    Fx = cu_poly(dims)
    ntime = getntime(C)
    ntau = getntau(C)
    for i = 1:ntime, j = 1:i
        A.ret[i,j] = Fr((i - 1) * h, (j - 1) * h)
        Acc.ret[i,j] = -Matrix(Fr((j - 1) * h, (i - 1) * h)')
    end
    for j = 1:ntime, i = 1:j
        A.less[i,j] = Fl((i - 1) * h, (j - 1) * h)
        Acc.less[i,j] = -Matrix(Fl((j - 1) * h, (i - 1) * h)')
    end
    for i = 1:ntime, m = 1:ntau
        A.lmix[i,m] = Fm((i - 1) * h, (m - 1) * hv)
        Acc.lmix[i,m] = Fc((i - 1) * h, (m - 1) * hv)
    end
    for m = 1:ntau
        A.mat[m] = Fx((m - 1) * hv, 0.0)
        Acc.mat[m] = A.mat[m]
    end
    return A, Acc
end

#
# Random contour-ordered Green's function used as initial value of the
# target.
#
function cu_random(C::Cn, sign)
    G = ℱ(C, sign)
    dims = getdims(C)
    for m = 1:getntau(C)
        G.mat[m] = rand(ComplexF64, dims...)
    end
    for i = 1:getntime(C)
        for j = 1:i
            G.ret[i,j] = rand(ComplexF64, dims...)
        end
        for m = 1:getntau(C)
            G.lmix[i,m] = rand(ComplexF64, dims...)
        end
        for j = i:getntime(C)
            G.less[i,j] = rand(ComplexF64, dims...)
        end
    end
    return G
end

#
# Contour function f(t) with a non-trivial Matsubara value.
#
function cu_func(C::Cn, h)
    dims = getdims(C)
    M0 = rand(ComplexF64, dims...)
    M1 = rand(ComplexF64, dims...)
    f = Cf(getntime(C), dims...)
    for i = 1:getntime(C)
        f[i] = M0 .+ M1 .* sin(1.3 * (i - 1) * h)
    end
    f[0] = M0 .+ 0.2 .* M1
    return f
end

#
# Relative tolerance with respect to the magnitude of the reference.
#
cu_tol(G::ℱ, tstp) = 1e-12 * max(1.0, distance(G, ℱ(Cn(getntime(G), getntau(G),
    getdims(G)..., 1.0, 1.0), getsign(G)), tstp))

@testset verbose = true "KadanoffBaym: convolution utilities" begin

    ntime = 9
    ntau = 13
    beta = 2.0
    h = 0.05
    hv = beta / (ntau - 1)
    korders = kb_release() ? (1:5) : (1, 3, 5)

    @testset "incr_convolution! = C₀ + α (A ∗ f ∗ B)" begin
        for sign in (FERMI, BOSE), dims in ((1, 1), (2, 2)), k in korders
            C = Cn(ntime, ntau, dims..., (ntime - 1) * h, beta)
            I = Integrator(k)
            A, Acc = cu_make(C, sign, h, hv)
            B, Bcc = cu_make(C, sign, h, hv)
            f = cu_func(C, h)
            C₀ = cu_random(C, sign)
            for (fx, α) in ((nothing, 1.0), (f, 0.3 - 0.7im), (f, 0.0))
                # Reference: full convolution
                Cref = ℱ(C, sign)
                if fx isa Cf
                    convolution(Cref, A, Acc, fx, B, Bcc, I, beta, h)
                else
                    convolution(Cref, A, Acc, B, Bcc, I, beta, h)
                end
                Exp = deepcopy(C₀)
                incr!(Exp, Cref, α)
                #
                # Full target
                X = deepcopy(C₀)
                for tstp = 0:ntime
                    incr_convolution!(tstp, X, A, Acc, fx, B, Bcc, I, beta, h, α)
                end
                for tstp = 0:ntime
                    @test distance(X, Exp, tstp) ≤ cu_tol(Exp, tstp)
                end
                if α == 0.0
                    @test X == C₀
                end
                #
                # Time slice target, only the given time step is touched
                for tstp = 0:ntime
                    x = C₀[tstp]
                    incr_convolution!(tstp, x, A, Acc, fx, B, Bcc, I, beta, h, α)
                    @test distance(x, Exp, tstp) ≤ cu_tol(Exp, tstp)
                    Y = deepcopy(C₀)
                    incr_convolution!(tstp, Y, A, Acc, fx, B, Bcc, I, beta, h, α)
                    for t = 0:ntime
                        t == tstp && continue
                        @test distance(Y, C₀, t) == 0.0
                    end
                end
            end
        end
    end

    @testset "component entries" begin
        for sign in (FERMI, BOSE), k in korders
            C = Cn(ntime, ntau, 2, 2, (ntime - 1) * h, beta)
            I = Integrator(k)
            A, Acc = cu_make(C, sign, h, hv)
            B, Bcc = cu_make(C, sign, h, hv)
            f = cu_func(C, h)
            α = 0.6 + 0.2im
            Cref = ℱ(C, sign)
            convolution(Cref, A, Acc, f, B, Bcc, I, beta, h)
            C₀ = cu_random(C, sign)
            Exp = deepcopy(C₀)
            incr!(Exp, Cref, α)
            #
            X = deepcopy(C₀)
            incr_convolution_mat!(X, A, f, B, I, beta, α)
            @test distance(X, Exp, 0) ≤ cu_tol(Exp, 0)
            for tstp = 1:ntime
                X = deepcopy(C₀)
                incr_convolution_ret!(tstp, X, A, Acc, f, B, Bcc, I, h, α)
                @test distance(X, Exp, tstp, Val(:ret)) ≤ cu_tol(Exp, tstp)
                @test distance(X, C₀, tstp, Val(:tv)) == 0.0
                @test distance(X, C₀, tstp, Val(:les)) == 0.0
                #
                X = deepcopy(C₀)
                incr_convolution_tv!(tstp, X, A, Acc, f, B, Bcc, I, beta, h, α)
                @test distance(X, Exp, tstp, Val(:tv)) ≤ cu_tol(Exp, tstp)
                @test distance(X, C₀, tstp, Val(:ret)) == 0.0
                @test distance(X, C₀, tstp, Val(:les)) == 0.0
                #
                X = deepcopy(C₀)
                incr_convolution_les!(tstp, X, A, Acc, f, B, Bcc, I, beta, h, α)
                @test distance(X, Exp, tstp, Val(:les)) ≤ cu_tol(Exp, tstp)
                @test distance(X, C₀, tstp, Val(:ret)) == 0.0
                @test distance(X, C₀, tstp, Val(:tv)) == 0.0
                #
                # The lesser entry is exactly the sum of the three partials
                Y = deepcopy(C₀)
                incr_convolution_ret_less!(tstp, Y, A, Acc, f, B, Bcc, I, beta, h, α)
                incr_convolution_less_adv!(tstp, Y, A, Acc, f, B, Bcc, I, beta, h, α)
                incr_convolution_lmix_rmix!(tstp, Y, A, Acc, f, B, Bcc, I, beta, h, α)
                @test Y == X
            end
        end
    end

    @testset "partial lesser: target is not an input" begin
        for sign in (FERMI, BOSE), k in korders
            C = Cn(ntime, ntau, 2, 2, (ntime - 1) * h, beta)
            I = Integrator(k)
            A, Acc = cu_make(C, sign, h, hv)
            B, Bcc = cu_make(C, sign, h, hv)
            f = cu_func(C, h)
            for tstp = 1:ntime
                for pfun in (incr_convolution_ret_less!,
                             incr_convolution_less_adv!,
                             incr_convolution_lmix_rmix!)
                    # The increment does not depend on the old target
                    X₁ = ℱ(C, sign)
                    X₂ = ℱ(C, C64(1e6), sign)
                    pfun(tstp, X₁, A, Acc, f, B, Bcc, I, beta, h, 1.0)
                    pfun(tstp, X₂, A, Acc, f, B, Bcc, I, beta, h, 1.0)
                    for j = 1:tstp
                        @test X₂.less[j,tstp] - X₁.less[j,tstp] ≈ fill(C64(1e6), 2, 2) atol = 1e-6
                        @test X₁.less[j,tstp] ≈ X₂.less[j,tstp] .- 1e6 rtol = 1e-9 atol = 1e-8
                    end
                end
                #
                # The source of the Dyson/VIE2 time stepping (less_adv +
                # lmix_rmix) does not read the implicit unknown G^<(:,tstp).
                # Here G is the right operand B.
                srcs = map((0.0, 1e6)) do v
                    G = deepcopy(B)
                    Gcc = deepcopy(Bcc)
                    for j = 1:tstp
                        G.less[j,tstp] = fill(C64(v), 2, 2)
                        Gcc.less[j,tstp] = fill(C64(v), 2, 2)
                    end
                    X = ℱ(C, sign)
                    incr_convolution_less_adv!(tstp, X, A, Acc, f, G, Gcc, I, beta, h, 1.0)
                    incr_convolution_lmix_rmix!(tstp, X, A, Acc, f, G, Gcc, I, beta, h, 1.0)
                    X
                end
                @test distance(srcs[1], srcs[2], tstp, Val(:les)) == 0.0
            end
        end
    end

    @testset "partial lesser: start-up scratch buffer" begin
        for sign in (FERMI, BOSE), k = 1:5
            C = Cn(ntime, ntau, 2, 2, (ntime - 1) * h, beta)
            I = Integrator(k)
            A, Acc = cu_make(C, sign, h, hv)
            B, Bcc = cu_make(C, sign, h, hv)
            f = cu_func(C, h)
            α = 0.4 - 0.9im
            ws = KadanoffBaym._ConvWorkspace(A, B)
            # tstp = k is the boundary case: n₁ = k, length(buf) = k + 1
            for tstp = 1:k
                n₁ = max(tstp - 1, k)
                buf1 = [zeros(C64, 2, 2) for _ = 1:n₁+1]
                buf2 = [zeros(C64, 2, 2) for _ = 1:n₁+1]
                @test KadanoffBaym._incr_convolution_less_adv!(buf1, tstp,
                    A, Acc, f, B, Bcc, I, beta, h, α, ws) === buf1
                @test KadanoffBaym._incr_convolution_lmix_rmix!(buf2, tstp,
                    A, Acc, f, B, Bcc, I, beta, h, α, ws) === buf2
                #
                # Independent evaluation of all n₁ + 1 blocks
                for m = 1:n₁+1
                    x = zeros(C64, 2, 2)
                    for j = 1:n₁+1
                        a = m ≤ j ? A.less[m,j] : -Acc.less[j,m]'
                        b = j ≤ tstp ? Bcc.ret[tstp,j]' : -B.ret[j,tstp]
                        x .+= I.GIW[tstp-1,j-1] .* (a * f[j] * b)
                    end
                    @test buf1[m] ≈ α * h * x rtol = 1e-12 atol = 1e-13
                    x = zeros(C64, 2, 2)
                    for j = 1:ntau
                        b = -sign * Bcc.lmix[tstp,ntau-j+1]'
                        x .+= I.GIW[ntau-1,j-1] .* (A.lmix[m,j] * f[0] * b)
                    end
                    @test buf2[m] ≈ α * (-im * hv) * x rtol = 1e-12 atol = 1e-13
                end
                #
                # The public entries write only the blocks j ≤ tstp
                X = ℱ(C, sign)
                incr_convolution_less_adv!(tstp, X, A, Acc, f, B, Bcc, I, beta, h, α)
                Y = ℱ(C, sign)
                incr_convolution_lmix_rmix!(tstp, Y, A, Acc, f, B, Bcc, I, beta, h, α)
                for j = 1:tstp
                    @test X.less[j,tstp] == buf1[j]
                    @test Y.less[j,tstp] == buf2[j]
                end
                for t = 1:ntime
                    t == tstp && continue
                    @test distance(X, ℱ(C, sign), t) == 0.0
                    @test distance(Y, ℱ(C, sign), t) == 0.0
                end
            end
            # Out of the start-up region
            buf = [zeros(C64, 2, 2) for _ = 1:k+1]
            @test_throws AssertionError KadanoffBaym._incr_convolution_less_adv!(buf,
                k + 1, A, Acc, f, B, Bcc, I, beta, h, α, ws)
            buf = [zeros(C64, 2, 2) for _ = 1:k]
            @test_throws AssertionError KadanoffBaym._incr_convolution_lmix_rmix!(buf,
                1, A, Acc, f, B, Bcc, I, beta, h, α, ws)
        end
    end

    @testset "convolution_les_timediag and convolution_density_matrix" begin
        for sign in (FERMI, BOSE), dims in ((1, 1), (2, 2)), k = 1:5
            C = Cn(ntime, ntau, dims..., (ntime - 1) * h, beta)
            I = Integrator(k)
            A, Acc = cu_make(C, sign, h, hv)
            B, Bcc = cu_make(C, sign, h, hv)
            f = cu_func(C, h)
            for fx in (nothing, f)
                Cref = ℱ(C, sign)
                if fx isa Cf
                    convolution(Cref, A, Acc, fx, B, Bcc, I, beta, h)
                else
                    convolution(Cref, A, Acc, B, Bcc, I, beta, h)
                end
                for tstp = 0:ntime
                    x = convolution_les_timediag(tstp, A, Acc, fx, B, Bcc, I, beta, h)
                    ref = tstp == 0 ? -im * Cref.mat[ntau] : Cref.less[tstp,tstp]
                    @test x ≈ ref rtol = 1e-11 atol = 1e-11
                    ρ = convolution_density_matrix(tstp, A, Acc, fx, B, Bcc, I, beta, h)
                    @test ρ == -im .* x
                end
            end
        end
        # Hermitian convenience entries
        C = Cn(ntime, ntau, 2, 2, (ntime - 1) * h, beta)
        G = ℱ(C)
        init_green!(G, C64[-0.3 0.2im; -0.2im 0.5], 0.0, beta, h)
        f = cu_func(C, h)
        I = Integrator(3)
        for tstp = 0:ntime
            @test convolution_les_timediag(tstp, G, G, I, beta, h) ==
                  convolution_les_timediag(tstp, G, G, nothing, G, G, I, beta, h)
            @test convolution_les_timediag(tstp, G, f, G, I, beta, h) ==
                  convolution_les_timediag(tstp, G, G, f, G, G, I, beta, h)
            @test convolution_density_matrix(tstp, G, G, I, beta, h) ==
                  convolution_density_matrix(tstp, G, G, nothing, G, G, I, beta, h)
            @test convolution_density_matrix(tstp, G, f, G, I, beta, h) ==
                  convolution_density_matrix(tstp, G, G, f, G, G, I, beta, h)
        end
        # Hermitian convenience entries of incr_convolution!
        X = ℱ(C)
        Y = ℱ(C)
        for tstp = 0:ntime
            incr_convolution!(tstp, X, G, G, I, beta, h, 1.0)
            incr_convolution!(tstp, Y, G, G, nothing, G, G, I, beta, h, 1.0)
        end
        @test X == Y
        X = ℱ(C)
        Y = ℱ(C)
        for tstp = 0:ntime
            incr_convolution!(tstp, X, G, f, G, I, beta, h, 1.0)
            incr_convolution!(tstp, Y, G, G, f, G, G, I, beta, h, 1.0)
        end
        @test X == Y
    end

    @testset "preconditions" begin
        C = Cn(ntime, ntau, 2, 2, (ntime - 1) * h, beta)
        A, Acc = cu_make(C, FERMI, h, hv)
        I = Integrator(2)
        X = ℱ(C)
        @test_throws AssertionError incr_convolution!(ntime + 1, X, A, Acc, nothing, A, Acc, I, beta, h, 1.0)
        @test_throws AssertionError incr_convolution_ret!(0, X, A, Acc, nothing, A, Acc, I, h, 1.0)
        @test_throws AssertionError incr_convolution!(3, X[2], A, Acc, nothing, A, Acc, I, beta, h, 1.0)
        Cs = Cn(ntime, ntau, 1, 1, (ntime - 1) * h, beta)
        @test_throws AssertionError incr_convolution!(3, ℱ(Cs), A, Acc, nothing, A, Acc, I, beta, h, 1.0)
        @test_throws AssertionError incr_convolution!(3, X, A, Acc, Cf(Cs), A, Acc, I, beta, h, 1.0)
        #
        # The target must not share storage with the inputs
        B, Bcc = cu_make(C, FERMI, h, hv)
        for (Y, A₁, Acc₁, B₁, Bcc₁) in ((A, A, Acc, B, Bcc), (Acc, A, Acc, B, Bcc),
                                         (B, A, Acc, B, Bcc), (Bcc, A, Acc, B, Bcc))
            @test_throws AssertionError incr_convolution!(3, Y, A₁, Acc₁, nothing, B₁, Bcc₁, I, beta, h, 1.0)
            @test_throws AssertionError incr_convolution_ret!(3, Y, A₁, Acc₁, nothing, B₁, Bcc₁, I, h, 1.0)
            @test_throws AssertionError incr_convolution_tv!(3, Y, A₁, Acc₁, nothing, B₁, Bcc₁, I, beta, h, 1.0)
            @test_throws AssertionError incr_convolution_ret_less!(3, Y, A₁, Acc₁, nothing, B₁, Bcc₁, I, beta, h, 1.0)
            @test_throws AssertionError incr_convolution_less_adv!(3, Y, A₁, Acc₁, nothing, B₁, Bcc₁, I, beta, h, 1.0)
            @test_throws AssertionError incr_convolution_lmix_rmix!(3, Y, A₁, Acc₁, nothing, B₁, Bcc₁, I, beta, h, 1.0)
            @test_throws AssertionError incr_convolution_les!(3, Y, A₁, Acc₁, nothing, B₁, Bcc₁, I, beta, h, 1.0)
            @test_throws AssertionError convolution_time_step(3, Y, A₁, Acc₁, B₁, Bcc₁, I, beta, h)
            @test_throws AssertionError convolution(Y, A₁, Acc₁, B₁, Bcc₁, I, beta, h)
        end
        # A different object sharing one component with an input
        S = deepcopy(A)
        S.lmix = B.lmix
        @test_throws AssertionError incr_convolution!(3, S, A, Acc, nothing, B, Bcc, I, beta, h, 1.0)
        # A deep copy does not
        incr_convolution!(3, deepcopy(A), A, Acc, nothing, B, Bcc, I, beta, h, 1.0)
    end

    #
    # Port of libcntr/test/convolution.cpp:848-885
    #
    @testset "NESSi convolution.cpp: time-diagonal convolution" begin
        nt = 7
        ntauN = 500
        betaN = 5.0
        hN = 0.01
        wa = 1.123
        wb = 0.345
        kt = 5
        mu = 0.0
        eps = 1e-6
        CN = Cn(nt + 1, ntauN + 1, 1, 1, nt * hN, betaN)
        A = ℱ(CN, FERMI)
        B = ℱ(CN, FERMI)
        init_green!(A, fill(C64(wa), 1, 1), mu, betaN, hN)
        init_green!(B, fill(C64(wb), 1, 1), mu, betaN, hN)
        AB = ℱ(CN, FERMI)
        convolution(AB, A, A, B, B, Integrator(kt), betaN, hN)
        err = 0.0
        for tstp = -1:nt
            ρ = convolution_density_matrix(tstp + 1, A, B, Integrator(kt), betaN, hN)
            if tstp == -1
                err = abs(AB.mat[1][1,1] - ρ[1,1])
            else
                err = abs(AB.less[tstp+1,tstp+1][1,1] - im * ρ[1,1])
            end
        end
        # NESSi checks the last error only (REQUIRE outside of the loop)
        @test err < eps
        #
        # Stronger check: all time steps. The Matsubara branch returns
        # -Cᴹ(β), which equals Cᴹ(0⁺) for fermions only up to the jump
        # condition; it is compared with -Cᴹ(β) itself.
        for tstp = 0:nt+1
            ρ = convolution_density_matrix(tstp, A, B, Integrator(kt), betaN, hN)
            if tstp == 0
                @test abs(-AB.mat[ntauN+1][1,1] - ρ[1,1]) < eps
            else
                @test abs(AB.less[tstp,tstp][1,1] - im * ρ[1,1]) < eps
            end
        end
    end

    #
    # Allocation gate: the warmed-call allocations must not grow with the
    # number of time steps.
    #
    @testset "allocation gate" begin
        k = 3
        ntauA = 11
        I = Integrator(k)
        results = Dict{Symbol,Vector{Int}}()
        for nt in (8, 32, 128)
            C = Cn(nt, ntauA, 2, 2, (nt - 1) * h, beta)
            G = ℱ(C)
            init_green!(G, C64[-0.3 0.2im; -0.2im 0.5], 0.0, beta, h)
            f = cu_func(C, h)
            X = ℱ(C)
            ws = KadanoffBaym._ConvWorkspace(G, G)
            calls = (
                ret       = () -> incr_convolution_ret!(nt, X, G, G, f, G, G, I, h, 1.0),
                tv        = () -> incr_convolution_tv!(nt, X, G, G, f, G, G, I, beta, h, 1.0),
                ret_less  = () -> incr_convolution_ret_less!(nt, X, G, G, f, G, G, I, beta, h, 1.0),
                less_adv  = () -> incr_convolution_less_adv!(nt, X, G, G, f, G, G, I, beta, h, 1.0),
                lmix_rmix = () -> incr_convolution_lmix_rmix!(nt, X, G, G, f, G, G, I, beta, h, 1.0),
                les       = () -> incr_convolution_les!(nt, X, G, G, f, G, G, I, beta, h, 1.0),
                ws_les    = () -> KadanoffBaym._incr_convolution_les!(nt, X, G, G, f, G, G, I, beta, h, 1.0, ws),
                ws_ret    = () -> KadanoffBaym._incr_convolution_ret!(nt, X, G, G, f, G, G, I, h, 1.0, ws),
                ws_tv     = () -> KadanoffBaym._incr_convolution_tv!(nt, X, G, G, f, G, G, I, beta, h, 1.0, ws),
            )
            for (name, call) in pairs(calls)
                call()
                call()
                bytes = @allocated call()
                push!(get!(results, name, Int[]), bytes)
            end
        end
        for (name, bytes) in results
            # Fixed scratch (workspace, Matsubara f₀ Bᴹ buffer) is allowed;
            # growth with nt is not.
            @test maximum(bytes) - minimum(bytes) ≤ 256
        end
        # The workspace-bearing versions with a warmed workspace allocate
        # only the constant scratch, independent of nt.
        @test maximum(results[:ws_les]) ≤ 4096
        @test maximum(results[:ws_ret]) ≤ 4096
    end

    #
    # Release only: convergence order of the equal-time lesser convolution
    # of two free fermionic Green's functions (1x1). The exact result is
    # C^<(t,t) = (A^<(t,t) - B^<(t,t)) / (wa - wb).
    #
    if kb_release()
        @testset "release: equal-time convolution order gate" begin
            @test kb_test_profile() == "release"
            wa = 1.123
            wb = 0.345
            betaN = 1.0
            T = 0.25
            nA(w) = 1.0 / (exp(betaN * w) + 1.0)
            # C^<(t,t) of 1/(iω-wa) * 1/(iω-wb), constant in time
            exact = im * (nA(wa) - nA(wb)) / (wa - wb)
            #
            # Grids per order: the errors must stay above the round-off
            # floor. The observed order is k + 2 (the integrand is smooth
            # and the error is dominated by the imaginary-time integral),
            # and the gate requires at least k + 1.
            grids = Dict(1 => [16, 32, 64, 128], 2 => [8, 16, 32, 64],
                         3 => [4, 8, 16, 32], 4 => [6, 8, 12, 16, 24],
                         5 => [6, 8, 10, 12])
            #
            # Error of the equal-time convolution with integration order kk
            # on a grid with N time steps.
            eqt_error = function (kk, N)
                hN = T / N
                CN = Cn(N + 1, N + 1, 1, 1, T, betaN)
                A = ℱ(CN, FERMI)
                B = ℱ(CN, FERMI)
                init_green!(A, fill(C64(wa), 1, 1), 0.0, betaN, hN)
                init_green!(B, fill(C64(wb), 1, 1), 0.0, betaN, hN)
                x = convolution_les_timediag(N + 1, A, B, Integrator(kk), betaN, hN)
                abs(x[1,1] - exact)
            end
            for k = 1:5
                Ns = grids[k]
                errs = [eqt_error(k, N) for N in Ns]
                test_order_gate(Ns, errs, k + 1, 1e-8)
                # Reduced-order mutant (k - 2, see t_response.jl) must be
                # rejected by the same gate.
                if k ≥ 3
                    errm = [eqt_error(k - 2, N) for N in Ns]
                    @test !check_order_gate(Ns, errm, k + 1, 1e-8)[1]
                end
            end
        end
    end
end
