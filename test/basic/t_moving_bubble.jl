#
# t_moving_bubble.jl
#
# To test the moving Bubble1/Bubble2 diagrams against the full contour
# bubbles. The full Green's functions have zero mixing components (the
# moving types do not store them), so the real-time components of the
# full and the moving bubbles agree.
#
# NESSi: cntr_bubble_moving_impl.hpp.
#

isdefined(Main, :kb_test_profile) ||
    include(joinpath(@__DIR__, "..", "common", "profile.jl"))

using Test
using LinearAlgebra
using Random
using KadanoffBaym

# Full Green's function with fixed random real-time data, the Matsubara
# and mixing components are zero
function mb_full(nt, d, sign; seed)
    rng = MersenneTwister(seed)
    C = Cn(nt, 4, d, d, (nt - 1) * 0.1, 1.0)
    F = ℱ(C, sign)
    for n = 1:nt, m = 1:n
        # G^R(t,t) and G^<(t,t) are anti-hermitian
        R = randn(rng, C64, d, d)
        F.ret[n, m] = m == n ? (R - R') / 2 : R
        L = randn(rng, C64, d, d)
        F.less[m, n] = m == n ? (L - L') / 2 : L
    end
    return F
end

# Moving copy of the full function X (hermitian partner Xcc) with the
# leading time step n
function mb_moving(X, Xcc, tc, n)
    G = ℱᵐᵒᵛ(tc, getdims(X)[1], getsign(X))
    set_from_G_backward!(G, X, Xcc, n)
    return G
end

# Fill the leading row of the moving function from the full functions,
# the same rule as set_from_G_backward!
function mb_fillrow!(Xm, X, Xcc)
    n = gettstp(Xm)
    v = timestep_view(Xm, 0)
    for j = 0:min(Xm.tc, n - 1)
        v[:ret, j] = X.ret[n, n - j]
        v[:less, j] = -Matrix(Xcc.less[n - j, n]')
    end
end

# Maximum difference between the leading row of the moving C and the full
# contour C at the time step n (the lesser only for hermitian inputs)
function mb_rowdiff(Cm, Cf, n; less = true)
    e = 0.0
    for j = 0:Cm.tc
        n - j ≥ 1 || continue
        e = max(e, maximum(abs.(Cm[:ret, n, n - j] - Cf.ret[n, n - j])))
        less || continue
        e = max(e, maximum(abs.(Cm[:less, n - j, n] - Cf.less[n - j, n])))
    end
    return e
end

@testset verbose = true "KadanoffBaym: moving bubbles (full equivalence with zero mixing components)" begin

    @testset "$name (d = $d, signs $sa/$sb, tc = $tc, herm = $herm)" for name in (:bubble1, :bubble2),
            d in (1, 2), (sa, sb) in ((FERMI, FERMI), (FERMI, BOSE), (BOSE, BOSE)), tc in (3, 6),
            herm in (true, false), imap in (:herm, :mixed)
        nt = 12
        n = 8
        bub! = name == :bubble1 ? bubble1! : bubble2!
        A = mb_full(nt, d, sa; seed = 1)
        B = mb_full(nt, d, sb; seed = 3)
        # Hermitian inputs: Acc = A and Bcc = B; otherwise unrelated partners.
        # Then only the retarded component is compared: the full bubble
        # stores C^<(t',t) while the moving one stores C^<(t,t'), they are
        # related by hermiticity only for hermitian inputs.
        Acc = herm ? A : mb_full(nt, d, sa; seed = 2)
        Bcc = herm ? B : mb_full(nt, d, sb; seed = 4)
        Cf = ℱ(Cn(nt, 4, d, d, (nt - 1) * 0.1, 1.0), sa * sb)
        # Orbital maps (c1,c2) -> (a1,a2,b1,b2). The full contour bubble
        # stores C^<(t',t), the moving one C^<(t,t'); they are related by
        # C^<(t',t) = -C^<(t,t')^† only if C is hermitian, i.e. for
        # hermitian inputs and a map that commutes with the transposition
        # (:herm). The :mixed map is checked for the retarded component.
        idx(c1, c2) = imap == :herm ? (d - c1 + 1, d - c2 + 1, c1, c2) :
                                      (d - c1 + 1, c2, c1, d - c2 + 1)
        cmpless = herm && imap == :herm
        for m = 1:nt, c1 = 1:d, c2 = 1:d
            a1, a2, b1, b2 = idx(c1, c2)
            bub!(m, Cf, c1, c2, A, Acc, a1, a2, B, Bcc, b1, b2)
        end
        Am = mb_moving(A, Acc, tc, n)
        Accm = mb_moving(Acc, A, tc, n)
        Bm = mb_moving(B, Bcc, tc, n)
        Bccm = mb_moving(Bcc, B, tc, n)
        Cm = ℱᵐᵒᵛ(tc, d, sa * sb)
        set_from_G_backward!(Cm, ℱ(Cn(nt, 4, d, d, (nt - 1) * 0.1, 1.0), sa * sb), n)
        function step!(t)
            for c1 = 1:d, c2 = 1:d
                a1, a2, b1, b2 = idx(c1, c2)
                bub!(t, Cm, c1, c2, Am, Accm, a1, a2, Bm, Bccm, b1, b2)
            end
        end
        step!(n)
        @test mb_rowdiff(Cm, Cf, n; less = cmpless) < 1e-12
        # Two forward! steps: the new rows are computed, the history row of
        # the step n is unchanged
        for s = 1:2
            for (Xm, X, Xcc) in ((Am, A, Acc), (Accm, Acc, A), (Bm, B, Bcc), (Bccm, Bcc, B))
                forward!(Xm)
                mb_fillrow!(Xm, X, Xcc)
            end
            forward!(Cm)
            step!(n + s)
            @test mb_rowdiff(Cm, Cf, n + s; less = cmpless) < 1e-12
        end
        @test mb_rowdiff(Cm, Cf, n; less = cmpless) < 1e-12
    end

    @testset "failure paths" begin
        d, tc, n = 2, 3, 6
        F = mb_full(8, d, FERMI; seed = 1)
        A = mb_moving(F, F, tc, n)
        C = mb_moving(mb_full(8, d, BOSE; seed = 2), mb_full(8, d, BOSE; seed = 2), tc, n)
        @test_throws ArgumentError bubble1!(n + 1, C, A, A)
        @test_throws ArgumentError bubble1!(n, A, A, A)
        @test_throws ArgumentError bubble2!(n, mb_moving(F, F, tc, n), A, A)
        @test_throws BoundsError bubble2!(n, C, 3, 1, A, 1, 1, A, 1, 1)
        @test_throws ArgumentError bubble1!(n, C, A, mb_moving(F, F, tc + 1, n))
        @test_throws ArgumentError bubble1!(n, C, A, ℱᵐᵒᵛ(tc, d))
    end
end
