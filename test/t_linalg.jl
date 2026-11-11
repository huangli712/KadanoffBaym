#
# t_linalg.jl
#
# To test the private block linear solvers in src/linalg.jl.
#
# NESSi Catch tests ported here:
#   libcntr/test/linalg.cpp, sections complex_sq_solve and
#   complex_sq_solve_many (the real-valued sections are covered by the
#   complex solver as well)
#

using Test
using LinearAlgebra
using KadanoffBaym

const _solve_block_right = KadanoffBaym._solve_block_right
const _solve_block_left = KadanoffBaym._solve_block_left
const _solve_block_right! = KadanoffBaym._solve_block_right!
const _BlockSolver = KadanoffBaym._BlockSolver

#
# Deterministic, non-hermitian and non-commuting complex block
#
function la_block(d::Int, s::Real)
    return [complex(sin(1.3 * s + 0.7 * i + 1.1 * j), cos(0.9 * s + 0.4 * i - 1.7 * j))
            for i = 1:d, j = 1:d]
end

#
# Diagonally dominant block system with k × k blocks of size d × d
#
function la_system(k::Int, d::Int)
    M = [la_block(d, 10p + q) for p = 1:k, q = 1:k]
    for p = 1:k
        M[p,p] += (2.0 * k * d) * I
    end
    return M
end

# ∑_q M_{pq} X_q - Q_p, normalized
function la_res_right(M, X, Q)
    k = length(Q)
    r = sum(norm(sum(M[p,q] * X[q] for q = 1:k) - Q[p]) for p = 1:k)
    return r / sum(norm(Q[p]) for p = 1:k)
end

# ∑_q X_q M_{pq} - Q_p, normalized
function la_res_left(M, X, Q)
    k = length(Q)
    r = sum(norm(sum(X[q] * M[p,q] for q = 1:k) - Q[p]) for p = 1:k)
    return r / sum(norm(Q[p]) for p = 1:k)
end

@testset "linalg: NESSi Catch (linalg.cpp)" begin
    A = ComplexF64[1.0+0.2im 2.2 3.4; 4.6 5.7+1.0im 6.8; 7.5 8.5 9.6]
    #
    # complex_sq_solve
    b = reshape(ComplexF64[1.1+0.1im, 2.4, 3.5+0.3im], 3, 1)
    x_exact = ComplexF64[0.17752793262472366+0.01775980726928994im,
                         -0.1949605345360082+0.015227852835187461im,
                         0.3985109425906921+0.0038921558730451444im]
    X = _solve_block_right(reshape([A], 1, 1), [b])
    @test norm(X[1][:,1] - x_exact) < 1e-12
    #
    # complex_sq_solve_many: two right-hand sides
    B = ComplexF64[1.1+0.1im 0.3+1.1im; 2.4 2.4+1.2im; 3.5+0.3im 1.2+0.7im]
    X_exact = ComplexF64[0.17752793262472366+0.01775980726928994im -0.13432028547481906+0.23896436702037188im;
                         -0.1949605345360082+0.015227852835187461im 0.369779042978215-1.5729342395547015im;
                         0.3985109425906921+0.0038921558730451444im -0.0974708046097587+1.2789279462043928im]
    X = _solve_block_right(reshape([A], 1, 1), [B])
    @test norm(X[1] - X_exact) < 1e-12
end

@testset "linalg: block systems (d = $d)" for d = 1:3
    for k = 1:5
        M = la_system(k, d)
        Q = [la_block(d, 100 + 7p) for p = 1:k]
        #
        Xr = _solve_block_right(M, Q)
        @test la_res_right(M, Xr, Q) < 1e-12
        #
        Xl = _solve_block_left(M, Q)
        @test la_res_left(M, Xl, Q) < 1e-12
        #
        # The two actions differ for non-commuting blocks
        if d > 1
            @test la_res_right(M, Xl, Q) > 1e-3
            @test la_res_left(M, Xr, Q) > 1e-3
        end
    end
end

@testset "linalg: scalar blocks and write-through targets" begin
    # d = 1 follows the same code path as the matrix case
    M = reshape([fill(2.0 + 1.0im, 1, 1)], 1, 1)
    Q = [fill(1.0 - 3.0im, 1, 1)]
    @test _solve_block_right(M, Q)[1][1,1] ≈ (1.0 - 3.0im) / (2.0 + 1.0im) atol = 1e-15
    @test _solve_block_left(M, Q)[1][1,1] ≈ (1.0 - 3.0im) / (2.0 + 1.0im) atol = 1e-15
    #
    # The solution is written into views
    d, k = 2, 3
    M = la_system(k, d)
    Q = [la_block(d, 50 + p) for p = 1:k]
    data = zeros(ComplexF64, d, d, k)
    X = [view(data, :, :, p) for p = 1:k]
    ws = _BlockSolver(k, d)
    _solve_block_right!(X, M, Q, ws)
    @test la_res_right(M, [data[:,:,p] for p = 1:k], Q) < 1e-12
    #
    # Several right-hand side columns (d × m blocks)
    Q3 = [hcat(la_block(d, 70 + p), la_block(d, 80 + p)) for p = 1:k]
    X3 = _solve_block_right(M, Q3)
    @test size(X3[1]) == (d, 2d)
    @test la_res_right(M, X3, Q3) < 1e-12
end

@testset "linalg: error paths" begin
    d, k = 2, 2
    M = la_system(k, d)
    Q = [la_block(d, p) for p = 1:k]
    #
    # Shape mismatch
    @test_throws DimensionMismatch _solve_block_right(M[1:1,1:2], Q)
    @test_throws DimensionMismatch _solve_block_right(M, [Q[1]])
    @test_throws DimensionMismatch _solve_block_right(M, [Q[1], zeros(ComplexF64, 3, 2)])
    Mbad = copy(M)
    Mbad[1,2] = zeros(ComplexF64, 3, 3)
    @test_throws DimensionMismatch _solve_block_right(Mbad, Q)
    @test_throws DimensionMismatch _solve_block_left(Mbad, Q)
    @test_throws DimensionMismatch _solve_block_right(M, Matrix{ComplexF64}[])
    #
    # Singular block system
    Ms = [ones(ComplexF64, d, d) for p = 1:k, q = 1:k]
    @test_throws SingularException _solve_block_right(Ms, Q)
    @test_throws SingularException _solve_block_left(Ms, Q)
    #
    # Non-finite input
    Mn = copy(M)
    Mn[2,2] = fill(NaN + 0.0im, d, d)
    @test_throws DomainError _solve_block_right(Mn, Q)
    Qn = copy(Q)
    Qn[1] = fill(Inf + 0.0im, d, d)
    @test_throws DomainError _solve_block_left(M, Qn)
end
