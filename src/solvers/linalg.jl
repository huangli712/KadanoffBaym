#
# Project : Lavender
# Source  : linalg.jl
# Author  : Li Huang (huangli@caep.cn)
# Status  : Unstable
#
# Last modified: 2026/11/09
#

#=
*Remarks* : *Block Linear Systems*

The start-up and time-stepping procedures of the Dyson equation lead to
small linear systems whose unknowns are 𝑑 × 𝑑 matrices. The right action
system reads

```math
\begin{equation}
\sum^{n}_{q=1} M_{pq} X_q = Q_p, \quad p = 1, \cdots, n,
\end{equation}
```

and the left action system reads

```math
\begin{equation}
\sum^{n}_{q=1} X_q M_{pq} = Q_p, \quad p = 1, \cdots, n.
\end{equation}
```

The right action system is solved by assembling the 𝑛 × 𝑛 blocks into a
(𝑛𝑑) × (𝑛𝑑) matrix, and all the columns of ``Q_p`` are treated as multiple
right-hand sides. The left action system is transformed into a right one
by taking the hermitian conjugate of every block,
``\sum_q M^{\dagger}_{pq} X^{\dagger}_q = Q^{\dagger}_p``. Note that the
block structure itself is not transposed.

*References* :

Please see [`NESSi`] `cntr_elements.hpp`, `element_linsolve_right` and
`element_linsolve_left`.
=#

#=
### *Private Struct* : *_BlockSolver*
=#

"""
    _BlockSolver

Workspace for the block linear systems. `n` is the number of blocks, `d`
is the dimension of each block, and `m` is the number of columns of the
right-hand sides (right action) or the number of rows of the right-hand
sides (left action). It is a private struct.

`Mbig` stores the assembled matrix, `Mlu` stores its LU factorization,
`Qbig`, `Xbig`, and `Rbig` store the right-hand sides, the solution, and
the residual, respectively.
"""
struct _BlockSolver
    n    :: I64
    d    :: I64
    m    :: I64
    Mbig :: Matrix{C64}
    Mlu  :: Matrix{C64}
    Qbig :: Matrix{C64}
    Xbig :: Matrix{C64}
    Rbig :: Matrix{C64}
end

"""
    _BlockSolver(n::I64, d::I64, m::I64 = d)

Constructor. See [`_BlockSolver`](@ref) for the meanings of `n`, `d`,
and `m`.
"""
function _BlockSolver(n::I64, d::I64, m::I64 = d)
    @assert n ≥ 1
    @assert d ≥ 1
    @assert m ≥ 1
    N = n * d
    _BlockSolver(n, d, m,
                 zeros(C64, N, N), zeros(C64, N, N),
                 zeros(C64, N, m), zeros(C64, N, m), zeros(C64, N, m))
end

#=
### *Private Service Functions* : *Block Access*
=#

"""
    _mbig(ws::_BlockSolver, p::I64, q::I64)

Return a write-through view of the (p,q) block of the assembled matrix.
It is a private function.
"""
@inline function _mbig(ws::_BlockSolver, p::I64, q::I64)
    d = ws.d
    return view(ws.Mbig, (p - 1) * d + 1 : p * d, (q - 1) * d + 1 : q * d)
end

"""
    _qbig(ws::_BlockSolver, p::I64)

Return a write-through view of the p-th block of the right-hand sides.
It is a private function.
"""
@inline function _qbig(ws::_BlockSolver, p::I64)
    d = ws.d
    return view(ws.Qbig, (p - 1) * d + 1 : p * d, :)
end

"""
    _xbig(ws::_BlockSolver, p::I64)

Return a view of the p-th block of the solution. It is a private
function.
"""
@inline function _xbig(ws::_BlockSolver, p::I64)
    d = ws.d
    return view(ws.Xbig, (p - 1) * d + 1 : p * d, :)
end

#=
### *Private Service Functions* : *Solvers*
=#

"""
    _solve_block!(ws::_BlockSolver)

Solve `Mbig * Xbig = Qbig`, where `Mbig` and `Qbig` have already been
assembled. The input data must be finite, the system must be regular,
and the normalized residual is checked. No explicit inverse is formed
and no regularization is applied. It is a private function.

### Arguments
* ws -> A _BlockSolver struct.

### Returns
* ws.Xbig -> The solution.
"""
function _solve_block!(ws::_BlockSolver)
    # Sanity check
    all(isfinite, ws.Mbig) || throw(DomainError(ws.Mbig, "matrix is not finite"))
    all(isfinite, ws.Qbig) || throw(DomainError(ws.Qbig, "right-hand side is not finite"))

    # LU factorization with partial pivoting
    copyto!(ws.Mlu, ws.Mbig)
    F = lu!(ws.Mlu; check = false)
    issuccess(F) || throw(SingularException(F.info))

    # Solve the linear equations
    copyto!(ws.Xbig, ws.Qbig)
    ldiv!(F, ws.Xbig)
    all(isfinite, ws.Xbig) || throw(DomainError(ws.Xbig, "solution is not finite"))

    # Check the normalized residual
    mul!(ws.Rbig, ws.Mbig, ws.Xbig)
    ws.Rbig .-= ws.Qbig
    res = norm(ws.Rbig)
    scale = norm(ws.Mbig) * norm(ws.Xbig) + norm(ws.Qbig)
    if res > 1e-10 * scale
        error("The block linear system is ill-conditioned (residual = $res)")
    end

    return ws.Xbig
end

"""
    _check_blocks(M::AbstractMatrix, Q::AbstractVector, ws::_BlockSolver,
                  qsize::Tuple{I64,I64})

Check the shapes of the blocks. It is a private function.
"""
function _check_blocks(M::AbstractMatrix, Q::AbstractVector, ws::_BlockSolver,
                       qsize::Tuple{I64,I64})
    n, d = ws.n, ws.d
    size(M) == (n, n) || throw(DimensionMismatch("M must have $n × $n blocks"))
    length(Q) == n || throw(DimensionMismatch("Q must have $n blocks"))
    for p = 1:n
        size(Q[p]) == qsize || throw(DimensionMismatch("wrong size of Q[$p]"))
        for q = 1:n
            size(M[p,q]) == (d, d) || throw(DimensionMismatch("M[$p,$q] must be $d × $d"))
        end
    end
end

"""
    _solve_block_right!(X::AbstractVector, M::AbstractMatrix,
                        Q::AbstractVector, ws::_BlockSolver)

Solve ``\\sum_q M_{pq} X_q = Q_p``. `M[p,q]` is a 𝑑 × 𝑑 block, `Q[p]`
and `X[p]` are 𝑑 × 𝑚 blocks. `X[p]` must be writable (a matrix or a
write-through view). It is a private function.

### Arguments
* X  -> Blocks of the solution.
* M  -> Blocks of the matrix.
* Q  -> Blocks of the right-hand sides.
* ws -> A _BlockSolver struct.

### Returns
* X -> Blocks of the solution.

See also: [`_solve_block_left!`](@ref).
"""
function _solve_block_right!(X::AbstractVector, M::AbstractMatrix,
                             Q::AbstractVector, ws::_BlockSolver)
    _check_blocks(M, Q, ws, (ws.d, ws.m))
    length(X) == ws.n || throw(DimensionMismatch("X must have $(ws.n) blocks"))

    for p = 1:ws.n
        copyto!(_qbig(ws, p), Q[p])
        for q = 1:ws.n
            copyto!(_mbig(ws, p, q), M[p,q])
        end
    end

    _solve_block!(ws)

    for p = 1:ws.n
        copyto!(X[p], _xbig(ws, p))
    end

    return X
end

"""
    _solve_block_left!(X::AbstractVector, M::AbstractMatrix,
                       Q::AbstractVector, ws::_BlockSolver)

Solve ``\\sum_q X_q M_{pq} = Q_p``. `M[p,q]` is a 𝑑 × 𝑑 block, `Q[p]`
and `X[p]` are 𝑚 × 𝑑 blocks. `X[p]` must be writable (a matrix or a
write-through view). It is a private function.

### Arguments
* X  -> Blocks of the solution.
* M  -> Blocks of the matrix.
* Q  -> Blocks of the right-hand sides.
* ws -> A _BlockSolver struct.

### Returns
* X -> Blocks of the solution.

See also: [`_solve_block_right!`](@ref).
"""
function _solve_block_left!(X::AbstractVector, M::AbstractMatrix,
                            Q::AbstractVector, ws::_BlockSolver)
    _check_blocks(M, Q, ws, (ws.m, ws.d))
    length(X) == ws.n || throw(DimensionMismatch("X must have $(ws.n) blocks"))

    # ∑_q M_{pq}† X_q† = Q_p†
    for p = 1:ws.n
        adjoint!(_qbig(ws, p), Q[p])
        for q = 1:ws.n
            adjoint!(_mbig(ws, p, q), M[p,q])
        end
    end

    _solve_block!(ws)

    for p = 1:ws.n
        adjoint!(X[p], _xbig(ws, p))
    end

    return X
end

"""
    _solve_block_right!(X::AbstractMatrix, M::AbstractMatrix{C64},
                        Q::AbstractMatrix, ws::_BlockSolver)

Solve `M X = Q` for a single 𝑑 × 𝑑 block `M`, `Q` and `X` are
𝑑 × 𝑚 matrices. `ws` must be created by `_BlockSolver(1, d, m)`. It
is a private function.
"""
function _solve_block_right!(X::AbstractMatrix, M::AbstractMatrix{C64},
                             Q::AbstractMatrix, ws::_BlockSolver)
    @assert ws.n == 1
    copyto!(ws.Mbig, M)
    copyto!(ws.Qbig, Q)
    _solve_block!(ws)
    copyto!(X, ws.Xbig)
    return X
end

"""
    _solve_block_left!(X::AbstractMatrix, M::AbstractMatrix{C64},
                       Q::AbstractMatrix, ws::_BlockSolver)

Solve `X M = Q` for a single 𝑑 × 𝑑 block `M`, `Q` and `X` are
𝑚 × 𝑑 matrices. `ws` must be created by `_BlockSolver(1, d, m)`. It
is a private function.
"""
function _solve_block_left!(X::AbstractMatrix, M::AbstractMatrix{C64},
                            Q::AbstractMatrix, ws::_BlockSolver)
    @assert ws.n == 1
    adjoint!(ws.Mbig, M)
    adjoint!(ws.Qbig, Q)
    _solve_block!(ws)
    adjoint!(X, ws.Xbig)
    return X
end

"""
    _solve_block_right(M::AbstractMatrix{<:AbstractMatrix},
                       Q::AbstractVector{<:AbstractMatrix})

Solve ``\\sum_q M_{pq} X_q = Q_p`` and return the blocks of `X`. It is
a private function.

See also: [`_solve_block_right!`](@ref).
"""
function _solve_block_right(M::AbstractMatrix{<:AbstractMatrix},
                            Q::AbstractVector{<:AbstractMatrix})
    n = length(Q)
    n ≥ 1 || throw(DimensionMismatch("Q is empty"))
    d, m = size(Q[1])
    ws = _BlockSolver(n, d, m)
    X = [zeros(C64, d, m) for _ = 1:n]
    return _solve_block_right!(X, M, Q, ws)
end

"""
    _solve_block_left(M::AbstractMatrix{<:AbstractMatrix},
                      Q::AbstractVector{<:AbstractMatrix})

Solve ``\\sum_q X_q M_{pq} = Q_p`` and return the blocks of `X`. It is
a private function.

See also: [`_solve_block_left!`](@ref).
"""
function _solve_block_left(M::AbstractMatrix{<:AbstractMatrix},
                           Q::AbstractVector{<:AbstractMatrix})
    n = length(Q)
    n ≥ 1 || throw(DimensionMismatch("Q is empty"))
    m, d = size(Q[1])
    ws = _BlockSolver(n, d, m)
    X = [zeros(C64, m, d) for _ = 1:n]
    return _solve_block_left!(X, M, Q, ws)
end
