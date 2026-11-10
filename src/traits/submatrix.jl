#
# Project : Lavender
# Source  : submatrix.jl
# Author  : Li Huang (huangli@caep.cn)
# Status  : Unstable
#
# Last modified: 2026/11/10
#

function _submatrix_indices(G::ℱ, rows, cols)
    row_indices = rows isa Integer ? [rows] : collect(rows)
    col_indices = cols isa Integer ? [cols] : collect(cols)
    if isempty(row_indices) || isempty(col_indices)
        throw(DimensionMismatch("Orbital index selections must be non-empty."))
    end
    if !all(index -> index isa Integer, row_indices) ||
       !all(index -> index isa Integer, col_indices)
        throw(BoundsError(G, (rows, cols)))
    end

    ndim1, ndim2 = getdims(G)
    if any(index -> index < 1 || index > ndim1, row_indices) ||
       any(index -> index < 1 || index > ndim2, col_indices)
        throw(BoundsError(G, (rows, cols)))
    end
    return I64.(row_indices), I64.(col_indices)
end


"""
    submatrix(G::ℱ, rows, cols)

Create a contour-ordered Green's function containing the selected orbital
rows and columns of `G` in the requested order.

### Arguments

* `G`: Source contour-ordered Green's function.
* `rows`: Orbital row indices.
* `cols`: Orbital column indices.

### Returns

A new `ℱ` with the same sign and time grids as `G`.
"""
function submatrix(G::ℱ{T}, rows, cols) where {T}
    row_indices, col_indices = _submatrix_indices(G, rows, cols)
    ndim1 = length(row_indices)
    ndim2 = length(col_indices)
    ntime = getntime(G)
    ntau = getntau(G)
    result = ℱ(
        getsign(G),
        Gᵐᵃᵗ(ntau, ndim1, ndim2, zero(T)),
        Gʳᵉᵗ(ntime, ndim1, ndim2, zero(T)),
        Gˡᵐⁱˣ(ntime, ntau, ndim1, ndim2, zero(T)),
        Gˡᵉˢˢ(ntime, ndim1, ndim2, zero(T)),
    )

    for q = 1:ntau
        result.mat[q] = G.mat[q][row_indices,col_indices]
    end
    for i = 1:ntime
        for j = 1:i
            result.ret[i,j] = G.ret[i,j][row_indices,col_indices]
        end
        for q = 1:ntau
            result.lmix[i,q] = G.lmix[i,q][row_indices,col_indices]
        end
        for j = i:ntime
            result.less[i,j] = G.less[i,j][row_indices,col_indices]
        end
    end
    return result
end


"""
    set_submatrix!(G::ℱ, rows, cols, X::ℱ)

Write every independent contour block of `X` into the selected orbital rows
and columns of `G`.

### Arguments

* `G`: Destination contour-ordered Green's function.
* `rows`: Destination orbital row indices.
* `cols`: Destination orbital column indices.
* `X`: Source contour-ordered Green's function.

### Returns

The modified destination `G`.
"""
function set_submatrix!(G::ℱ, rows, cols, X::ℱ)
    row_indices, col_indices = _submatrix_indices(G, rows, cols)
    expected_dims = (length(row_indices), length(col_indices))
    if getdims(X) != expected_dims ||
       getntime(X) != getntime(G) ||
       getntau(X) != getntau(G)
        throw(DimensionMismatch("Source dimensions and time grids must match the selected destination."))
    end

    for q = 1:getntau(G)
        G.mat[q][row_indices,col_indices] = X.mat[q]
    end
    for i = 1:getntime(G)
        for j = 1:i
            G.ret[i,j][row_indices,col_indices] = X.ret[i,j]
        end
        for q = 1:getntau(G)
            G.lmix[i,q][row_indices,col_indices] = X.lmix[i,q]
        end
        for j = i:getntime(G)
            G.less[i,j][row_indices,col_indices] = X.less[i,j]
        end
    end
    return G
end


