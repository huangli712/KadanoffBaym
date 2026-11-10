#
# Project : Lavender
# Source  : copyzone.jl
# Author  : Li Huang (huangli@caep.cn)
# Status  : Unstable
#
# Last modified: 2026/11/10
#

#=
### *CopyZone Struct*
=#

"""
    CopyZone(x₁::I64, y₁::I64, x₂::I64, y₂::I64)

Struct to define a rectangular zone for element-wise copy operations.

### Arguments
* x₁ -> Starting row index (1-based).
* y₁ -> Starting column index (1-based).
* x₂ -> Ending row index (1-based).
* y₂ -> Ending column index (1-based).

### Notes
This struct defines a rectangular region `[x₁:x₂, y₁:y₂]` for copying
elements between two different matrices. All indices are 1-based and
must satisfy `x₂ ≥ x₁ ≥ 1` and `y₂ ≥ y₁ ≥ 1`.

See also: [`elemcpy!`](@ref).
"""
struct CopyZone
    x₁::I64
    y₁::I64
    x₂::I64
    y₂::I64
end

"""
    CopyZone(x::I64, y::I64)

Create a single-point copy zone.

### Arguments
* x -> Row index (1-based).
* y -> Column index (1-based).

### Returns
* CopyZone object representing a single cell at position (x, y).

See also: [`elemcpy!`](@ref).
"""
function CopyZone(x::I64, y::I64)
    return CopyZone(x, y, x, y)
end

"""
    CopyZone(x::I64, y::I64, δ::I64)

Create a square-like copy zone starting from position (x, y).

### Arguments
* x -> Starting row index (1-based).
* y -> Starting column index (1-based).
* δ -> Width (or height) of the square region (must be ≥ 1).

### Returns
* CopyZone object representing a square region `[x:x+δ-1, y:y+δ-1]`.

See also: [`elemcpy!`](@ref).
"""
function CopyZone(x::I64, y::I64, δ::I64)
    @assert δ ≥ 1
    return CopyZone(x, y, x + δ - 1, y + δ - 1)
end

"""
    isvalid(cz::CopyZone)

Check if a copy zone is valid.

### Arguments
* cz -> CopyZone object to validate.

### Returns
* `true` if the copy zone is valid, `false` otherwise.

### Notes
A copy zone is valid if all indices satisfy `x₂ ≥ x₁ ≥ 1` and `y₂ ≥ y₁ ≥ 1`.

See also: [`CopyZone`](@ref).
"""
function isvalid(cz::CopyZone)
    return cz.x₂ ≥ cz.x₁ ≥ 1 && cz.y₂ ≥ cz.y₁ ≥ 1
end
