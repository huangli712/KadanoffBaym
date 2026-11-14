#
# Project : Lavender
# Source  : text_output.jl
# Author  : Li Huang (huangli@caep.cn)
# Status  : Unstable
#
# Last modified: 2026/11/10
#

#
# Deterministic plain-text output of the Phase 5 examples. The files are
# first written to a temporary file in the same directory and then moved
# to the target, so an interrupted run never leaves a partial file.
#

using Printf

_fmt17(x::Real) = @sprintf("%.17g", x)

# Write through a temporary file and move it to `path`
function _atomic_write(f::Function, path::AbstractString)
    dir = dirname(abspath(path))
    isdir(dir) || throw(ArgumentError("no such directory $dir"))
    tmp, io = mktemp(dir)
    try
        f(io)
        close(io)
        mv(tmp, path; force = true)
    catch
        isopen(io) && close(io)
        rm(tmp; force = true)
        rethrow()
    end
    return path
end

"""
    write_table(path, names, rows)

Write a table: the first line is `# 1:name1 2:name2 ⋯`, every row is a
vector of real numbers written with 17 significant digits. `rows` is any
iterable of vectors; all rows must have `length(names)` entries.
"""
function write_table(path::AbstractString, names::AbstractVector{<:AbstractString}, rows)
    _atomic_write(path) do io
        println(io, "# ", join(("$(i):$(n)" for (i, n) in enumerate(names)), " "))
        for r in rows
            length(r) == length(names) ||
                throw(DimensionMismatch("a row has $(length(r)) entries, expected $(length(names))"))
            all(isfinite, r) || throw(DomainError(r, "non-finite value in the output"))
            println(io, join((_fmt17(x) for x in r), " "))
        end
    end
end

"""
    write_green_components(path, rows)

Write selected Green's function components as `component i j a b re im`
(NESSi indices, see the roadmap). `rows` is any iterable of tuples
`(component, i, j, a, b, value)`.
"""
function write_green_components(path::AbstractString, rows)
    _atomic_write(path) do io
        println(io, "# component i j a b re im")
        for (c, i, j, a, b, v) in rows
            (isfinite(real(v)) && isfinite(imag(v))) ||
                throw(DomainError(v, "non-finite value in the output"))
            println(io, c, " ", i, " ", j, " ", a, " ", b, " ", _fmt17(real(v)), " ",
                    _fmt17(imag(v)))
        end
    end
end
