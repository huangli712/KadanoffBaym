#
# Project : Lavender
# Source  : ness.jl
# Author  : Li Huang (huangli@caep.cn)
# Status  : Unstable
#
# Last modified: 2026/11/10
#

#=
### *Input and Output* : *NESS2*
=#

#=
*Remarks* : *NESS2 Text Format*

The steady-state data is written as plain text. The header line is

```text
# FFTArray Nft size1
# HermMatrixNESS Nft size1 h dw
```

and every other line contains `domain component index i j re im`, where
`domain` is `time` or `freq`, `component` is `data` (FFTArray) or `ret`
and `les` (HermMatrixNESS), `index` is the 1-based storage index, and
`i, j` are the orbital indices. All matrix elements are written. The
reader checks the header, the number of lines, the ranges of the indices,
duplicated entries, and finite values. It is the plain text counterpart
of `print_to_file` and `read_from_file` in NESSi.
=#

"""
    _write_fft_rows(io::IO, A::FFTArray{C64}, comp::String)

Write the time and frequency data of `A`. It is a private function.
"""
function _write_fft_rows(io::IO, A::FFTArray{C64}, comp::String)
    d₁, d₂ = getdims(A)
    for (dom, X) in (("time", A.time), ("freq", A.freq))
        for n = 1:getnft(A), j = 1:d₂, i = 1:d₁
            z = X[i, j, n]
            @printf(io, "%s %s %d %d %d %.17g %.17g\n", dom, comp, n, i, j, real(z), imag(z))
        end
    end
    return nothing
end

"""
    _read_fft_rows!(lines, arrays::Dict{String,FFTArray{C64}}, path::String)

Parse the data lines into the given arrays (component name => FFTArray).
It is a private function.
"""
function _read_fft_rows!(lines, arrays::Dict{String,FFTArray{C64}}, path::String)
    A = first(values(arrays))
    Nft = getnft(A)
    d₁, d₂ = getdims(A)
    nexp = 2 * length(arrays) * Nft * d₁ * d₂
    length(lines) == nexp ||
        throw(ArgumentError("$path: expected $nexp data lines, got $(length(lines))"))
    seen = Set{Tuple{String,String,I64,I64,I64}}()
    for (l, line) in enumerate(lines)
        f = split(line)
        length(f) == 7 ||
            throw(ArgumentError("$path: line $(l + 1) must have 7 columns"))
        dom, comp = String(f[1]), String(f[2])
        dom in ("time", "freq") ||
            throw(ArgumentError("$path: unknown domain $dom at line $(l + 1)"))
        haskey(arrays, comp) ||
            throw(ArgumentError("$path: unknown component $comp at line $(l + 1)"))
        n = tryparse(I64, f[3])
        i = tryparse(I64, f[4])
        j = tryparse(I64, f[5])
        (n !== nothing && i !== nothing && j !== nothing) ||
            throw(ArgumentError("$path: bad index at line $(l + 1)"))
        (1 ≤ n ≤ Nft && 1 ≤ i ≤ d₁ && 1 ≤ j ≤ d₂) ||
            throw(ArgumentError("$path: index out of range at line $(l + 1)"))
        re = tryparse(F64, f[6])
        im_ = tryparse(F64, f[7])
        (re !== nothing && im_ !== nothing && isfinite(re) && isfinite(im_)) ||
            throw(ArgumentError("$path: bad value at line $(l + 1)"))
        key = (dom, comp, n, i, j)
        key in seen && throw(ArgumentError("$path: duplicated entry at line $(l + 1)"))
        push!(seen, key)
        X = dom == "time" ? arrays[comp].time : arrays[comp].freq
        X[i, j, n] = complex(re, im_)
    end
    return nothing
end

"""
    _read_ness_header(path::String, kind::String, nfield::I64)

Read and check the header of a NESS2 text file. It is a private function.
"""
function _read_ness_header(path::String, kind::String, nfield::I64)
    lines = filter(!isempty, strip.(readlines(path)))
    isempty(lines) && throw(ArgumentError("$path: empty file"))
    h = split(lines[1])
    (length(h) == nfield && h[1] == "#" && h[2] == kind) ||
        throw(ArgumentError("$path: bad header, expected `# $kind ...`"))
    return h, lines[2:end]
end

"""
    write_fftarray(path::String, A::FFTArray{C64})

Write an FFTArray struct to a plain text file, see *Remarks* : *NESS2 Text
Format*.

See also: [`read_fftarray`](@ref).
"""
function write_fftarray(path::String, A::FFTArray{C64})
    open(path, "w") do io
        @printf(io, "# FFTArray %d %d\n", getnft(A), getdims(A)[1])
        _write_fft_rows(io, A, "data")
    end
    return nothing
end

"""
    read_fftarray(path::String)

Read an FFTArray struct from a plain text file, see *Remarks* : *NESS2
Text Format*.

See also: [`write_fftarray`](@ref).
"""
function read_fftarray(path::String)
    h, lines = _read_ness_header(path, "FFTArray", 4)
    Nft = tryparse(I64, h[3])
    size1 = tryparse(I64, h[4])
    (Nft !== nothing && size1 !== nothing && Nft ≥ 1 && size1 ≥ 1) ||
        throw(ArgumentError("$path: bad header values"))
    A = FFTArray(Nft, size1)
    _read_fft_rows!(lines, Dict("data" => A), path)
    return A
end

"""
    write_ness(path::String, G::HermMatrixNESS{C64})

Write a HermMatrixNESS struct to a plain text file, see *Remarks* : *NESS2
Text Format*.

See also: [`read_ness`](@ref).
"""
function write_ness(path::String, G::HermMatrixNESS{C64})
    open(path, "w") do io
        @printf(io, "# HermMatrixNESS %d %d %.17g %.17g\n",
                G.grid.Nft, getdims(G)[1], G.grid.h, G.grid.dw)
        _write_fft_rows(io, G.ret, "ret")
        _write_fft_rows(io, G.les, "les")
    end
    return nothing
end

"""
    read_ness(path::String)

Read a HermMatrixNESS struct from a plain text file, see *Remarks* : *NESS2
Text Format*.

See also: [`write_ness`](@ref).
"""
function read_ness(path::String)
    h, lines = _read_ness_header(path, "HermMatrixNESS", 6)
    Nft = tryparse(I64, h[3])
    size1 = tryparse(I64, h[4])
    hh = tryparse(F64, h[5])
    dw = tryparse(F64, h[6])
    (Nft !== nothing && size1 !== nothing && hh !== nothing && dw !== nothing &&
     size1 ≥ 1) || throw(ArgumentError("$path: bad header values"))
    G = HermMatrixNESS(GridInfo(Nft, hh, dw), size1)
    _read_fft_rows!(lines, Dict("ret" => G.ret, "les" => G.les), path)
    return G
end

