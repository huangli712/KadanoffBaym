#
# Project : Lavender
# Source  : write.jl
# Author  : Li Huang (huangli@caep.cn)
# Status  : Unstable
#
# Last modified: 2026/11/10
#

"""
    Base.write(fname::AbstractString, C::Cn)

Write `Cn` struct to disk file which is specified by `fname`. Note that
the file format is defined at function `Base.show(io::IO, C::Cn)`.

### Examples
```julia
using KadanoffBaym
C = Cn(10.0, 5.0)
write("contour.data", C)
```

See also: [`Cn`](@ref).
"""
function Base.write(fname::AbstractString, C::Cn)
    open(fname, "w") do fout
        println(fout, C)
    end
end

"""
    Base.write(fname::AbstractString, cf::Cf{T})

Write `Cf` struct to disk file which is specified by `fname`. Note that
the file format is defined at function `Base.show(io::IO, cf::Cf{T})`.

### Examples
```julia
using KadanoffBaym
cf = Cf(11, 2)
write("cf.data", cf)
```

See also: [`Cf`](@ref).
"""
function Base.write(fname::AbstractString, cf::Cf{T}) where {T}
    open(fname, "w") do fout
        println(fout, cf)
    end
end

"""
    Base.write(fname::AbstractString, mat::Gᵐᵃᵗ{T})

Write `Gᵐᵃᵗ` struct to disk file which is specified by `fname`. Note that
the file format is defined at function `Base.show(io::IO, mat::Gᵐᵃᵗ{T})`.

### Examples
```julia
using KadanoffBaym
mat = Gᵐᵃᵗ(11,2,2,0.2-0.3im)
write("mat.data", mat)
```

See also: [`Gᵐᵃᵗ`](@ref).
"""
function Base.write(fname::AbstractString, mat::Gᵐᵃᵗ{T}) where {T}
    open(fname, "w") do fout
        println(fout, mat)
    end
end

"""
    Base.write(fname::AbstractString, ret::Gʳᵉᵗ{T})

Write `Gʳᵉᵗ` struct to disk file which is specified by `fname`. Note that
the file format is defined at function `Base.show(io::IO, ret::Gʳᵉᵗ{T})`.

### Examples
```julia
using KadanoffBaym
ret = Gʳᵉᵗ(11,2,2,0.2-1.1im)
write("ret.data", ret)
```

See also: [`Gʳᵉᵗ`](@ref).
"""
function Base.write(fname::AbstractString, ret::Gʳᵉᵗ{T}) where {T}
    open(fname, "w") do fout
        println(fout, ret)
    end
end

"""
    Base.write(fname::AbstractString, lmix::Gˡᵐⁱˣ{T})

Write `Gˡᵐⁱˣ` struct to disk file which is specified by `fname`. Note that
the file format is defined at function `Base.show(io::IO, lmix::Gˡᵐⁱˣ{T})`.

### Examples
```julia
using KadanoffBaym
lmix = Gˡᵐⁱˣ(11,3,1,2,0.33+0.78im)
write("lmix.data", lmix)
```

See also: [`Gˡᵐⁱˣ`](@ref).
"""
function Base.write(fname::AbstractString, lmix::Gˡᵐⁱˣ{T}) where {T}
    open(fname, "w") do fout
        println(fout, lmix)
    end
end

"""
    Base.write(fname::AbstractString, less::Gˡᵉˢˢ{T})

Write `Gˡᵉˢˢ` struct to disk file which is specified by `fname`. Note that
the file format is defined at function `Base.show(io::IO, less::Gˡᵉˢˢ{T})`.

### Examples
```julia
using KadanoffBaym
less = Gˡᵉˢˢ(11,2,3,0.4im)
write("less.data", less)
```

See also: [`Gˡᵉˢˢ`](@ref).
"""
function Base.write(fname::AbstractString, less::Gˡᵉˢˢ{T}) where {T}
    open(fname, "w") do fout
        println(fout, less)
    end
end

"""
    Base.write(fname::AbstractString, mat::gᵐᵃᵗ{S})

Write `gᵐᵃᵗ` struct to disk file which is specified by `fname`. Note that
the file format is defined at function `Base.show(io::IO, mat::gᵐᵃᵗ{S})`.

### Examples
```julia
using KadanoffBaym
mat = gᵐᵃᵗ(11,2,2,0.2-0.3im)
write("mat.data", mat)
```

See also: [`gᵐᵃᵗ`](@ref).
"""
function Base.write(fname::AbstractString, mat::gᵐᵃᵗ{S}) where {S}
    open(fname, "w") do fout
        println(fout, mat)
    end
end

"""
    Base.write(fname::AbstractString, ret::gʳᵉᵗ{S})

Write `gʳᵉᵗ` struct to disk file which is specified by `fname`. Note that
the file format is defined at `Base.show(io::IO, ret::gʳᵉᵗ{S})`.

### Examples
```julia
using KadanoffBaym
ret = gʳᵉᵗ(4,2,2,0.4im)
write("ret.data", ret)
```

See also: [`gʳᵉᵗ`](@ref).
"""
function Base.write(fname::AbstractString, ret::gʳᵉᵗ{S}) where {S}
    open(fname, "w") do fout
        println(fout, ret)
    end
end

"""
    Base.write(fname::AbstractString, lmix::gˡᵐⁱˣ{S})

Write `gˡᵐⁱˣ` struct to disk file which is specified by `fname`. Note that
the file format is defined at function `Base.show(io::IO, lmix::gˡᵐⁱˣ{S})`.

### Examples
```julia
using KadanoffBaym
lmix = gˡᵐⁱˣ(5,2,2,0.3im)
write("lmix.data", lmix)
```

See also: [`gˡᵐⁱˣ`](@ref).
"""
function Base.write(fname::AbstractString, lmix::gˡᵐⁱˣ{S}) where {S}
    open(fname, "w") do fout
        println(fout, lmix)
    end
end

"""
    Base.write(fname::AbstractString, less::gˡᵉˢˢ{S})

Write `gˡᵉˢˢ` struct to disk file which is specified by `fname`. Note that
the file format is defined at function `Base.show(io::IO, less::gˡᵉˢˢ{S})`.

### Examples
```julia
using KadanoffBaym
less = gˡᵉˢˢ(5,2,2,0.3im)
write("less.data", less)
```

See also: [`gˡᵉˢˢ`](@ref).
"""
function Base.write(fname::AbstractString, less::gˡᵉˢˢ{S}) where {S}
    open(fname, "w") do fout
        println(fout, less)
    end
end

"""
    write(fname::AbstractString, cfm::ℱ{T})

Write the standard contour-ordered Green's functions to given file.

### Examples
```julia
using KadanoffBaym
C = Cn(5,5,2,2,10.0,5.0)
F = ℱ(C, 0.3im, FERMI)
write("gf.data", F)
```

See also: [`ℱ`](@ref).
"""
function Base.write(fname::AbstractString, cfm::ℱ{T}) where {T}
    open(fname, "w") do fout
        println(fout, cfm)
    end
end

"""
    write(fname::AbstractString, pgf::PseudoGF{T})

Write the pseudo-particle contour-ordered Green's function to given file.

See also: [`PseudoGF`](@ref).
"""
function Base.write(fname::AbstractString, pgf::PseudoGF{T}) where {T}
    open(fname, "w") do fout
        println(fout, pgf)
    end
end

"""
    write(fname::AbstractString, cfv::𝒻{S})

Write the standard contour-ordered Green's functions to given file.

### Examples
```julia
using KadanoffBaym
C = Cn(5,5,2,2,10.0,5.0)
f = 𝒻(C, 3, FERMI)
write("gf.data", f)
```

See also: [`𝒻`](@ref).
"""
function Base.write(fname::AbstractString, cfv::𝒻{S}) where {S}
    open(fname, "w") do fout
        println(fout, cfv)
    end
end

"""
    Base.write(fn::AbstractString, G::Union{ℱᵐᵒᵛ,Cfᵐᵒᵛ})

Write the moving object to the file `fn` as plain text.
"""
function Base.write(fn::AbstractString, G::Union{ℱᵐᵒᵛ,Cfᵐᵒᵛ})
    open(fn, "w") do io
        show(io, G)
    end
end

