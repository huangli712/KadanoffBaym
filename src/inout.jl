#
# Project : Lavender
# Source  : inout.jl
# Author  : Li Huang (huangli@caep.cn)
# Status  : Unstable
#
# Last modified: 2025/11/01
#

#=
### *Cn* : *I/O*
=#

"""
    Base.show(io::IO, C::Cn)

Display `Cn` struct on the `IO` stream.

See also: [`Cn`](@ref).
"""
function Base.show(io::IO, C::Cn)
    println(io, "# Kadanoff-Baym Contour")
    #
    println(io, "ntime : ", C.ntime)
    println(io, "ntau  : ", C.ntau )
    println(io, "ndim1 : ", C.ndim1)
    println(io, "ndim2 : ", C.ndim2)
    println(io, "tmax  : ", C.tmax )
    println(io, "beta  : ", C.beta )
    println(io, "dt    : ", C.dt   )
    println(io, "dtau  : ", C.dtau )
end

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
    Base.read!(io::IO, C::Cn)

Extract parameters from the `IO` stream, and then use them to initialize
the given `Cn` struct. Note that the correctness of the parameters won't
be checked in this function.

See also: [`Cn`](@ref)
"""
function Base.read!(io::IO, C::Cn)
    readline(io) # Skip the comment line
    #
    arr = line_to_array(io)
    C.ntime = parse(I64, arr[3])
    arr = line_to_array(io)
    C.ntau = parse(I64, arr[3])
    #
    arr = line_to_array(io)
    C.ndim1 = parse(I64, arr[3])
    arr = line_to_array(io)
    C.ndim2 = parse(I64, arr[3])
    #
    arr = line_to_array(io)
    C.tmax = parse(F64, arr[3])
    arr = line_to_array(io)
    C.beta = parse(F64, arr[3])
    #
    arr = line_to_array(io)
    C.dt = parse(F64, arr[3])
    arr = line_to_array(io)
    C.dtau = parse(F64, arr[3])
end

"""
    Base.read!(fname::AbstractString, C::Cn)

Extract parameters from disk file which is specified by `fname`, and then
use them to initialize the given `Cn` struct.

### Examples
```julia
using KadanoffBaym
C = Cn()
read!("contour.data", C)
```

See also: [`Cn`](@ref)
"""
function Base.read!(fname::AbstractString, C::Cn)
    if isfile(fname)
        open(fname, "r") do fin
            Base.read!(fin, C)
        end
    else
        error("The $fname file doesn't exist!")
    end
end

#=
### *Cf* : *I/O*
=#

"""
    Base.show(io::IO, cf::Cf{T})

Display `Cf` struct on the `IO` stream.

See also: [`Cf`](@ref).
"""
function Base.show(io::IO, cf::Cf{T}) where {T}
    println(io, "# Contour-Based Function")
    #
    println(io, "ntime : ", cf.ntime)
    println(io, "ndim1 : ", cf.ndim1)
    println(io, "ndim2 : ", cf.ndim2)
    #
    println(io, "data  : ")
    #
    for i = 1:getsize(cf) + 1
        @printf(io, ">%4i :\n", i)
        for m = 1:cf.ndim2
            for n = 1:cf.ndim1
                v = cf.data[i][n,m]
                if T == F64
                    @printf(io, "%4i %4i %16.12f\n", n, m, v)
                elseif T == C64
                    @printf(io, "%4i %4i %16.12f %16.12f\n", n, m, real(v), imag(v))
                else
                    error("The datatype $T is unsupported!")
                end
            end
        end
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
    Base.read!(io::IO, cf::Cf{T})

Extract data from the `IO` stream, and then use them to initialize the
given `Cf` struct.

See also: [`Cf`](@ref).
"""
function Base.read!(io::IO, cf::Cf{T}) where {T}
    readline(io) # Skip the comment line
    #
    # Extract parameters
    arr = line_to_array(io)
    cf.ntime = parse(I64, arr[3])
    arr = line_to_array(io)
    cf.ndim1 = parse(I64, arr[3])
    arr = line_to_array(io)
    cf.ndim2 = parse(I64, arr[3])
    #
    readline(io) # Skip the comment line
    #
    # Prepare necessary memory
    element = fill(zero(T), cf.ndim1, cf.ndim2)
    cf.data = VecArray{T}(undef, cf.ntime + 1)
    #
    # Extract function data
    for i = 1:getsize(cf) + 1
        readline(io) # Skip the comment line
        #
        for m = 1:cf.ndim2
            for n = 1:cf.ndim1
                if T == F64
                    arr = line_to_array(io)
                    element[n,m] = parse(F64, arr[3])
                elseif T == C64
                    arr = line_to_array(io)
                    element[n,m] = parse(F64, arr[3]) + parse(F64, arr[4]) * im
                else
                    error("The datatype $T is unsupported!")
                end
            end
        end
        #
        cf.data[i] = copy(element)
    end
end

"""
    Base.read!(fname::AbstractString, cf::Cf{T})

Extract data from disk file which is specified by `fname`, and then use
them to initialize the given `Cf` struct.

### Examples
```julia
using KadanoffBaym
cf = Cf(11, 2)
read!("cf.data", cf)
```

See also: [`Cf`](@ref).
"""
function Base.read!(fname::AbstractString, cf::Cf{T}) where {T}
    if isfile(fname)
        open(fname, "r") do fin
            Base.read!(fin, cf)
        end
    else
        error("The $fname file doesn't exist!")
    end
end

#=
### *Gᵐᵃᵗ* : *I/O*
=#

"""
    Base.show(io::IO, mat::Gᵐᵃᵗ{T})

Display `Gᵐᵃᵗ` struct on the `IO` stream. Here `Gᵐᵃᵗ` means the Matsubara
component of contour-ordered Green's function ``G^M``.

See also: [`Gᵐᵃᵗ`](@ref).
"""
function Base.show(io::IO, mat::Gᵐᵃᵗ{T}) where {T}
    println(io, "# Contour-Ordered Green's Function: Matsubara Component (G)")
    #
    println(io, "type  : ", mat.type)
    println(io, "ntau  : ", mat.ntau)
    println(io, "ndim1 : ", mat.ndim1)
    println(io, "ndim2 : ", mat.ndim2)
    #
    println(io, "data  : ")
    #
    for i = 1:getsize(mat)
        @printf(io, ">%4i :\n", i)
        for m = 1:mat.ndim2
            for n = 1:mat.ndim1
                v = mat.data[i,1][n,m]
                if T == F64
                    @printf(io, "%4i %4i %16.12f\n", n, m, v)
                elseif T == C64
                    @printf(io, "%4i %4i %16.12f %16.12f\n", n, m, real(v), imag(v))
                else
                    error("The datatype $T is unsupported!")
                end
            end
        end
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
    Base.read!(io::IO, mat::Gᵐᵃᵗ{T})

Extract data from the `IO` stream, and then use them to initialize the
given `Gᵐᵃᵗ` struct.

See also: [`Gᵐᵃᵗ`](@ref).
"""
function Base.read!(io::IO, mat::Gᵐᵃᵗ{T}) where {T}
    readline(io) # Skip the comment line
    #
    # Extract parameters
    arr = line_to_array(io)
    mat.type = arr[3]
    arr = line_to_array(io)
    mat.ntau = parse(I64, arr[3])
    arr = line_to_array(io)
    mat.ndim1 = parse(I64, arr[3])
    arr = line_to_array(io)
    mat.ndim2 = parse(I64, arr[3])
    #
    readline(io) # Skip the comment line
    #
    # Prepare nessary memory
    element = fill(zero(T), mat.ndim1, mat.ndim2)
    mat.data = MatArray{T}(undef, mat.ntau, 1)
    #
    # Extract function data
    for i = 1:getsize(mat)
        readline(io) # Skip the comment line
        #
        for m = 1:mat.ndim2
            for n = 1:mat.ndim1
                if T == F64
                    arr = line_to_array(io)
                    element[n,m] = parse(F64, arr[3])
                elseif T == C64
                    arr = line_to_array(io)
                    element[n,m] = parse(F64, arr[3]) + parse(F64, arr[4]) * im
                else
                    error("The datatype $T is unsupported!")
                end
            end
        end
        #
        mat.data[i,1] = copy(element)
    end
end

"""
    Base.read!(fname::AbstractString, mat::Gᵐᵃᵗ{T})

Extract data from disk file which is specified by `fname`, and then use
them to initialize the given `Gᵐᵃᵗ` struct.

### Examples
```julia
using KadanoffBaym
mat = Gᵐᵃᵗ(11,2,2,0.2-0.3im)
read!("mat.data", mat)
```

See also: [`Gᵐᵃᵗ`](@ref).
"""
function Base.read!(fname::AbstractString, mat::Gᵐᵃᵗ{T}) where {T}
    if isfile(fname)
        open(fname, "r") do fin
            Base.read!(fin, mat)
        end
    else
        error("The $fname file doesn't exist!")
    end
end

#=
### *Gʳᵉᵗ* : *I/O*
=#

"""
    Base.show(io::IO, ret::Gʳᵉᵗ{T})

Display `Gʳᵉᵗ` struct on the `IO` stream. Here `Gʳᵉᵗ` means the retarded
component of contour-ordered Green's function.

See also: [`Gʳᵉᵗ`](@ref).
"""
function Base.show(io::IO, ret::Gʳᵉᵗ{T}) where {T}
    println(io, "# Contour-Ordered Green's Function: Retarded Component (G)")
    #
    println(io, "type  : ", ret.type)
    println(io, "ntime : ", ret.ntime)
    println(io, "ndim1 : ", ret.ndim1)
    println(io, "ndim2 : ", ret.ndim2)
    #
    println(io, "data  : ")
    #
    for i = 1:getsize(ret)
        for j = 1:getsize(ret)
            @printf(io, ">%4i %4i :\n", j, i)
            for m = 1:ret.ndim2
                for n = 1:ret.ndim1
                    v = ret.data[j,i][n,m]
                    if T == F64
                        @printf(io, "%4i %4i %16.12f\n", n, m, v)
                    elseif T == C64
                        @printf(io, "%4i %4i %16.12f %16.12f\n", n, m, real(v), imag(v))
                    else
                        error("The datatype $T is unsupported!")
                    end
                end
            end
        end
    end
end

"""
    Base.write(fname::AbstractString, ret::Gʳᵉᵗ{T})

Write `Gʳᵉᵗ` struct to disk file which is specified by `fname`. Note that
the file format is defined at function `Base.show(io::IO, ret::Gʳᵉᵗ{T})`.

See also: [`Gʳᵉᵗ`](@ref).
"""
function Base.write(fname::AbstractString, ret::Gʳᵉᵗ{T}) where {T}
    open(fname, "w") do fout
        println(fout, ret)
    end
end

"""
    Base.read!(io::IO, ret::Gʳᵉᵗ{T})

Extract data from the `IO` stream, and then use them to initialize the
given `Gʳᵉᵗ` struct.

See also: [`Gʳᵉᵗ`](@ref).
"""
function Base.read!(io::IO, ret::Gʳᵉᵗ{T}) where {T}
    readline(io) # Skip the comment line
    #
    # Extract parameters
    arr = line_to_array(io)
    ret.type = arr[3]
    arr = line_to_array(io)
    ret.ntime = parse(I64, arr[3])
    arr = line_to_array(io)
    ret.ndim1 = parse(I64, arr[3])
    arr = line_to_array(io)
    ret.ndim2 = parse(I64, arr[3])
    #
    readline(io) # Skip the comment line
    #
    # Prepare necessary memory
    element = fill(zero(T), ret.ndim1, ret.ndim2)
    ret.data = MatArray{T}(undef, ret.ntime, ret.ntime)
    #
    # Extract function data
    for i = 1:getsize(ret)
        for j = 1:getsize(ret)
            readline(io) # Skip the comment line
            #
            for m = 1:ret.ndim2
                for n = 1:ret.ndim1
                    if T == F64
                        arr = line_to_array(io)
                        element[n,m] = parse(F64, arr[3])
                    elseif T == C64
                        arr = line_to_array(io)
                        element[n,m] = parse(F64, arr[3]) + parse(F64, arr[4]) * im
                    else
                        error("The datatype $T is unsupported!")
                    end
                end
            end
            #
            ret.data[j,i] = copy(element)
        end
    end
end

"""
    Base.read!(fname::AbstractString, ret::Gʳᵉᵗ{T})

Extract data from disk file which is specified by `fname`, and then use
them to initialize the given `Gʳᵉᵗ` struct.

See also: [`Gʳᵉᵗ`](@ref).
"""
function Base.read!(fname::AbstractString, ret::Gʳᵉᵗ{T}) where {T}
    if isfile(fname)
        open(fname, "r") do fin
            Base.read!(fin, ret)
        end
    else
        error("The $fname file doesn't exist!")
    end
end

#=
### *Gˡᵐⁱˣ* : *I/O*
=#

"""
    Base.show(io::IO, lmix::Gˡᵐⁱˣ{T})

Display `Gˡᵐⁱˣ` struct on the `IO` stream. Here `Gˡᵐⁱˣ` means the
left-mixing component of contour-ordered Green's function.

See also: [`Gˡᵐⁱˣ`](@ref).
"""
function Base.show(io::IO, lmix::Gˡᵐⁱˣ{T}) where {T}
    println(io, "# Contour-Ordered Green's Function: Left-Mixing Component (G)")
    #
    println(io, "type  : ", lmix.type)
    println(io, "ntime : ", lmix.ntime)
    println(io, "ntau  : ", lmix.ntau)
    println(io, "ndim1 : ", lmix.ndim1)
    println(io, "ndim2 : ", lmix.ndim2)
    #
    println(io, "data  : ")
    #
    for i = 1:getntau(lmix)
        for j = 1:getntime(lmix)
            @printf(io, ">%4i %4i :\n", j, i)
            for m = 1:lmix.ndim2
                for n = 1:lmix.ndim1
                    v = lmix.data[j,i][n,m]
                    if T == F64
                        @printf(io, "%4i %4i %16.12f\n", n, m, v)
                    elseif T == C64
                        @printf(io, "%4i %4i %16.12f %16.12f\n", n, m, real(v), imag(v))
                    else
                        error("The datatype $T is unsupported!")
                    end
                end
            end
        end
    end
end

"""
    Base.write(fname::AbstractString, lmix::Gˡᵐⁱˣ{T})

Write `Gˡᵐⁱˣ` struct to disk file which is specified by `fname`. Note that
the file format is defined at function `Base.show(io::IO, lmix::Gˡᵐⁱˣ{T})`.

See also: [`Gˡᵐⁱˣ`](@ref).
"""
function Base.write(fname::AbstractString, lmix::Gˡᵐⁱˣ{T}) where {T}
    open(fname, "w") do fout
        println(fout, lmix)
    end
end

"""
    Base.read!(io::IO, lmix::Gˡᵐⁱˣ{T})

Extract data from the `IO` stream, and then use them to initialize the
given `Gˡᵐⁱˣ` struct.

See also: [`Gˡᵐⁱˣ`](@ref).
"""
function Base.read!(io::IO, lmix::Gˡᵐⁱˣ{T}) where {T}
    readline(io) # Skip the comment line
    #
    # Extract parameters
    arr = line_to_array(io)
    lmix.type = arr[3]
    arr = line_to_array(io)
    lmix.ntime = parse(I64, arr[3])
    arr = line_to_array(io)
    lmix.ntau = parse(I64, arr[3])
    arr = line_to_array(io)
    lmix.ndim1 = parse(I64, arr[3])
    arr = line_to_array(io)
    lmix.ndim2 = parse(I64, arr[3])
    #
    readline(io) # Skip the comment line
    #
    # Prepare necessary memory
    element = fill(zero(T), lmix.ndim1, lmix.ndim2)
    lmix.data = MatArray{T}(undef, lmix.ntime, lmix.ntau)
    #
    # Extract function data
    for i = 1:getntau(lmix)
        for j = 1:getntime(lmix)
            readline(io) # Skip the comment line
            #
            for m = 1:lmix.ndim2
                for n = 1:lmix.ndim1
                    if T == F64
                        arr = line_to_array(io)
                        element[n,m] = parse(F64, arr[3])
                    elseif T == C64
                        arr = line_to_array(io)
                        element[n,m] = parse(F64, arr[3]) + parse(F64, arr[4]) * im
                    else
                        error("The datatype $T is unsupported!")
                    end
                end
            end
            #
            lmix.data[j,i] = copy(element)
        end
    end
end

"""
    Base.read!(fname::AbstractString, lmix::Gˡᵐⁱˣ{T})

Extract data from disk file which is specified by `fname`, and then use
them to initialize the given `Gˡᵐⁱˣ` struct.

See also: [`Gˡᵐⁱˣ`](@ref).
"""
function Base.read!(fname::AbstractString, lmix::Gˡᵐⁱˣ{T}) where {T}
    if isfile(fname)
        open(fname, "r") do fin
            Base.read!(fin, lmix)
        end
    else
        error("The $fname file doesn't exist!")
    end
end

#=
### *Gˡᵉˢˢ* : *I/O*
=#

"""
    Base.show(io::IO, less::Gˡᵉˢˢ{T})

Display `Gˡᵉˢˢ` struct on the `IO` stream. Here `Gˡᵉˢˢ` means the lesser
component of contour-ordered Green's function.

See also: [`Gˡᵉˢˢ`](@ref).
"""
function Base.show(io::IO, less::Gˡᵉˢˢ{T}) where {T}
    println(io, "# Contour-Ordered Green's Function: Lesser Component (G)")
    #
    println(io, "type  : ", less.type)
    println(io, "ntime : ", less.ntime)
    println(io, "ndim1 : ", less.ndim1)
    println(io, "ndim2 : ", less.ndim2)
    #
    println(io, "data  : ")
    #
    for i = 1:getsize(less)
        for j = 1:getsize(less)
            @printf(io, ">%4i %4i :\n", j, i)
            for m = 1:less.ndim2
                for n = 1:less.ndim1
                    v = less.data[j,i][n,m]
                    if T == F64
                        @printf(io, "%4i %4i %16.12f\n", n, m, v)
                    elseif T == C64
                        @printf(io, "%4i %4i %16.12f %16.12f\n", n, m, real(v), imag(v))
                    else
                        error("The datatype $T is unsupported!")
                    end
                end
            end
        end
    end
end

"""
    Base.write(fname::AbstractString, less::Gˡᵉˢˢ{T})

Write `Gˡᵉˢˢ` struct to disk file which is specified by `fname`. Note that
the file format is defined at function `Base.show(io::IO, less::Gˡᵉˢˢ{T})`.

See also: [`Gˡᵉˢˢ`](@ref).
"""
function Base.write(fname::AbstractString, less::Gˡᵉˢˢ{T}) where {T}
    open(fname, "w") do fout
        println(fout, less)
    end
end

"""
    Base.read!(io::IO, less::Gˡᵉˢˢ{T})

Extract data from the `IO` stream, and then use them to initialize the
given `Gˡᵉˢˢ` struct.

See also: [`Gˡᵉˢˢ`](@ref).
"""
function Base.read!(io::IO, less::Gˡᵉˢˢ{T}) where {T}
    readline(io) # Skip the comment line
    #
    # Extract parameters
    arr = line_to_array(io)
    less.type = arr[3]
    arr = line_to_array(io)
    less.ntime = parse(I64, arr[3])
    arr = line_to_array(io)
    less.ndim1 = parse(I64, arr[3])
    arr = line_to_array(io)
    less.ndim2 = parse(I64, arr[3])
    #
    readline(io) # Skip the comment line
    #
    # Prepare necessary memory
    element = fill(zero(T), less.ndim1, less.ndim2)
    less.data = MatArray{T}(undef, less.ntime, less.ntime)
    #
    # Extract function data
    for i = 1:getsize(less)
        for j = 1:getsize(less)
            readline(io) # Skip the comment line
            #
            for m = 1:less.ndim2
                for n = 1:less.ndim1
                    if T == F64
                        arr = line_to_array(io)
                        element[n,m] = parse(F64, arr[3])
                    elseif T == C64
                        arr = line_to_array(io)
                        element[n,m] = parse(F64, arr[3]) + parse(F64, arr[4]) * im
                    else
                        error("The datatype $T is unsupported!")
                    end
                end
            end
            #
            less.data[j,i] = copy(element)
        end
    end
end

"""
    Base.read!(fname::AbstractString, less::Gˡᵉˢˢ{T})

Extract data from disk file which is specified by `fname`, and then use
them to initialize the given `Gˡᵉˢˢ` struct.

See also: [`Gˡᵉˢˢ`](@ref).
"""
function Base.read!(fname::AbstractString, less::Gˡᵉˢˢ{T}) where {T}
    if isfile(fname)
        open(fname, "r") do fin
            Base.read!(fin, less)
        end
    else
        error("The $fname file doesn't exist!")
    end
end

#=
### *gᵐᵃᵗ* : *I/O*
=#

"""
    Base.show(io::IO, mat::gᵐᵃᵗ{S})

Display `gᵐᵃᵗ` struct on the `IO` stream. Here `gᵐᵃᵗ` means the Matsubara
component of contour-ordered Green's function.

See also: [`gᵐᵃᵗ`](@ref).
"""
function Base.show(io::IO, mat::gᵐᵃᵗ{S}) where {S}
    println(io, "# Contour-Ordered Green's Function: Matsubara Component (g)")
    #
    println(io, "type  : ", mat.type)
    println(io, "ntau  : ", mat.ntau)
    println(io, "ndim1 : ", mat.ndim1)
    println(io, "ndim2 : ", mat.ndim2)
    #
    println(io, "data  : ")
    #
    for i = 1:getsize(mat)
        @printf(io, ">%4i :\n", i)
        for m = 1:mat.ndim2
            for n = 1:mat.ndim1
                v = mat.data[i][n,m]
                if S == F64
                    @printf(io, "%4i %4i %16.12f\n", n, m, v)
                elseif S == C64
                    @printf(io, "%4i %4i %16.12f %16.12f\n", n, m, real(v), imag(v))
                else
                    error("The datatype $S is unsupported!")
                end
            end
        end
    end
end

"""
    Base.write(fname::AbstractString, mat::gᵐᵃᵗ{S})

Write `gᵐᵃᵗ` struct to disk file which is specified by `fname`. Note that
the file format is defined at function `Base.show(io::IO, mat::gᵐᵃᵗ{S})`.

See also: [`gᵐᵃᵗ`](@ref).
"""
function Base.write(fname::AbstractString, mat::gᵐᵃᵗ{S}) where {S}
    open(fname, "w") do fout
        println(fout, mat)
    end
end

"""
    Base.read!(io::IO, mat::gᵐᵃᵗ{S})

Extract data from the `IO` stream, and then use them to initialize the
given `gᵐᵃᵗ` struct.

See also: [`gᵐᵃᵗ`](@ref).
"""
function Base.read!(io::IO, mat::gᵐᵃᵗ{S}) where {S}
    readline(io) # Skip the comment line
    #
    # Extract parameters
    arr = line_to_array(io)
    mat.type = arr[3]
    arr = line_to_array(io)
    mat.ntau = parse(I64, arr[3])
    arr = line_to_array(io)
    mat.ndim1 = parse(I64, arr[3])
    arr = line_to_array(io)
    mat.ndim2 = parse(I64, arr[3])
    #
    readline(io) # Skip the comment line
    #
    # Prepare necessary memory
    element = fill(zero(S), mat.ndim1, mat.ndim2)
    mat.data = VecArray{S}(undef, mat.ntau)
    #
    # Extract function data
    for i = 1:getsize(mat)
        readline(io) # Skip the comment line
        #
        for m = 1:mat.ndim2
            for n = 1:mat.ndim1
                if S == F64
                    arr = line_to_array(io)
                    element[n,m] = parse(F64, arr[3])
                elseif S == C64
                    arr = line_to_array(io)
                    element[n,m] = parse(F64, arr[3]) + parse(F64, arr[4]) * im
                else
                    error("The datatype $S is unsupported!")
                end
            end
        end
        #
        mat.data[i] = copy(element)
    end
end

"""
    Base.read!(fname::AbstractString, mat::gᵐᵃᵗ{S})

Extract data from disk file which is specified by `fname`, and then use
them to initialize the given `gᵐᵃᵗ` struct.

See also: [`gᵐᵃᵗ`](@ref).
"""
function Base.read!(fname::AbstractString, mat::gᵐᵃᵗ{S}) where {S}
    if isfile(fname)
        open(fname, "r") do fin
            Base.read!(fin, mat)
        end
    else
        error("The $fname file doesn't exist!")
    end
end

#=
### *gʳᵉᵗ* : *I/O*
=#

"""
    Base.show(io::IO, ret::gʳᵉᵗ{S})

Display `gʳᵉᵗ` struct on the `IO` stream. Here `gʳᵉᵗ` means the retarded
component of contour-ordered Green's function.

See also: [`gʳᵉᵗ`](@ref).
"""
function Base.show(io::IO, ret::gʳᵉᵗ{S}) where {S}
    println(io, "# Contour-Ordered Green's Function: Retarded Component (g)")
    #
    println(io, "type  : ", ret.type)
    println(io, "tstp  : ", ret.tstp)
    println(io, "ndim1 : ", ret.ndim1)
    println(io, "ndim2 : ", ret.ndim2)
    #
    println(io, "data  : ")
    #
    for i = 1:getsize(ret)
        @printf(io, ">%4i :\n", i)
        for m = 1:ret.ndim2
            for n = 1:ret.ndim1
                v = ret.data[i][n,m]
                if S == F64
                    @printf(io, "%4i %4i %16.12f\n", n, m, v)
                elseif S == C64
                    @printf(io, "%4i %4i %16.12f %16.12f\n", n, m, real(v), imag(v))
                else
                    error("The datatype $S is unsupported!")
                end
            end
        end
    end
end

"""
    Base.write(fname::AbstractString, ret::gʳᵉᵗ{S})

Write `gʳᵉᵗ` struct to disk file which is specified by `fname`. Note that
the file format is defined at `Base.show(io::IO, ret::gʳᵉᵗ{S})`.

See also: [`gʳᵉᵗ`](@ref).
"""
function Base.write(fname::AbstractString, ret::gʳᵉᵗ{S}) where {S}
    open(fname, "w") do fout
        println(fout, ret)
    end
end

"""
    Base.read!(io::IO, ret::gʳᵉᵗ{S})

Extract data from the `IO` stream, and then use them to initialize the
given `gʳᵉᵗ` struct.

See also: [`gʳᵉᵗ`](@ref).
"""
function Base.read!(io::IO, ret::gʳᵉᵗ{S}) where {S}
    readline(io) # Skip the comment line
    #
    # Extract parameters
    arr = line_to_array(io)
    ret.type = arr[3]
    arr = line_to_array(io)
    ret.tstp = parse(I64, arr[3])
    arr = line_to_array(io)
    ret.ndim1 = parse(I64, arr[3])
    arr = line_to_array(io)
    ret.ndim2 = parse(I64, arr[3])
    #
    readline(io) # Skip the comment line
    #
    # Prepare necessary memory
    element = fill(zero(S), ret.ndim1, ret.ndim2)
    ret.data = VecArray{S}(undef, ret.tstp)
    #
    # Extract function data
    for i = 1:getsize(ret)
        readline(io) # Skip the comment line
        #
        for m = 1:ret.ndim2
            for n = 1:ret.ndim1
                if S == F64
                    arr = line_to_array(io)
                    element[n,m] = parse(F64, arr[3])
                elseif S == C64
                    arr = line_to_array(io)
                    element[n,m] = parse(F64, arr[3]) + parse(F64, arr[4]) * im
                else
                    error("The datatype $S is unsupported!")
                end
            end
        end
        #
        ret.data[i] = copy(element)
    end
end

"""
    Base.read!(fname::AbstractString, ret::gʳᵉᵗ{S})

Extract data from disk file which is specified by `fname`, and then use
them to initialize the given `gʳᵉᵗ` struct.

See also: [`gʳᵉᵗ`](@ref).
"""
function Base.read!(fname::AbstractString, ret::gʳᵉᵗ{S}) where {S}
    if isfile(fname)
        open(fname, "r") do fin
            Base.read!(fin, ret)
        end
    else
        error("The $fname file doesn't exist!")
    end
end

#=
### *gˡᵐⁱˣ* : *I/O*
=#

"""
    Base.show(io::IO, lmix::gˡᵐⁱˣ{S})

Display `gˡᵐⁱˣ` struct on the `IO` stream. Here `gˡᵐⁱˣ` means the left-mixing
component of contour-ordered Green's function.

See also: [`gˡᵐⁱˣ`](@ref).
"""
function Base.show(io::IO, lmix::gˡᵐⁱˣ{S}) where {S}
    println(io, "# Contour-Ordered Green's Function: Left-Mixing Component (g)")
    #
    println(io, "type  : ", lmix.type)
    println(io, "ntau  : ", lmix.ntau)
    println(io, "ndim1 : ", lmix.ndim1)
    println(io, "ndim2 : ", lmix.ndim2)
    #
    println(io, "data  : ")
    #
    for i = 1:getsize(lmix)
        @printf(io, ">%4i :\n", i)
        for m = 1:lmix.ndim2
            for n = 1:lmix.ndim1
                v = lmix.data[i][n,m]
                if S == F64
                    @printf(io, "%4i %4i %16.12f\n", n, m, v)
                elseif S == C64
                    @printf(io, "%4i %4i %16.12f %16.12f\n", n, m, real(v), imag(v))
                else
                    error("The datatype $S is unsupported!")
                end
            end
        end
    end
end

"""
    Base.write(fname::AbstractString, lmix::gˡᵐⁱˣ{S})

Write `gˡᵐⁱˣ` struct to disk file which is specified by `fname`. Note that
the file format is defined at function `Base.show(io::IO, lmix::gˡᵐⁱˣ{S})`.

See also: [`gˡᵐⁱˣ`](@ref).
"""
function Base.write(fname::AbstractString, lmix::gˡᵐⁱˣ{S}) where {S}
    open(fname, "w") do fout
        println(fout, lmix)
    end
end

"""
    Base.read!(io::IO, lmix::gˡᵐⁱˣ{S})

Extract data from the `IO` stream, and then use them to initialize the
given `gˡᵐⁱˣ` struct.

See also: [`gˡᵐⁱˣ`](@ref).
"""
function Base.read!(io::IO, lmix::gˡᵐⁱˣ{S}) where {S}
    readline(io) # Skip the comment line
    #
    # Extract parameters
    arr = line_to_array(io)
    lmix.type = arr[3]
    arr = line_to_array(io)
    lmix.ntau = parse(I64, arr[3])
    arr = line_to_array(io)
    lmix.ndim1 = parse(I64, arr[3])
    arr = line_to_array(io)
    lmix.ndim2 = parse(I64, arr[3])
    #
    readline(io) # Skip the comment line
    #
    # Prepare necessary memory
    element = fill(zero(S), lmix.ndim1, lmix.ndim2)
    lmix.data = VecArray{S}(undef, lmix.ntau)
    #
    # Extract function data
    for i = 1:getsize(lmix)
        readline(io) # Skip the comment line
        #
        for m = 1:lmix.ndim2
            for n = 1:lmix.ndim1
                if S == F64
                    arr = line_to_array(io)
                    element[n,m] = parse(F64, arr[3])
                elseif S == C64
                    arr = line_to_array(io)
                    element[n,m] = parse(F64, arr[3]) + parse(F64, arr[4]) * im
                else
                    error("The datatype $T is unsupported!")
                end
            end
        end
        #
        lmix.data[i] = copy(element)
    end
end

"""
    Base.read!(fname::AbstractString, lmix::gˡᵐⁱˣ{S})

Extract data from disk file which is specified by `fname`, and then use
them to initialize the given `gˡᵐⁱˣ` struct.

See also: [`gˡᵐⁱˣ`](@ref).
"""
function Base.read!(fname::AbstractString, lmix::gˡᵐⁱˣ{S}) where {S}
    if isfile(fname)
        open(fname, "r") do fin
            Base.read!(fin, lmix)
        end
    else
        error("The $fname file doesn't exist!")
    end
end

#=
### *gˡᵉˢˢ* : *I/O*
=#

"""
    Base.show(io::IO, less::gˡᵉˢˢ{S})

Display `gˡᵉˢˢ` struct on the `IO` stream. Here `gˡᵉˢˢ` means the lesser
component of contour-ordered Green's function.

See also: [`gˡᵉˢˢ`](@ref).
"""
function Base.show(io::IO, less::gˡᵉˢˢ{S}) where {S}
    println(io, "# Contour-Ordered Green's Function: Lesser Component (g)")
    #
    println(io, "type  : ", less.type)
    println(io, "tstp  : ", less.tstp)
    println(io, "ndim1 : ", less.ndim1)
    println(io, "ndim2 : ", less.ndim2)
    #
    println(io, "data  : ")
    #
    for i = 1:getsize(less)
        @printf(io, ">%4i :\n", i)
        for m = 1:less.ndim2
            for n = 1:less.ndim1
                v = less.data[i][n,m]
                if S == F64
                    @printf(io, "%4i %4i %16.12f\n", n, m, v)
                elseif S == C64
                    @printf(io, "%4i %4i %16.12f %16.12f\n", n, m, real(v), imag(v))
                else
                    error("The datatype $T is unsupported!")
                end
            end
        end
    end
end

"""
    Base.write(fname::AbstractString, less::gˡᵉˢˢ{S})

Write `gˡᵉˢˢ` struct to disk file which is specified by `fname`. Note that
the file format is defined at function `Base.show(io::IO, less::gˡᵉˢˢ{S})`.

See also: [`gˡᵉˢˢ`](@ref).
"""
function Base.write(fname::AbstractString, less::gˡᵉˢˢ{S}) where {S}
    open(fname, "w") do fout
        println(fout, less)
    end
end

"""
    Base.read!(io::IO, less::gˡᵉˢˢ{S})

Extract data from the `IO` stream, and then use them to initialize the
given `gˡᵉˢˢ` struct.

See also: [`gˡᵉˢˢ`](@ref).
"""
function Base.read!(io::IO, less::gˡᵉˢˢ{S}) where {S}
    readline(io) # Skip the comment line
    #
    # Extract parameters
    arr = line_to_array(io)
    less.type = arr[3]
    arr = line_to_array(io)
    less.tstp = parse(I64, arr[3])
    arr = line_to_array(io)
    less.ndim1 = parse(I64, arr[3])
    arr = line_to_array(io)
    less.ndim2 = parse(I64, arr[3])
    #
    readline(io) # Skip the comment line
    #
    # Prepare necessary memory
    element = fill(zero(S), less.ndim1, less.ndim2)
    less.data = VecArray{S}(undef, less.tstp)
    #
    # Extract function data
    for i = 1:getsize(less)
        readline(io) # Skip the comment line
        #
        for m = 1:less.ndim2
            for n = 1:less.ndim1
                if S == F64
                    arr = line_to_array(io)
                    element[n,m] = parse(F64, arr[3])
                elseif S == C64
                    arr = line_to_array(io)
                    element[n,m] = parse(F64, arr[3]) + parse(F64, arr[4]) * im
                else
                    error("The datatype $T is unsupported!")
                end
            end
        end
        #
        less.data[i] = copy(element)
    end
end

"""
    Base.read!(fname::AbstractString, less::gˡᵉˢˢ{S})

Extract data from disk file which is specified by `fname`, and then use
them to initialize the given `gˡᵉˢˢ` struct.

See also: [`gˡᵉˢˢ`](@ref).
"""
function Base.read!(fname::AbstractString, less::gˡᵉˢˢ{S}) where {S}
    if isfile(fname)
        open(fname, "r") do fin
            Base.read!(fin, less)
        end
    else
        error("The $fname file doesn't exist!")
    end
end

#=
### *ℱ* : *I/O*
=#

"""
    Base.show(io::IO, cfm::ℱ{T})

Display `ℱ` struct on the `IO` stream. Here `ℱ` means the standard contour-
ordered Green's function, which includes four independent components,
namely `mat`, `ret`, `lmix`, and `less`.

See also: [`ℱ`](@ref).
"""
function Base.show(io::IO, cfm::ℱ{T}) where {T}
    println(io, "# Standard Contour-Ordered Green's Function:")
    #
    println(io, "sign  : ", cfm.sign)
    println(io)
    #
    println(io, cfm.mat)
    println(io, cfm.ret)
    println(io, cfm.lmix)
    println(io, cfm.less)
end

"""
    write(fname::AbstractString, cfm::ℱ{T})

Write the contour-ordered Green's functions to given file.

See also: [`ℱ`](@ref).
"""
function Base.write(fname::AbstractString, cfm::ℱ{T}) where {T}
    open(fname, "w") do fout
        println(fout, cfm)
    end
end

"""
    read!(io::IO, cfm::ℱ{T})

Read the contour-ordered Green's functions from the `IO` stream.

See also: [`ℱ`](@ref).
"""
function Base.read!(io::IO, cfm::ℱ{T}) where {T}
    readline(io) # Skip the comment line
    #
    # Extract parameters
    arr = line_to_array(io)
    cfm.sign = parse(I64, arr[3])
    #
    readline(io) # Skip the comment line
    #
    Base.read!(io, cfm.mat)
    Base.read!(io, cfm.ret)
    Base.read!(io, cfm.lmix)
    Base.read!(io, cfm.less)
end

"""
    read!(fname::AbstractString, cfm::ℱ{T})

Read the contour-ordered Green's functions from given file.

See also: [`ℱ`](@ref).
"""
function Base.read!(fname::AbstractString, cfm::ℱ{T}) where {T}
    if isfile(fname)
        open(fname, "r") do fin
            Base.read!(fin, cfm)
        end
    else
        error("The $fname file doesn't exist!")
    end
end

#=
### *𝒻* : *I/O*
=#

"""
    Base.show(io::IO, cfv::𝒻{S})

Display `𝒻` struct on the `IO` stream. Here `𝒻` means the standard contour-
ordered Green's function at given time step `tstp`, which includes four
independent components, namely `mat`, `ret`, `lmix`, and `less`.

See also: [`𝒻`](@ref).
"""
function Base.show(io::IO, cfv::𝒻{S}) where {S}
    println(io, "# Standard Contour-Ordered Green's Function:")
    #
    println(io, "sign  : ", cfv.sign)
    println(io, "tstp  : ", cfv.tstp)
    println(io)
    #
    println(io, cfv.mat)
    println(io, cfv.ret)
    println(io, cfv.lmix)
    println(io, cfv.less)
end

"""
    write(fname::AbstractString, cfv::𝒻{S})

Write the contour-ordered Green's functions to given file.

See also: [`𝒻`](@ref).
"""
function Base.write(fname::AbstractString, cfv::𝒻{S}) where {S}
    open(fname, "w") do fout
        println(fout, cfv)
    end
end

"""
    read!(io::IO, cfv::𝒻{S})

Read the contour-ordered Green's functions from the `IO` stream.

See also: [`𝒻`](@ref).
"""
function Base.read!(io::IO, cfv::𝒻{S}) where {S}
    readline(io) # Skip the comment line
    #
    # Extract parameters
    arr = line_to_array(io)
    cfv.sign = parse(I64, arr[3])
    arr = line_to_array(io)
    cfv.tstp = parse(I64, arr[3])
    #
    readline(io) # Skip the comment line
    #
    Base.read!(io, cfv.mat)
    Base.read!(io, cfv.ret)
    Base.read!(io, cfv.lmix)
    Base.read!(io, cfv.less)
end

"""
    read!(fname::AbstractString, cfv::𝒻{S})

Read the contour-ordered Green's functions from given file.

See also: [`𝒻`](@ref).
"""
function Base.read!(fname::AbstractString, cfv::𝒻{S}) where {S}
    if isfile(fname)
        open(fname, "r") do fin
            Base.read!(fin, cfv)
        end
    else
        error("The $fname file doesn't exist!")
    end
end
