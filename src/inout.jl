#
# Project : Lavender
# Source  : inout.jl
# Author  : Li Huang (huangli@caep.cn)
# Status  : Unstable
#
# Last modified: 2025/10/28
#

#=
### *Cn* : *I/O*
=#

"""
    Base.show(io::IO, C::Cn)

Display `Cn` struct on the io stream.

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
    Base.read!(fname::AbstractString, C::Cn)

Extract parameters from disk file, and then use them to initialize the
given `Cn` struct.

See also: [`Cn`](@ref)
"""
function Base.read!(fname::AbstractString, C::Cn)
    if isfile(fname)
        open(fname, "r") do fin
            readline(fin) # Skip the comment line
            #
            arr = line_to_array(fin)
            C.ntime = parse(I64, arr[3])
            arr = line_to_array(fin)
            C.ntau = parse(I64, arr[3])
            #
            arr = line_to_array(fin)
            C.ndim1 = parse(I64, arr[3])
            arr = line_to_array(fin)
            C.ndim2 = parse(I64, arr[3])
            #
            arr = line_to_array(fin)
            C.tmax = parse(F64, arr[3])
            arr = line_to_array(fin)
            C.beta = parse(F64, arr[3])
            #
            arr = line_to_array(fin)
            C.dt = parse(F64, arr[3])
            arr = line_to_array(fin)
            C.dtau = parse(F64, arr[3])
        end
    else
        error("The $fname file doesn't exist!")
    end
end

"""
    Base.write(fname::AbstractString, C::Cn)

Write `Cn` struct to disk file. Note that the file format is defined at
`Base.show(io::IO, C::Cn)`.

See also: [`Cn`](@ref).
"""
function Base.write(fname::AbstractString, C::Cn)
    open(fname, "w") do fout
        println(fout, C)
    end
end

#=
### *Cf* : *I/O*
=#

"""
    Base.show(io::IO, cf::Cf{T})

Display `Cf` struct on the io stream.

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
        @printf(io, "%4i :\n", i)
        for m = 1:cf.ndim1
            for n = 1:cf.ndim2
                v = cf.data[i][n,m]
                if T == F64
                    @printf(io, "  %4i %4i %16.12f\n", n, m, v)
                elseif T == C64
                    @printf(io, "  %4i %4i %16.12f %16.12f\n", n, m, real(v), imag(v))
                else
                    error("The datatype $T is unsupported!")
                end
            end
        end
    end
end

"""
    Base.read!(fname::AbstractString, cf::Cf{T})

Extract data from disk file, and then use them to initialize the given
`Cf` struct.

See also: [`Cf`](@ref).
"""
function Base.read!(fname::AbstractString, cf::Cf{T}) where {T}
    if isfile(fname)
        open(fname, "r") do fin
            readline(fin) # Skip the comment line
            #
            # Extract parameters
            arr = line_to_array(fin)
            cf.ntime = parse(I64, arr[3])
            arr = line_to_array(fin)
            cf.ndim1 = parse(I64, arr[3])
            arr = line_to_array(fin)
            cf.ndim2 = parse(I64, arr[3])
            #
            readline(fin) # Skip the comment line
            #
            # Prepare memory
            element = fill(zero(T), cf.ndim1, cf.ndim2)
            empty!(cf.data)
            #
            # Extract function data
            for i = 1:getsize(cf) + 1
                readline(fin) # Skip the comment line
                #
                for m = 1:cf.ndim1
                    for n = 1:cf.ndim2
                        if T == F64
                            arr = line_to_array(fin)
                            element[n,m] = parse(F64, arr[3])
                        elseif T == C64
                            arr = line_to_array(fin)
                            element[n,m] = parse(F64, arr[3]) + parse(F64, arr[4]) * im
                        else
                            error("The datatype $T is unsupported!")
                        end
                    end
                end
                #
                push!(cf.data, copy(element))
            end
        end
        @show cf
    else
        error("The $fname file doesn't exist!")
    end
end

"""
    Base.write(fname::AbstractString, cf::Cf{T})

Write `Cf` struct to disk file. Note that the file format is defined at
`Base.show(io::IO, cf::Cf{T})`.

See also: [`Cf`](@ref).
"""
function Base.write(fname::AbstractString, cf::Cf{T}) where {T}
    open(fname, "w") do fout
        println(fout, cf)
    end
end

#=
### *Gᵐᵃᵗ* : *I/O*
=#

"""
    Base.show(io::IO, mat::Gᵐᵃᵗ{T})

Display `Gᵐᵃᵗ` struct on the io stream. Here `Gᵐᵃᵗ` means the Matsubara
component of contour-ordered Green's function.

See also: [`Gᵐᵃᵗ`](@ref).
"""
function Base.show(io::IO, mat::Gᵐᵃᵗ{T}) where {T}
    println(io, "# Contour Green's Function: Matsubara Component (G)")
    #
    println(io, "type  : ", mat.type)
    println(io, "ntau  : ", mat.ntau)
    println(io, "ndim1 : ", mat.ndim1)
    println(io, "ndim2 : ", mat.ndim2)
    #
    println(io, "data  : ")
    #
    for i = 1:getsize(mat)
        @printf(io, "%4i :\n", i)
        for m = 1:mat.ndim1
            for n = 1:mat.ndim2
                v = mat.data[i,1][n,m]
                if T == F64
                    @printf(io, "  %4i %4i %16.12f\n", n, m, v)
                elseif T == C64
                    @printf(io, "  %4i %4i %16.12f %16.12f\n", n, m, real(v), imag(v))
                else
                    error("The datatype $T is unsupported!")
                end
            end
        end
    end
end

"""
    Base.read!(fname::AbstractString, mat::Gᵐᵃᵗ{T})

Extract data from disk file, and then use them to initialize the given
`Gᵐᵃᵗ` struct.

See also: [`Gᵐᵃᵗ`](@ref).
"""
function Base.read!(fname::AbstractString, mat::Gᵐᵃᵗ{T}) where {T}
    if isfile(fname)
        # TODO
    else
        error("The $fname file doesn't exist!")
    end
end

"""
    Base.write(fname::AbstractString, mat::Gᵐᵃᵗ{T})

Write `Gᵐᵃᵗ` struct to disk file. Note that the file format is defined at
`Base.show(io::IO, mat::Gᵐᵃᵗ{T})`.

See also: [`Gᵐᵃᵗ`](@ref).
"""
function Base.write(fname::AbstractString, mat::Gᵐᵃᵗ{T}) where {T}
    open(fname, "w") do fout
        println(fout, mat)
    end
end

#=
### *Gʳᵉᵗ* : *I/O*
=#

"""
    Base.show(io::IO, ret::Gʳᵉᵗ{T})

Display `Gʳᵉᵗ` struct on the io stream. Here `Gʳᵉᵗ` means the retarded
component of contour-ordered Green's function.

See also: [`Gʳᵉᵗ`](@ref).
"""
function Base.show(io::IO, ret::Gʳᵉᵗ{T}) where {T}
    println(io, "# Contour Green's Function: Retarded Component (G)")
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
            @printf(io, "%4i %4i :\n", j, i)
            for m = 1:ret.ndim1
                for n = 1:ret.ndim2
                    v = ret.data[j,i][n,m]
                    if T == F64
                        @printf(io, "  %4i %4i %16.12f\n", n, m, v)
                    elseif T == C64
                        @printf(io, "  %4i %4i %16.12f %16.12f\n", n, m, real(v), imag(v))
                    else
                        error("The datatype $T is unsupported!")
                    end
                end
            end
        end
    end
end

"""
    Base.read!(fname::AbstractString, ret::Gʳᵉᵗ{T})

Extract data from disk file, and then use them to initialize the given
`Gʳᵉᵗ` struct.

See also: [`Gʳᵉᵗ`](@ref).
"""
function Base.read!(fname::AbstractString, ret::Gʳᵉᵗ{T}) where {T}
    if isfile(fname)
        # TODO
    else
        error("The $fname file doesn't exist!")
    end
end

"""
    Base.write(fname::AbstractString, ret::Gʳᵉᵗ{T})

Write `Gʳᵉᵗ` struct to disk file. Note that the file format is defined at
`Base.show(io::IO, ret::Gʳᵉᵗ{T})`.

See also: [`Gʳᵉᵗ`](@ref).
"""
function Base.write(fname::AbstractString, ret::Gʳᵉᵗ{T}) where {T}
    open(fname, "w") do fout
        println(fout, ret)
    end
end

#=
### *Gˡᵐⁱˣ* : *I/O*
=#

"""
    Base.show(io::IO, lmix::Gˡᵐⁱˣ{T})

Display `Gˡᵐⁱˣ` struct on the io stream. Here `Gˡᵐⁱˣ` means the left-mixing
component of contour-ordered Green's function.

See also: [`Gˡᵐⁱˣ`](@ref).
"""
function Base.show(io::IO, lmix::Gˡᵐⁱˣ{T}) where {T}
    println(io, "# Contour Green's Function: Left-Mixing Component (G)")
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
            @printf(io, "%4i %4i :\n", j, i)
            for m = 1:lmix.ndim1
                for n = 1:lmix.ndim2
                    v = lmix.data[j,i][n,m]
                    if T == F64
                        @printf(io, "  %4i %4i %16.12f\n", n, m, v)
                    elseif T == C64
                        @printf(io, "  %4i %4i %16.12f %16.12f\n", n, m, real(v), imag(v))
                    else
                        error("The datatype $T is unsupported!")
                    end
                end
            end
        end
    end
end

"""
    Base.read!(fname::AbstractString, lmix::Gˡᵐⁱˣ{T})

Extract data from disk file, and then use them to initialize the given
`Gˡᵐⁱˣ` struct.

See also: [`Gˡᵐⁱˣ`](@ref).
"""
function Base.read!(fname::AbstractString, lmix::Gˡᵐⁱˣ{T}) where {T}
    if isfile(fname)
        # TODO
    else
        error("The $fname file doesn't exist!")
    end
end

"""
    Base.write(fname::AbstractString, lmix::Gˡᵐⁱˣ{T})

Write `Gˡᵐⁱˣ` struct to disk file. Note that the file format is defined at
`Base.show(io::IO, lmix::Gˡᵐⁱˣ{T})`.

See also: [`Gˡᵐⁱˣ`](@ref).
"""
function Base.write(fname::AbstractString, lmix::Gˡᵐⁱˣ{T}) where {T}
    open(fname, "w") do fout
        println(fout, lmix)
    end
end

#=
### *Gˡᵉˢˢ* : *I/O*
=#

"""
    Base.show(io::IO, less::Gˡᵉˢˢ{T})

Display `Gˡᵉˢˢ` struct on the io stream. Here `Gˡᵉˢˢ` means the lesser
component of contour-ordered Green's function.

See also: [`Gˡᵉˢˢ`](@ref).
"""
function Base.show(io::IO, less::Gˡᵉˢˢ{T}) where {T}
    println(io, "# Contour Green's Function: Lesser Component (G)")
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
            @printf(io, "%4i %4i :\n", j, i)
            for m = 1:less.ndim1
                for n = 1:less.ndim2
                    v = less.data[j,i][n,m]
                    if T == F64
                        @printf(io, "  %4i %4i %16.12f\n", n, m, v)
                    elseif T == C64
                        @printf(io, "  %4i %4i %16.12f %16.12f\n", n, m, real(v), imag(v))
                    else
                        error("The datatype $T is unsupported!")
                    end
                end
            end
        end
    end
end

"""
    Base.read!(fname::AbstractString, less::Gˡᵉˢˢ{T})

Extract data from disk file, and then use them to initialize the given
`Gˡᵉˢˢ` struct.

See also: [`Gˡᵉˢˢ`](@ref).
"""
function Base.read!(fname::AbstractString, less::Gˡᵉˢˢ{T}) where {T}
    if isfile(fname)
        # TODO
    else
        error("The $fname file doesn't exist!")
    end
end

"""
    Base.write(fname::AbstractString, less::Gˡᵉˢˢ{T})

Write `Gˡᵉˢˢ` struct to disk file. Note that the file format is defined at
`Base.show(io::IO, less::Gˡᵉˢˢ{T})`.

See also: [`Gˡᵉˢˢ`](@ref).
"""
function Base.write(fname::AbstractString, less::Gˡᵉˢˢ{T}) where {T}
    open(fname, "w") do fout
        println(fout, less)
    end
end

#=
### *gᵐᵃᵗ* : *I/O*
=#

"""
    Base.show(io::IO, mat::gᵐᵃᵗ{S})

Display `gᵐᵃᵗ` struct on the io stream. Here `gᵐᵃᵗ` means the Matsubara
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
        @printf(io, "%4i :\n", i)
        for m = 1:mat.ndim1
            for n = 1:mat.ndim2
                v = mat.data[i][n,m]
                if S == F64
                    @printf(io, "  %4i %4i %16.12f\n", n, m, v)
                elseif S == C64
                    @printf(io, "  %4i %4i %16.12f %16.12f\n", n, m, real(v), imag(v))
                else
                    error("The datatype $S is unsupported!")
                end
            end
        end
    end
end

"""
    Base.read!(fname::AbstractString, mat::gᵐᵃᵗ{S})

Extract data from disk file, and then use them to initialize the given
`gᵐᵃᵗ` struct.

See also: [`gᵐᵃᵗ`](@ref).
"""
function Base.read!(fname::AbstractString, mat::gᵐᵃᵗ{S}) where {S}
    if isfile(fname)
        # TODO
    else
        error("The $fname file doesn't exist!")
    end
end

"""
    Base.write(fname::AbstractString, mat::gᵐᵃᵗ{S})

Write `gᵐᵃᵗ` struct to disk file. Note that the file format is defined at
`Base.show(io::IO, mat::gᵐᵃᵗ{T})`.

See also: [`gᵐᵃᵗ`](@ref).
"""
function Base.write(fname::AbstractString, mat::gᵐᵃᵗ{S}) where {S}
    open(fname, "w") do fout
        println(fout, mat)
    end
end

#=
### *gʳᵉᵗ* : *I/O*
=#

"""
    Base.show(io::IO, ret::gʳᵉᵗ{S})

Display `gʳᵉᵗ` struct on the io stream. Here `gʳᵉᵗ` means the retarded
component of contour-ordered Green's function.

See also: [`gʳᵉᵗ`](@ref).
"""
function Base.show(io::IO, ret::gʳᵉᵗ{S}) where {S}
    println(io, "# Contour Green's Function: Retarded Component (g)")
    #
    println(io, "type  : ", ret.type)
    println(io, "tstp  : ", ret.tstp)
    println(io, "ndim1 : ", ret.ndim1)
    println(io, "ndim2 : ", ret.ndim2)
    #
    println(io, "data  : ")
    #
    for i = 1:getsize(ret)
        @printf(io, "%4i :\n", i)
        for m = 1:ret.ndim1
            for n = 1:ret.ndim2
                v = ret.data[i][n,m]
                if S == F64
                    @printf(io, "  %4i %4i %16.12f\n", n, m, v)
                elseif S == C64
                    @printf(io, "  %4i %4i %16.12f %16.12f\n", n, m, real(v), imag(v))
                else
                    error("The datatype $S is unsupported!")
                end
            end
        end
    end
end

"""
    Base.read!(fname::AbstractString, ret::gʳᵉᵗ{S})

Extract data from disk file, and then use them to initialize the given
`gʳᵉᵗ` struct.

See also: [`gʳᵉᵗ`](@ref).
"""
function Base.read!(fname::AbstractString, ret::gʳᵉᵗ{S}) where {S}
    if isfile(fname)
        # TODO
    else
        error("The $fname file doesn't exist!")
    end
end

"""
    Base.write(fname::AbstractString, ret::gʳᵉᵗ{S})

Write `gʳᵉᵗ` struct to disk file. Note that the file format is defined at
`Base.show(io::IO, ret::gʳᵉᵗ{T})`.

See also: [`gʳᵉᵗ`](@ref).
"""
function Base.write(fname::AbstractString, ret::gʳᵉᵗ{S}) where {S}
    open(fname, "w") do fout
        println(fout, ret)
    end
end

#=
### *gˡᵐⁱˣ* : *I/O*
=#

"""
    Base.show(io::IO, lmix::gˡᵐⁱˣ{S})

Display `gˡᵐⁱˣ` struct on the io stream. Here `gˡᵐⁱˣ` means the left-mixing
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
        @printf(io, "%4i :\n", i)
        for m = 1:lmix.ndim1
            for n = 1:lmix.ndim2
                v = lmix.data[i][n,m]
                if S == F64
                    @printf(io, "  %4i %4i %16.12f\n", n, m, v)
                elseif S == C64
                    @printf(io, "  %4i %4i %16.12f %16.12f\n", n, m, real(v), imag(v))
                else
                    error("The datatype $S is unsupported!")
                end
            end
        end
    end
end

"""
    Base.read!(fname::AbstractString, lmix::gˡᵐⁱˣ{S})

Extract data from disk file, and then use them to initialize the given
`gˡᵐⁱˣ` struct.

See also: [`gˡᵐⁱˣ`](@ref).
"""
function Base.read!(fname::AbstractString, lmix::gˡᵐⁱˣ{S}) where {S}
    if isfile(fname)
        # TODO
    else
        error("The $fname file doesn't exist!")
    end
end

"""
    Base.write(fname::AbstractString, lmix::gˡᵐⁱˣ{S})

Write `gˡᵐⁱˣ` struct to disk file. Note that the file format is defined at
`Base.show(io::IO, lmix::gˡᵐⁱˣ{T})`.

See also: [`gˡᵐⁱˣ`](@ref).
"""
function Base.write(fname::AbstractString, lmix::gˡᵐⁱˣ{S}) where {S}
    open(fname, "w") do fout
        println(fout, lmix)
    end
end

#=
### *gˡᵉˢˢ* : *I/O*
=#

"""
    Base.show(io::IO, less::gˡᵉˢˢ{S})

Display `gˡᵉˢˢ` struct on the io stream. Here `gˡᵉˢˢ` means the lesser
component of contour-ordered Green's function.

See also: [`gˡᵉˢˢ`](@ref).
"""
function Base.show(io::IO, less::gˡᵉˢˢ{S}) where {S}
    println(io, "# Contour Green's Function: Lesser Component (g)")
    #
    println(io, "type  : ", less.type)
    println(io, "tstp  : ", less.tstp)
    println(io, "ndim1 : ", less.ndim1)
    println(io, "ndim2 : ", less.ndim2)
    #
    println(io, "data  : ")
    #
    for i = 1:getsize(less)
        @printf(io, "%4i :\n", i)
        for m = 1:less.ndim1
            for n = 1:less.ndim2
                v = less.data[i][n,m]
                if S == F64
                    @printf(io, "  %4i %4i %16.12f\n", n, m, v)
                elseif S == C64
                    @printf(io, "  %4i %4i %16.12f %16.12f\n", n, m, real(v), imag(v))
                else
                    error("The datatype $T is unsupported!")
                end
            end
        end
    end
end

"""
    Base.read!(fname::AbstractString, less::gˡᵉˢˢ{S})

Extract data from disk file, and then use them to initialize the given
`gˡᵉˢˢ` struct.

See also: [`gˡᵉˢˢ`](@ref).
"""
function Base.read!(fname::AbstractString, less::gˡᵉˢˢ{S}) where {S}
    if isfile(fname)
        # TODO
    else
        error("The $fname file doesn't exist!")
    end
end

"""
    Base.write(fname::AbstractString, less::gˡᵉˢˢ{S})

Write `gˡᵉˢˢ` struct to disk file. Note that the file format is defined at
`Base.show(io::IO, less::gˡᵉˢˢ{T})`.

See also: [`gˡᵉˢˢ`](@ref).
"""
function Base.write(fname::AbstractString, less::gˡᵉˢˢ{S}) where {S}
    open(fname, "w") do fout
        println(fout, less)
    end
end

#=
### *ℱ* : *I/O*
=#

"""
    Base.show(io::IO, cfm::ℱ{T})

Display `ℱ` struct on the io stream. Here `ℱ` means the standard contour-
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
    read!(fname::AbstractString, cfm::ℱ{T})

Read the contour-ordered Green's functions from given file.

See also: [`ℱ`](@ref).
"""
function Base.read!(fname::AbstractString, cfm::ℱ{T}) where {T}
    if isfile(fname)
        # TODO
    else
        error("The $fname file doesn't exist!")
    end
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

#=
### *𝒻* : *I/O*
=#

"""
    Base.show(io::IO, cfv::𝒻{S}) where {S}

See also: [`𝒻`](@ref).
"""
function Base.show(io::IO, cfv::𝒻{S}) where {S}
    sorry()
end

"""
    read!(fname::AbstractString, cfv::𝒻{S})

Read the contour-ordered Green's functions from given file.

See also: [`𝒻`](@ref).
"""
function Base.read!(fname::AbstractString, cfv::𝒻{S}) where {S}
    if isfile(fname)
        # TODO
    else
        error("The $fname file doesn't exist!")
    end
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
