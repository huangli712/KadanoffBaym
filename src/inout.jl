#
# Project : Lavender
# Source  : inout.jl
# Author  : Li Huang (huangli@caep.cn)
# Status  : Unstable
#
# Last modified: 2025/10/25
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

See also: [`Cn`](@ref)
"""
function Base.read!(fname::AbstractString, C::Cn)
    sorry()
end

"""
    Base.write(fname::AbstractString, C::Cn)

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
    println(io, "data  : ")
    #
    for i = 1:getntime(cf) + 1
        @printf(io, "%4i :", i)
        for m = 1:cf.ndim1
            for n = 1:cf.ndim2
                v = cf.data[i][n,m]
                print(io, " $v " )
            end
        end
        println(io)
    end
end

"""
    Base.read!(fname::AbstractString, cf::Cf{T})

See also: [`Cf`](@ref).
"""
function Base.read!(fname::AbstractString, cf::Cf{T}) where {T}
    sorry()
end

"""
    Base.write(fname::AbstractString, cf::Cf{T}) where {T}

See also: [`Cf`](@ref).
"""
function Base.write(fname::AbstractString, cf::Cf{T}) where {T}
    sorry()
end

#=
### *Gᵐᵃᵗ* : *I/O*
=#

"""
    Base.show(io::IO, mat::Gᵐᵃᵗ{T})

Display `Gᵐᵃᵗ` struct on the io stream. Here `Gᵐᵃᵗ` means the Matsubara
component of contour Green's function.

See also: [`Gᵐᵃᵗ`](@ref).
"""
function Base.show(io::IO, mat::Gᵐᵃᵗ{T}) where {T}
    println(io, "# Contour Green's Function: Matsubara Component")
    #
    println(io, "type  : ", mat.type)
    println(io, "ntau  : ", mat.ntau)
    println(io, "ndim1 : ", mat.ndim1)
    println(io, "ndim2 : ", mat.ndim2)
    #
    println(io, "data  : ")
    for i = 1:getntau(mat)
        println(io, i, " ", mat.data[i,1])
    end
end

"""
    Base.read!(fname::AbstractString, mat::Gᵐᵃᵗ{T}) where {T}

See also: [`Gᵐᵃᵗ`](@ref).
"""
function Base.read!(fname::AbstractString, mat::Gᵐᵃᵗ{T}) where {T}
    sorry()
end

"""
    Base.write(fname::AbstractString, mat::Gᵐᵃᵗ{T}) where {T}

See also: [`Gᵐᵃᵗ`](@ref).
"""
function Base.write(fname::AbstractString, mat::Gᵐᵃᵗ{T}) where {T}
    sorry()
end

#=
### *Gʳᵉᵗ* : *I/O*
=#

"""
    Base.show(io::IO, ret::Gʳᵉᵗ{T})

Display `Gʳᵉᵗ` struct on the io stream. Here `Gʳᵉᵗ` means the retarded
component of contour Green's function.

See also: [`Gʳᵉᵗ`](@ref).
"""
function Base.show(io::IO, ret::Gʳᵉᵗ{T}) where {T}
end

function Base.read!(fname::AbstractString, ret::Gʳᵉᵗ{T}) where {T}
    sorry()
end

function Base.write(fname::AbstractString, ret::Gʳᵉᵗ{T}) where {T}
    sorry()
end

#=
### *Gˡᵐⁱˣ* : *I/O*
=#

"""
    Base.show(io::IO, lmix::Gˡᵐⁱˣ{T})

Display `Gˡᵐⁱˣ` struct on the io stream. Here `Gˡᵐⁱˣ` means the left-mixing
component of contour Green's function.

See also: [`Gˡᵐⁱˣ`](@ref).
"""
function Base.show(io::IO, lmix::Gˡᵐⁱˣ{T}) where {T}
end

function Base.read!(fname::AbstractString, lmix::Gˡᵐⁱˣ{T}) where {T}
    sorry()
end

function Base.write(fname::AbstractString, lmix::Gˡᵐⁱˣ{T}) where {T}
    sorry()
end

#=
### *Gˡᵉˢˢ* : *I/O*
=#

"""
    Base.show(io::IO, less::Gˡᵉˢˢ{T})

Display `Gˡᵉˢˢ` struct on the io stream. Here `Gˡᵉˢˢ` means the lesser
component of contour Green's function.

See also: [`Gˡᵉˢˢ`](@ref).
"""
function Base.show(io::IO, less::Gˡᵉˢˢ{T}) where {T}
end

function Base.read!(fname::AbstractString, less::Gˡᵉˢˢ{T}) where {T}
    sorry()
end

function Base.write(fname::AbstractString, less::Gˡᵉˢˢ{T}) where {T}
    sorry()
end

#=
### *gᵐᵃᵗ* : *I/O*
=#

"""
    Base.show(io::IO, mat::gᵐᵃᵗ{S})

Display `gᵐᵃᵗ` struct on the io stream. Here `gᵐᵃᵗ` means the Matsubara
component of contour Green's function.

See also: [`gᵐᵃᵗ`](@ref).
"""
function Base.show(io::IO, mat::gᵐᵃᵗ{S}) where {S}
end

function Base.read!(fname::AbstractString, mat::gᵐᵃᵗ{S}) where {S}
    sorry()
end

function Base.write(fname::AbstractString, mat::gᵐᵃᵗ{S}) where {S}
    sorry()
end

#=
### *gʳᵉᵗ* : *I/O*
=#

"""
    Base.show(io::IO, ret::gʳᵉᵗ{S})

Display `gʳᵉᵗ` struct on the io stream. Here `gʳᵉᵗ` means the retarded
component of contour Green's function.

See also: [`gʳᵉᵗ`](@ref).
"""
function Base.show(io::IO, ret::gʳᵉᵗ{S}) where {S}
end

function Base.read!(fname::AbstractString, ret::gʳᵉᵗ{S}) where {S}
    sorry()
end

function Base.write(fname::AbstractString, ret::gʳᵉᵗ{S}) where {S}
    sorry()
end

#=
### *gˡᵐⁱˣ* : *I/O*
=#

"""
    Base.show(io::IO, lmix::gˡᵐⁱˣ{S})

Display `gˡᵐⁱˣ` struct on the io stream. Here `gˡᵐⁱˣ` means the left-mixing
component of contour Green's function.

See also: [`gˡᵐⁱˣ`](@ref).
"""
function Base.show(io::IO, lmix::gˡᵐⁱˣ{S}) where {S}
end

function Base.read!(fname::AbstractString, lmix::gˡᵐⁱˣ{S}) where {S}
    sorry()
end

function Base.write(fname::AbstractString, lmix::gˡᵐⁱˣ{S}) where {S}
    sorry()
end

#=
### *gˡᵉˢˢ* : *I/O*
=#

"""
    Base.show(io::IO, less::gˡᵉˢˢ{S})

Display `gˡᵉˢˢ` struct on the io stream. Here `gˡᵉˢˢ` means the lesser
component of contour Green's function.

See also: [`gˡᵉˢˢ`](@ref).
"""
function Base.show(io::IO, less::gˡᵉˢˢ{S}) where {S}
end

function Base.read!(fname::AbstractString, less::gˡᵉˢˢ{S}) where {S}
    sorry()
end

function Base.write(fname::AbstractString, less::gˡᵉˢˢ{S}) where {S}
    sorry()
end

#=
### *ℱ* : *I/O*
=#

function Base.show(io::IO, cfm::ℱ{T}) where {T}
    sorry()
end

"""
    read!(fname::AbstractString, cfm::ℱ{T})

Read the contour Green's functions from given file.
"""
function Base.read!(fname::AbstractString, cfm::ℱ{T}) where {T}
    sorry()
end

"""
    write(fname::AbstractString, cfm::ℱ{T})

Write the contour Green's functions to given file.
"""
function Base.write(fname::AbstractString, cfm::ℱ{T}) where {T}
    sorry()
end

#=
### *𝒻* : *I/O*
=#

function Base.show(io::IO, cfv::𝒻{S}) where {S}
    sorry()
end

"""
    read!(fname::AbstractString, cfv::𝒻{S})

Read the contour Green's functions from given file.
"""
function Base.read!(fname::AbstractString, cfv::𝒻{S}) where {S}
    sorry()
end

"""
    write(fname::AbstractString, cfv::𝒻{S})

Write the contour Green's functions to given file.
"""
function Base.write(fname::AbstractString, cfv::𝒻{S}) where {S}
    sorry()
end
