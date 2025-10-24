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

Display `Cn` struct on the terminal.

See also: [`Cn`](@ref).
"""
function Base.show(io::IO, C::Cn)
    println(io, "ntime : ", C.ntime)
    println(io, "ntau  : ", C.ntau )
    println(io, "ndim1 : ", C.ndim1)
    println(io, "ndim2 : ", C.ndim2)
    println(io, "tmax  : ", C.tmax )
    println(io, "beta  : ", C.beta )
    println(io, "dt    : ", C.dt   )
    println(io, "dtau  : ", C.dtau )
end

#=
### *Cf* : *I/O*
=#

function Base.show(io::IO, cf::Cf{T}) where {T}
end

#=
### *Gᵐᵃᵗ* : *I/O*
=#

function Base.show(io::IO, mat::Gᵐᵃᵗ{T}) where {T}
    println(io, "type: ", mat.type)
    println(io, "ntau: ", mat.ntau)
    println(io, "ndim1:", mat.ndim1)
    println(io, "ndim2:", mat.ndim2)
    println(io, "data:", mat.data)
end

#=
### *Gʳᵉᵗ* : *I/O*
=#

#=
### *Gˡᵐⁱˣ* : *I/O*
=#

#=
### *Gˡᵉˢˢ* : *I/O*
=#

#=
### *gᵐᵃᵗ* : *I/O*
=#

#=
### *gʳᵉᵗ* : *I/O*
=#

#=
### *gˡᵐⁱˣ* : *I/O*
=#

#=
### *gˡᵉˢˢ* : *I/O*
=#

#=
### *ℱ* : *I/O*
=#

#=
### *ℱ* : *I/O*
=#

"""
    read!(fname::AbstractString, cfm::ℱ{T})

Read the contour Green's functions from given file.
"""
function read!(fname::AbstractString, cfm::ℱ{T}) where {T}
    sorry()
end

"""
    write(fname::AbstractString, cfm::ℱ{T})

Write the contour Green's functions to given file.
"""
function write(fname::AbstractString, cfm::ℱ{T}) where {T}
    sorry()
end

#=
### *𝒻* : *I/O*
=#

"""
    read!(fname::AbstractString, cfv::𝒻{S})

Read the contour Green's functions from given file.
"""
function read!(fname::AbstractString, cfv::𝒻{S}) where {S}
    sorry()
end

"""
    write(fname::AbstractString, cfv::𝒻{S})

Write the contour Green's functions to given file.
"""
function write(fname::AbstractString, cfv::𝒻{S}) where {S}
    sorry()
end
