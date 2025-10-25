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
    println(io, "data:")
    for i = 1:getntau(mat)
        println(io, i, " ", mat.data[i,1])
    end
end

#=
### *Gʳᵉᵗ* : *I/O*
=#

function Base.show(io::IO, ret::Gʳᵉᵗ{T}) where {T}
end

#=
### *Gˡᵐⁱˣ* : *I/O*
=#

function Base.show(io::IO, lmix::Gˡᵐⁱˣ{T}) where {T}
end

#=
### *Gˡᵉˢˢ* : *I/O*
=#

function Base.show(io::IO, less::Gˡᵉˢˢ{T}) where {T}
end

#=
### *gᵐᵃᵗ* : *I/O*
=#

function Base.show(io::IO, mat::gᵐᵃᵗ{S}) where {S}
end

#=
### *gʳᵉᵗ* : *I/O*
=#

function Base.show(io::IO, ret::gʳᵉᵗ{S}) where {S}
end

#=
### *gˡᵐⁱˣ* : *I/O*
=#

function Base.show(io::IO, lmix::gˡᵐⁱˣ{S}) where {S}
end

#=
### *gˡᵉˢˢ* : *I/O*
=#

function Base.show(io::IO, less::gˡᵉˢˢ{S}) where {S}
end

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
