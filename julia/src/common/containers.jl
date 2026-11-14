#
# Project : Lavender
# Source  : containers.jl
# Author  : Li Huang (huangli@caep.cn)
# Status  : Unstable
#
# Last modified: 2026/11/10
#

#=
### *Derived Types*
=#

"""
    Element{T}

Type definition. A matrix.
"""
const Element{T} = Array{T,2}

"""
    MatArray{T}

Type definition. A matrix of matrix.
"""
const MatArray{T} = Matrix{Element{T}}

"""
    VecArray{T}

Type definition. A vector of matrix.
"""
const VecArray{T} = Vector{Element{T}}
