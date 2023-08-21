#
# File: const.jl
#
# Define some global constants and alias of datatypes.
#

#=
### *Intrinsic Types*
=#

"""
    I64

Basic numerical type.
"""
const I64 = Int64

"""
    F64

Basic numerical type.
"""
const F64 = Float64

"""
    C64

Basic numerical type.
"""
const C64 = ComplexF64

#=
### *Basic Constants*
=#

"""
    FERMI

Basic physical constant.
"""
const FERMI = -1

"""
    BOSE

Basic physical constant.
"""
const BOSE = 1

"""
    CZERO

Basic numerical constant.
"""
const CZERO = 0.0 + 0.0im

"""
    CZI

Basic numerical constant.
"""
const CZI = 0.0 + 1.0im

"""
    CONE

Basic numerical constant.
"""
const CONE = 1.0 + 0.0im

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