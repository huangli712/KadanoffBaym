#
# Project : Lavender
# Source  : types.jl
# Author  : Li Huang (huangli@caep.cn)
# Status  : Unstable
#
# Last modified: 2025/10/15
#

#=
### *Customized Types*
=#

"Customized types. It is used to define the following dicts."
const DType = Any

"Customized types. It is used to define the following dicts."
const ADT = Array{DType,1}

#=
### *Customized Dictionaries*
=#

#=
*Remarks* :

The values in the following dictionaries are actually arrays, which
usually contain four elements:
* Element[1] -> Actually value.
* Element[2] -> If it is 1, this key-value pair is mandatory.
                If it is 0, this key-value pair is optional.
* Element[3] -> Numerical type (A julia Symbol).
* Element[4] -> Brief explanations.

The following dictionaries are used as global variables.
=#

"""
    PBASE

Dictionary for configuration parameters: general setup.
"""
const PBASE    = Dict{String,ADT}(
    "ntime" => [missing, 1, :I64   , "Number of time slices in real time axis"],
    "ntau"  => [missing, 1, :I64   , "Number of time slices in imaginary time axis"],
    "ndim1" => [missing, 1, :I64   , "Size of operators that stored in the contour"],
    "ndim2" => [missing, 1, :I64   , "Size of operators that stored in the contour"],
    "tmax"  => [missing, 1, :F64   , "Maximum time in real time axis"],
    "beta"  => [missing, 1, :F64   , "Inverse temperature"],
    "dt"    => [missing, 1, :F64   , "time step in real time axis"],
    "dtau"  => [missing, 1, :F64   , "time step in imaginary time axis"],
)

# Default parameters for PBASE
const _PBASE   = Dict{String,Any}(
    "ntime" => 201,
    "ntau"  => 1001,
    "ndim1" => 1,
    "ndim2" => 1,
    "tmax"  => 4.0,
    "beta"  => 8.0,
    "dt"    => 0.02,
    "dtau"  => 0.008,
)

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

#=
### *Abstract Types*
=#

#=
*Remarks* :

We need a few abstract types to construct the type systems.These abstract
types include:

* *CnAbstractType*
* *CnAbstractMatrix*
* *CnAbstractVector*
* *CnAbstractFunction*

They should not be used in the user's applications directly.
=#

"""
    CnAbstractType

Top abstract type for all objects defined on contour.
"""
abstract type CnAbstractType end

"""
    CnAbstractMatrix{T}

Abstract matrix type defined on contour.
"""
abstract type CnAbstractMatrix{T} <: CnAbstractType end

"""
    CnAbstractVector{T}

Abstract vector type defined on contour.
"""
abstract type CnAbstractVector{T} <: CnAbstractType end

"""
    CnAbstractFunction{T}

Abstract contour function.
"""
abstract type CnAbstractFunction{T} <: CnAbstractType end

"""
    subtypetree(roottype, level::I64 = 1, indent::I64 = 4)

Display the entire type hierarchy starting from the specified `roottype`
"""
function subtypetree(roottype, level::I64 = 1, indent::I64 = 4)
    level == 1 && println(roottype)
    for s in subtypes(roottype)
        println(join(fill(" ", level * indent)) * string(s))
        subtypetree(s, level + 1, indent)
    end
end
