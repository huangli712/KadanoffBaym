#
# Project : Lavender
# Source  : types.jl
# Author  : Li Huang (huangli@caep.cn)
# Status  : Unstable
#
# Last modified: 2025/10/20
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
    PCONTOUR

Dictionary for configuration parameters: contour setup.
"""
const PCONTOUR = Dict{String,ADT}(
    "ntime" => [missing, 1, :I64   , "Number of time slices in real time axis"],
    "ntau"  => [missing, 1, :I64   , "Number of time slices in imaginary time axis"],
    "ndim1" => [missing, 1, :I64   , "Size of operators that stored in the contour"],
    "ndim2" => [missing, 1, :I64   , "Size of operators that stored in the contour"],
    "tmax"  => [missing, 1, :F64   , "Maximum time in real time axis"],
    "beta"  => [missing, 1, :F64   , "Inverse temperature"],
)

# Default parameters for PCONTOUR
const _PCONTOUR = Dict{String,Any}(
    "ntime" => 201,
    "ntau"  => 1001,
    "ndim1" => 1,
    "ndim2" => 1,
    "tmax"  => 4.0,
    "beta"  => 8.0,
)

"""
    PMODEL

Dictionary for configuration parameters: model setup.
"""
const PMODEL = Dict{String,ADT}(
    "system" => [missing, 1, :String, "System's name"],
)

# Default parameters for PMODEL
const _PMODEL = Dict{String,Any}(
    "system" => "unknown",
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
*Remarks* : *Type System*

We need a few abstract types to construct the type systems.These abstract
types include:

* *CnAbstractType*
* *CnAbstractContour*
* *CnAbstractMatrix*
* *CnAbstractVector*
* *CnAbstractFunction*

They should not be used in the user's codes directly. The hierarchical
types appeared in the KadanoffBaym library are collected as follows:

* *CnAbstractType*
  * *CnAbstractContour*
    * *Cn*
  * *CnAbstractMatrix*
    * *Gᵐᵃᵗ*
    * *Gʳᵉᵗ*
    * *Gˡᵐⁱˣ*
    * *Gˡᵉˢˢ*
    * *Gᵐᵃᵗᵐ*
    * *Gᵃᵈᵛ*
    * *Gʳᵐⁱˣ*
    * *Gᵍᵗʳ*
  * *CnAbstractVector*
    * *gᵐᵃᵗ*
    * *gʳᵉᵗ*
    * *gˡᵐⁱˣ*
    * *gˡᵉˢˢ*
    * *gᵐᵃᵗᵐ*
    * *gᵃᵈᵛ*
    * *gʳᵐⁱˣ*
    * *gᵍᵗʳ*
  * *CnAbstractFunction*
    * *Cf*
    * *ℱ*
    * *𝒻*


The `subtypetree()` function can be used to sketch this type system.  
=#

"""
    CnAbstractType

Top abstract type for all objects defined on contour.
"""
abstract type CnAbstractType end

"""
    CnAbstractContour

Abstract contour type.
"""
abstract type CnAbstractContour <: CnAbstractType end

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

#=
### *Helper Functions*
=#

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
