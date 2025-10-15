#
# Project : Lavender
# Source  : types.jl
# Author  : Li Huang (huangli@caep.cn)
# Status  : Unstable
#
# Last modified: 2025/10/13
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
