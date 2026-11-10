#
# Project : Lavender
# Source  : abstract.jl
# Author  : Li Huang (huangli@caep.cn)
# Status  : Unstable
#
# Last modified: 2026/11/10
#

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
* *CnAbstractStationaryFunction*
* *CnAbstractMovingMatrix*
* *CnAbstractMovingVector*
* *CnAbstractMovingFunction*

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
    * *CnAbstractMovingMatrix*
      * *Gʳᵉᵗᵐᵒᵛ*
      * *Gˡᵉˢˢᵐᵒᵛ*
  * *CnAbstractVector*
    * *gᵐᵃᵗ*
    * *gʳᵉᵗ*
    * *gˡᵐⁱˣ*
    * *gˡᵉˢˢ*
    * *gᵐᵃᵗᵐ*
    * *gᵃᵈᵛ*
    * *gʳᵐⁱˣ*
    * *gᵍᵗʳ*
    * *CnAbstractMovingVector*
      * *gʳᵉᵗᵐᵒᵛ*
      * *gˡᵉˢˢᵐᵒᵛ*
  * *CnAbstractFunction*
    * *Cf*
    * *ℱ*
    * *𝒻*
    * *PseudoGF*
    * *PseudoGFStep*
    * *CnAbstractMovingFunction*
      * *ℱᵐᵒᵛ*
      * *𝒻ᵐᵒᵛ*
      * *𝒻ᵛⁱᵉʷᵐᵒᵛ*
      * *Cfᵐᵒᵛ*
  * *CnAbstractStationaryFunction*
    * *HermMatrixNESS*

The moving (truncated time window) types store only the real-time
components in a window of `tc + 1` time steps, see `moving.jl`.

`CnAbstractFunction` contains the functions defined on the contour,
while `CnAbstractStationaryFunction` contains the stationary functions
of the relative time or the frequency (NESS2). The numerical storage
`FFTArray` and the grid `GridInfo` are not in this type system.

The `subtypetree()` function can be used to sketch this type system.
=#

"""
    CnAbstractType

Top abstract type for all objects of the KadanoffBaym library, including
the objects defined on contour and the stationary functions of the
relative time or the frequency.
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

"""
    CnAbstractMovingMatrix{T}

Abstract matrix type defined on a truncated (moving) time window.
"""
abstract type CnAbstractMovingMatrix{T} <: CnAbstractMatrix{T} end

"""
    CnAbstractMovingVector{T}

Abstract vector type (one time step) defined on a truncated (moving)
time window.
"""
abstract type CnAbstractMovingVector{T} <: CnAbstractVector{T} end

"""
    CnAbstractMovingFunction{T}

Abstract contour function defined on a truncated (moving) time window.
"""
abstract type CnAbstractMovingFunction{T} <: CnAbstractFunction{T} end

"""
    CnAbstractStationaryFunction{T}

Abstract stationary function of the relative time or the frequency.
"""
abstract type CnAbstractStationaryFunction{T} <: CnAbstractType end
