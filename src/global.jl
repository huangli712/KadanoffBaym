#
# Project : Lavender
# Source  : global.jl
# Author  : Li Huang (huangli@caep.cn)
# Status  : Unstable
#
# Last modified: 2024/08/12
#

#=
### *Global Constants* : *Numerical Types*
=#

"""
    I64

Alias of Integer type (64 bit).
"""
const I64 = Int64

"""
    F64

Alias of Float type (64 bit).
"""
const F64 = Float64

"""
    C64

Alias of Complex type (64 bit).
"""
const C64 = ComplexF64

#=
### *Global Constants* : *Basic Constants*
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
### *Global Constants* : *Derived Types*
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
### *Global Constants* : *Literal Strings*
=#

"""
    __LIBNAME__

Name of this julia toolkit.

See also: [`__VERSION__`](@ref).
"""
const __LIBNAME__ = "KadanoffBaym"

"""
    __VERSION__

Version of this julia toolkit.

See also: [`__RELEASE__`](@ref).
"""
const __VERSION__ = v"0.0.1-devel.240811"

"""
    __RELEASE__

Release date of this julia toolkit.

See also: [`__AUTHORS__`](@ref).
"""
const __RELEASE__ = "2024/08"

#=
*Remarks* :

The elements of the Array `__AUTHORS__` should be a `NamedTuple` object,
such as:

```julia
(name = "author's name", email = "author's email")
```
=#

"""
    __AUTHORS__

Core authors of this julia toolkit.

See also: [`__LIBNAME__`](@ref).
"""
const __AUTHORS__ = [(name = "Li Huang", email = "huangli@caep.cn")]

"""
    authors()

Print authors / contributors of the `ACFlow` toolkit.

See also: [`__AUTHORS__`](@ref).
"""
function authors()
    println("Authors (Until $__RELEASE__):")
    for a in __AUTHORS__
        println("  $(a.name) (email: $(a.email))")
    end
end
