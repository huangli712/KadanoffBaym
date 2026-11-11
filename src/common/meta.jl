#
# Project : Lavender
# Source  : meta.jl
# Author  : Li Huang (huangli@caep.cn)
# Status  : Unstable
#
# Last modified: 2026/11/10
#

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
const __VERSION__ = v"0.3.8-devel.261111"

"""
    __RELEASE__

Release date of this julia toolkit.

See also: [`__AUTHORS__`](@ref).
"""
const __RELEASE__ = "2026/01"

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

Print authors / contributors of the `KadanoffBaym` library.

See also: [`__AUTHORS__`](@ref).
"""
function authors()
    println("Authors (Until $__RELEASE__):")
    for a in __AUTHORS__
        println("  $(a.name) (email: $(a.email))")
    end
end
