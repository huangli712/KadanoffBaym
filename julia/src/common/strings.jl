#
# Project : Lavender
# Source  : strings.jl
# Author  : Li Huang (huangli@caep.cn)
# Status  : Unstable
#
# Last modified: 2026/11/10
#

#=
### *I/O Operations*
=#

"""
    line_to_array(io::IOStream)

Convert a line (reading from an IOStream) to a string array.

### Arguments
* io -> An IOStream struct.

### Returns
* arr -> An array  of String.
"""
@inline function line_to_array(io::IOStream)
    split(readline(io), " ", keepempty = false)
end

"""
    line_to_array(str::AbstractString)

Convert a string (AbstractString) to a string array.

### Arguments
* str -> A String.

### Returns
* arr -> An array of String.

### Examples
```julia-repl
julia> str = "Hello World!"
"Hello World!"

julia> line_to_array(str)
2-element Vector{SubString{String}}:
 "Hello"
 "World!"
```
"""
@inline function line_to_array(str::AbstractString)
    split(str, " ", keepempty = false)
end
