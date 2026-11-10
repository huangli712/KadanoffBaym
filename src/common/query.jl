#
# Project : Lavender
# Source  : query.jl
# Author  : Li Huang (huangli@caep.cn)
# Status  : Unstable
#
# Last modified: 2026/11/10
#

"""
    see_dict()

Display all of the relevant configuration parameters to the terminal.

### Arguments
N/A

### Returns
N/A

See also: [`fil_dict`](@ref).
"""
function see_dict()
    println("[ param: contour ]")
    #
    println("ntime  : ", get_c("ntime") )
    println("ntau   : ", get_c("ntau")  )
    println("ndim1  : ", get_c("ndim1") )
    println("ndim2  : ", get_c("ndim2") )
    println("tmax   : ", get_c("tmax")  )
    println("beta   : ", get_c("beta")  )
    #
    println()
    #
    println("[ param: model ]")
    #
    println("system : ", get_m("system"))
    #
    flush(stdout)
end

"""
    get_c(key::String)

Extract configurations from dict: PCONTOUR.

### Arguments
* key -> Key of parameter.

### Returns
* value -> Value of parameter.

See also: [`PCONTOUR`](@ref).
"""
@inline function get_c(key::String)
    if haskey(PCONTOUR, key)
        PCONTOUR[key][1]
    else
        error("Sorry, PCONTOUR does not contain key: $key")
    end
end

"""
    get_m(key::String)

Extract configurations from dict: PMODEL.

### Arguments
* key -> Key of parameter.

### Returns
* value -> Value of parameter.

See also: [`PMODEL`](@ref).
"""
@inline function get_m(key::String)
    if haskey(PMODEL, key)
        PMODEL[key][1]
    else
        error("Sorry, PMODEL does not contain key: $key")
    end
end
