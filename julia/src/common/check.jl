#
# Project : Lavender
# Source  : check.jl
# Author  : Li Huang (huangli@caep.cn)
# Status  : Unstable
#
# Last modified: 2026/11/10
#

"""
    chk_dict()

Validate the correctness and consistency of configurations.

### Arguments
N/A

### Returns
N/A

See also: [`fil_dict`](@ref), [`_v`](@ref).
"""
function chk_dict()
    @assert get_c("ntime") ≥ 2
    @assert get_c("ntau")  ≥ 2
    @assert get_c("ndim1") ≥ 1
    @assert get_c("ndim2") ≥ 1
    @assert get_c("tmax")  ≥ 0.0
    @assert get_c("beta")  ≥ 0.0

    PA = [PCONTOUR, PMODEL]

    for P in PA
        foreach(x -> _v(x.first, x.second), P)
    end
end

"""
    _v(key::String, val::Array{Any,1})

Verify the value array. Called by chk_dict() function only.

### Arguments
* key -> Key of parameter.
* val -> Value of parameter.

### Returns
N/A

See also: [`chk_dict`](@ref).
"""
@inline function _v(key::String, val::Array{Any,1})
    # To check if the value is updated
    if isa(val[1], Missing) && val[2] > 0
        error("Sorry, key ($key) shoule be set")
    end

    # To check if the type of value is correct
    if !isa(val[1], Missing) && !isa(val[1], eval(val[3]))
        error("Sorry, type of key ($key) is wrong")
    end
end
