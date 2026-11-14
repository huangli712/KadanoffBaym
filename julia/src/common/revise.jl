#
# Project : Lavender
# Source  : revise.jl
# Author  : Li Huang (huangli@caep.cn)
# Status  : Unstable
#
# Last modified: 2026/11/10
#

"""
    fil_dict(cfg::Dict{String,Any})

Transfer configurations from dict `cfg` to internal dicts (including
`PCONTOUR` and `PMODEL` etc). In other words, all the relevant internal
dicts should be filled / updated in this function.

### Arguments
* cfg -> A dict struct that contains all the configurations (from neq.toml).

### Returns
N/A
"""
function fil_dict(cfg::Dict{String,Any})
    # For contour block
    contour = cfg["contour"]
    for key in keys(contour)
        if haskey(PCONTOUR, key)
            PCONTOUR[key][1] = contour[key]
        else
            error("Sorry, $key is not supported currently")
        end
    end

    # For model block
    model = cfg["model"]
    for key in keys(model)
        if haskey(PMODEL, key)
            PMODEL[key][1] = model[key]
        else
            error("Sorry, $key is not supported currently")
        end
    end
end

"""
    rev_dict_c(contour::Dict{String,Any})

Setup the configuration dictionary: `PCONTOUR`.

### Arguments
* contour -> A dict struct that contains configurations from the [contour] block.

### Returns
N/A

See also: [`PCONTOUR`](@ref).
"""
function rev_dict_c(contour::Dict{String,Any})
    for key in keys(contour)
        if haskey(PCONTOUR, key)
            PCONTOUR[key][1] = contour[key]
        else
            error("Sorry, $key is not supported currently")
        end
    end
    foreach(x -> _v(x.first, x.second), PCONTOUR)
end

"""
    rev_dict_c(contour::Dict{String,Vector{Any}})

Setup the configuration dictionary: `PCONTOUR`.

### Arguments
* contour -> A dict struct that contains configurations from the [contour] block.

### Returns
N/A

See also: [`PCONTOUR`](@ref).
"""
function rev_dict_c(contour::Dict{String,Vector{Any}})
    for key in keys(contour)
        if haskey(PCONTOUR, key)
            PCONTOUR[key][1] = contour[key][1]
        else
            error("Sorry, $key is not supported currently")
        end
    end
    foreach(x -> _v(x.first, x.second), PCONTOUR)
end

"""
    rev_dict_m(model::Dict{String,Any})

Setup the configuration dictionary: `PMODEL`.

### Arguments
* model -> A dict struct that contains configurations from the [model] block.

### Returns
N/A

See also: [`PMODEL`](@ref).
"""
function rev_dict_m(model::Dict{String,Any})
    for key in keys(model)
        if haskey(PMODEL, key)
            PMODEL[key][1] = model[key]
        else
            error("Sorry, $key is not supported currently")
        end
    end
    foreach(x -> _v(x.first, x.second), PMODEL)
end

"""
    rev_dict_m(model::Dict{String,Vector{Any}})

Setup the configuration dictionary: `PMODEL`.

### Arguments
* model -> A dict struct that contains configurations from the [model] block.

### Returns
N/A

See also: [`PMODEL`](@ref).
"""
function rev_dict_m(model::Dict{String,Vector{Any}})
    for key in keys(model)
        if haskey(PMODEL, key)
            PMODEL[key][1] = model[key][1]
        else
            error("Sorry, $key is not supported currently")
        end
    end
    foreach(x -> _v(x.first, x.second), PMODEL)
end
