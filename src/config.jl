#
# Project : Lavender
# Source  : config.jl
# Author  : Li Huang (huangli@caep.cn)
# Status  : Unstable
#
# Last modified: 2025/10/20
#

"""
    inp_toml(f::String, key::String, necessary::Bool)

Parse the configuration file (in toml format). It reads only parts of
the configuration file, which depends on the value of `key`.

### Arguments
* f -> Filename of configuration.
* key -> Parameter's name.
* necessary -> If it is true and configuration is absent, raise an error.

### Returns
* value -> Parameter's value.
"""
function inp_toml(f::String, key::String, necessary::Bool)
    if isfile(f)
        dict = TOML.parsefile(f)
        #
        if haskey(dict, key)
            dict[key]
        else
            error("Do not have this key: $key in file: $f")
        end
    else
        if necessary
            error("Please make sure that the file $f really exists")
        else
            nothing
        end
    end
end

"""
    inp_toml(f::String, necessary::Bool)

Parse the configuration file (in toml format). It reads the whole file.

### Arguments
* f -> Filename of configuration.
* necessary -> If it is true and configuration is absent, raise an error.

### Returns
* dict -> A Dictionary struct that contains all the key-value pairs.
"""
function inp_toml(f::String, necessary::Bool)
    if isfile(f)
        dict = TOML.parsefile(f)
        return dict
    else
        if necessary
            error("Please make sure that the file $f really exists")
        else
            nothing
        end
    end
end

"""
    fil_dict(cfg::Dict{String,Any})

Transfer configurations from dict `cfg` to internal dicts (including
`PCONTOUR` etc). In other words, all the relevant internal
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
end

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
    println("[ Param: contour ]")
    #
    println("ntime : ", get_c("ntime"))
    println("ntau  : ", get_c("ntau") )
    println("ndim1 : ", get_c("ndim1"))
    println("ndim2 : ", get_c("ndim2"))
    println("tmax  : ", get_c("tmax") )
    println("beta  : ", get_c("beta") )
    #
    println()
    #
    flush(stdout)
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
    chk_dict()

Validate the correctness and consistency of configurations.

### Arguments
N/A

### Returns
N/A

See also: [`fil_dict`](@ref), [`_v`](@ref).
"""
function chk_dict()
    @assert get_c("ntime") ≥ 1
    @assert get_c("ntau")  ≥ 1
    @assert get_c("ndim1") ≥ 1
    @assert get_c("ndim2") ≥ 1
    #
    @assert get_c("tmax") ≥ 0.0
    @assert get_c("beta") ≥ 0.0

    PA = [PCONTOUR]

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
