#
# Project : Lavender
# Source  : config.jl
# Author  : Li Huang (huangli@caep.cn)
# Status  : Unstable
#
# Last modified: 2025/10/16
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
`PBASE` etc). In other words, all the relevant internal
dicts should be filled / updated in this function.

### Arguments
* cfg -> A dict struct that contains all the configurations (from neq.toml).

### Returns
N/A
"""
function fil_dict(cfg::Dict{String,Any})
    # For BASE block
    BASE = cfg["BASE"]
    for key in keys(BASE)
        if haskey(PBASE, key)
            PBASE[key][1] = BASE[key]
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
    println("[ Param: base ]")
    #
    println("finput  : ", get_b("finput") )
    println("solver  : ", get_b("solver") )
    println("ktype   : ", get_b("ktype")  )
    println("mtype   : ", get_b("mtype")  )
    println("grid    : ", get_b("grid")   )
    println("mesh    : ", get_b("mesh")   )
    println("ngrid   : ", get_b("ngrid")  )
    println("nmesh   : ", get_b("nmesh")  )
    println("wmax    : ", get_b("wmax")   )
    println("wmin    : ", get_b("wmin")   )
    println("beta    : ", get_b("beta")   )
    println("offdiag : ", get_b("offdiag"))
    println("fwrite  : ", get_b("fwrite") )
    println("pmodel  : ", get_b("pmodel") )
    println("pmesh   : ", get_b("pmesh")  )
    println("exclude : ", get_b("exclude"))
    #
    println()
    #
    flush(stdout)
end

"""
    rev_dict_b(BASE::Dict{String,Any})

Setup the configuration dictionary: `PBASE`.

### Arguments
* BASE -> A dict struct that contains configurations from the [BASE] block.

### Returns
N/A

See also: [`PBASE`](@ref).
"""
function rev_dict_b(BASE::Dict{String,Any})
    for key in keys(BASE)
        if haskey(PBASE, key)
            PBASE[key][1] = BASE[key]
        else
            error("Sorry, $key is not supported currently")
        end
    end
    foreach(x -> _v(x.first, x.second), PBASE)
end

"""
    rev_dict_b(BASE::Dict{String,Vector{Any}})

Setup the configuration dictionary: `PBASE`.

### Arguments
* BASE -> A dict struct that contains configurations from the [BASE] block.

### Returns
N/A

See also: [`PBASE`](@ref).
"""
function rev_dict_b(BASE::Dict{String,Vector{Any}})
    for key in keys(BASE)
        if haskey(PBASE, key)
            PBASE[key][1] = BASE[key][1]
        else
            error("Sorry, $key is not supported currently")
        end
    end
    foreach(x -> _v(x.first, x.second), PBASE)
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
    @assert get_b("solver") in ("MaxEnt", "BarRat", "NevanAC", "StochAC", "StochSK", "StochOM", "StochPX")
    @assert get_b("ktype") in ("fermi", "boson", "bsymm")
    @assert get_b("mtype") in ("flat", "gauss", "1gauss", "2gauss", "lorentz", "1lorentz", "2lorentz", "risedecay", "file")
    @assert get_b("grid") in ("ftime", "fpart", "btime", "bpart", "ffreq", "ffrag", "bfreq", "bfrag")
    @assert get_b("mesh") in ("linear", "tangent", "lorentz", "halflorentz")
    @assert get_b("ngrid") ≥ 1
    @assert get_b("nmesh") ≥ 1
    @assert get_b("wmax") > get_b("wmin")
    @assert get_b("beta") ≥ 0.0

    PA = [PBASE]
    #
    @cswitch get_b("solver") begin
        # For MaxEnt solver
        @case "MaxEnt"
            push!(PA, PMaxEnt)
            #
            @assert get_m("method") in ("historic", "classic", "bryan", "chi2kink")
            @assert get_m("stype") in ("sj", "br")
            @assert get_m("nalph") ≥ 1
            @assert get_m("alpha") > 0.0
            @assert get_m("ratio") > 0.0
            break

        # For BarRat solver
        @case "BarRat"
            push!(PA, PBarRat)
            # It does not support imaginary time data.
            # The Prony approximation doesn't support broken data.
            @assert get_b("grid") in ("ffreq", "ffrag", "bfreq", "bfrag")
            #
            @assert get_r("atype") in ("cont", "delta")
            @assert get_r("denoise") in ("none", "prony_s", "prony_o")
            @assert get_r("epsilon") ≥ 0.0
            @assert get_r("pcut")    ≥ 1e-6
            @assert get_r("eta")     ≥ 1e-8
            break

        # For NevanAC solver
        @case "NevanAC"
            push!(PA, PNevanAC)
            # It does not support imaginary time data.
            # It does not support bosonic frequency data directly.
            @assert get_b("grid") in ("ffreq", "ffrag")
            #
            @assert get_n("hmax")  ≥ 10
            @assert get_n("alpha") ≥ 1e-8
            @assert get_n("eta")   ≥ 1e-8
            break

        # For StochAC solver
        @case "StochAC"
            push!(PA, PStochAC)
            @assert get_b("mtype") == "flat"
            #
            @assert get_a("nfine") ≥ 1000
            @assert get_a("ngamm") ≥ 1
            @assert get_a("nwarm") ≥ 100
            @assert get_a("nstep") ≥ 1000
            @assert get_a("ndump") ≥ 100
            @assert get_a("nalph") ≥ 1
            @assert get_a("alpha") > 0.0
            @assert get_a("ratio") > 0.0
            break

        # For StochSK solver
        @case "StochSK"
            push!(PA, PStochSK)
            #
            @assert get_k("method") in ("chi2min", "chi2kink")
            @assert get_k("nfine") ≥ 10000
            @assert get_k("ngamm") ≥ 1
            @assert get_k("nwarm") ≥ 1000
            @assert get_k("nstep") ≥ 10000
            @assert get_k("ndump") ≥ 100
            @assert get_k("retry") ≥ 10
            @assert get_k("theta") > 1e+4
            @assert get_k("ratio") > 0.0
            break

        # For StochOM solver
        @case "StochOM"
            push!(PA, PStochOM)
            #
            @assert get_s("ntry")  ≥ 200
            @assert get_s("nstep") ≥ 1000
            @assert get_s("nbox")  ≥ 2
            @assert get_s("sbox")  > 0.0
            @assert get_s("wbox")  > 0.0
            break

        # For StochPX solver
        @case "StochPX"
            push!(PA, PStochPX)
            # It does not support imaginary time data.
            @assert get_b("grid") in ("ffreq", "ffrag", "bfreq", "bfrag")
            #
            @assert get_x("method") in ("best", "mean")
            @assert get_x("nfine") ≥ 10000
            @assert get_x("npole") ≥ 1
            @assert get_x("ntry")  ≥ 10
            @assert get_x("nstep") ≥ 100
            @assert get_x("theta") ≥ 0.00
            @assert get_x("eta")   ≥ 1e-8
            break
    end

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
    get_b(key::String)

Extract configurations from dict: PBASE.

### Arguments
* key -> Key of parameter.

### Returns
* value -> Value of parameter.

See also: [`PBASE`](@ref).
"""
@inline function get_b(key::String)
    if haskey(PBASE, key)
        PBASE[key][1]
    else
        error("Sorry, PBASE does not contain key: $key")
    end
end
