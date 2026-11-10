#
# Project : Lavender
# Source  : example_config.jl
# Author  : Li Huang (huangli@caep.cn)
# Status  : Unstable
#
# Last modified: 2026/11/10
#

#
# TOML input of the Phase 5 examples (tutor/hubbard_chain,
# tutor/fkm_bethe_quench, tutor/holstein_bethe, tutor/holstein_impurity).
# It is private to the examples and is not exported by KadanoffBaym.
#
# Layout of a configuration file:
#
#   [contour]            nt, ntau (NESSi intervals), h, beta, solve_order
#   [solver]             mats_max_iter, mats_max_error, bootstrap_max_iter,
#                        bootstrap_max_error, corrector_steps, output_every
#   [model]              scalars of the example
#   [drive.<name>]       mode = "constant" and value, or mode = "file" and
#                        path (nt + 2 values, NESSi tstp = -1, ⋯, nt)
#   [output]             dir, prefix, save_green_components
#   [profile.quick]      overrides of the tables above (same keys)
#   [profile.release]
#
# Index mapping (NESSi -> KB): ntime = nt + 1, ntau_points = ntau + 1,
# the KB time step n is the NESSi time step n - 1, t = (n - 1) h, and the
# KB Matsubara step 0 is the NESSi step -1.
#

using TOML
using KadanoffBaym

const EXAMPLE_SECTIONS = ("contour", "solver", "model", "drive", "output", "profile")
const EXAMPLE_CONTOUR_KEYS = ("nt", "ntau", "h", "beta", "solve_order")
const EXAMPLE_SOLVER_KEYS = ("mats_max_iter", "mats_max_error", "bootstrap_max_iter",
                             "bootstrap_max_error", "corrector_steps")
const EXAMPLE_SOLVER_OPTIONAL = ("output_every",)
const EXAMPLE_OUTPUT_KEYS = ("dir", "prefix", "save_green_components")
const EXAMPLE_PROFILES = ("quick", "release")

"""
    ExampleConfig

Validated configuration of an example: the TOML file, the selected
profile, and the merged tables `contour`, `solver`, `model`, `drive`,
and `output`.
"""
struct ExampleConfig
    path    :: String
    dir     :: String
    profile :: String
    contour :: Dict{String,Any}
    solver  :: Dict{String,Any}
    model   :: Dict{String,Any}
    drive   :: Dict{String,Any}
    output  :: Dict{String,Any}
end

_cfg_error(path, msg) = throw(ArgumentError("$path: $msg"))

function _check_value(path, sec, key, v)
    if v isa Real
        isfinite(v) || _cfg_error(path, "[$sec] $key must be finite")
    elseif v isa AbstractVector
        for x in v
            _check_value(path, sec, key, x)
        end
    elseif !(v isa AbstractString || v isa Bool)
        _cfg_error(path, "[$sec] $key has an unsupported type")
    end
    return v
end

# Merge `src` into `dst`, the keys must be allowed
function _merge_table!(path, sec, dst::Dict{String,Any}, src, allowed)
    src isa AbstractDict || _cfg_error(path, "[$sec] must be a table")
    for (k, v) in src
        k in allowed || _cfg_error(path, "unknown key `$k` in [$sec]")
        dst[k] = _check_value(path, sec, k, v)
    end
    return dst
end

function _merge_drive!(path, dst::Dict{String,Any}, src, allowed)
    src isa AbstractDict || _cfg_error(path, "[drive] must be a table")
    for (name, t) in src
        name in allowed || _cfg_error(path, "unknown drive `$name`")
        t isa AbstractDict || _cfg_error(path, "[drive.$name] must be a table")
        d = Dict{String,Any}()
        _merge_table!(path, "drive.$name", d, t, ("mode", "value", "path"))
        mode = get(d, "mode", "")
        if mode == "constant"
            (haskey(d, "value") && d["value"] isa Real && !haskey(d, "path")) ||
                _cfg_error(path, "[drive.$name] constant mode needs a numeric `value`")
        elseif mode == "file"
            (haskey(d, "path") && d["path"] isa AbstractString && !haskey(d, "value")) ||
                _cfg_error(path, "[drive.$name] file mode needs a `path`")
        else
            _cfg_error(path, "[drive.$name] mode must be constant or file")
        end
        dst[name] = d
    end
    return dst
end

function _require(path, sec, d::Dict{String,Any}, keys)
    for k in keys
        haskey(d, k) || _cfg_error(path, "missing key `$k` in [$sec]")
    end
end

function _require_int(path, sec, d, k)
    d[k] isa Integer || _cfg_error(path, "[$sec] $k must be an integer")
end

"""
    load_example_config(path; profile = "quick", model_required = (),
                        model_optional = (), drives = (), drives_required = drives)

Read and validate the TOML file `path` and merge the overrides of
`[profile.<profile>]`. Unknown sections or keys, missing keys, and
non-finite values are rejected with an `ArgumentError`.

### Arguments
* path            -> TOML file.
* profile         -> "quick" or "release".
* model_required  -> Required keys of `[model]`.
* model_optional  -> Optional keys of `[model]`.
* drives          -> Allowed names of `[drive.<name>]`.
* drives_required -> Required drives.

### Returns
* An `ExampleConfig`.
"""
function load_example_config(path::AbstractString; profile::AbstractString = "quick",
                             model_required = (), model_optional = (),
                             drives = (), drives_required = drives)
    isfile(path) || throw(ArgumentError("$path: no such file"))
    profile in EXAMPLE_PROFILES ||
        throw(ArgumentError("profile must be quick or release, got $profile"))
    raw = TOML.parsefile(path)
    for k in keys(raw)
        k in EXAMPLE_SECTIONS || _cfg_error(path, "unknown section [$k]")
    end
    model_keys = (model_required..., model_optional...)
    allowed = Dict("contour" => EXAMPLE_CONTOUR_KEYS,
                   "solver" => (EXAMPLE_SOLVER_KEYS..., EXAMPLE_SOLVER_OPTIONAL...),
                   "model" => model_keys, "output" => EXAMPLE_OUTPUT_KEYS)
    tabs = Dict(s => Dict{String,Any}() for s in ("contour", "solver", "model", "output"))
    drv = Dict{String,Any}()
    apply!(t) = begin
        for (sec, v) in t
            if sec == "drive"
                _merge_drive!(path, drv, v, drives)
            elseif haskey(tabs, sec)
                _merge_table!(path, sec, tabs[sec], v, allowed[sec])
            else
                _cfg_error(path, "unknown section [$sec] in the profile")
            end
        end
    end
    apply!(Dict(k => v for (k, v) in raw if k != "profile"))
    # Both profiles must be present, they are validated even if not used
    prof = get(raw, "profile", Dict{String,Any}())
    prof isa AbstractDict || _cfg_error(path, "[profile] must be a table")
    for p in keys(prof)
        p in EXAMPLE_PROFILES || _cfg_error(path, "unknown profile `$p`")
    end
    for p in EXAMPLE_PROFILES
        haskey(prof, p) || _cfg_error(path, "missing [profile.$p]")
        prof[p] isa AbstractDict || _cfg_error(path, "[profile.$p] must be a table")
        for (sec, v) in prof[p]
            if sec == "drive"
                _merge_drive!(path, Dict{String,Any}(), v, drives)
            elseif haskey(tabs, sec)
                _merge_table!(path, "profile.$p.$sec", Dict{String,Any}(), v, allowed[sec])
            else
                _cfg_error(path, "unknown section [profile.$p.$sec]")
            end
        end
    end
    apply!(prof[profile])
    #
    _require(path, "contour", tabs["contour"], EXAMPLE_CONTOUR_KEYS)
    _require(path, "solver", tabs["solver"], EXAMPLE_SOLVER_KEYS)
    _require(path, "model", tabs["model"], model_required)
    _require(path, "output", tabs["output"], ("dir", "prefix"))
    for k in ("nt", "ntau", "solve_order")
        _require_int(path, "contour", tabs["contour"], k)
    end
    for k in ("mats_max_iter", "bootstrap_max_iter", "corrector_steps")
        _require_int(path, "solver", tabs["solver"], k)
    end
    get!(tabs["solver"], "output_every", 1)
    get!(tabs["output"], "save_green_components", false)
    tabs["output"]["save_green_components"] isa Bool ||
        _cfg_error(path, "[output] save_green_components must be a boolean")
    p = String(tabs["output"]["prefix"])
    (isempty(p) || endswith(p, ".h5")) && _cfg_error(path, "[output] bad prefix `$p`")
    for d in drives_required
        haskey(drv, d) || _cfg_error(path, "missing [drive.$d]")
    end
    #
    cfg = ExampleConfig(abspath(path), dirname(abspath(path)), String(profile),
                        tabs["contour"], tabs["solver"], tabs["model"], drv, tabs["output"])
    contour_params(cfg)
    return cfg
end

"""
    contour_params(cfg::ExampleConfig)

Contour parameters in both conventions: `nt`, `ntau` (NESSi), `ntime =
nt + 1`, `ntau_points = ntau + 1` (KB), `h`, `beta`, the order `k`, and
`I = Integrator(k)`.
"""
function contour_params(cfg::ExampleConfig)
    c = cfg.contour
    nt, ntau, k = c["nt"], c["ntau"], c["solve_order"]
    h, β = Float64(c["h"]), Float64(c["beta"])
    k in 1:5 || _cfg_error(cfg.path, "[contour] solve_order must be in 1:5, got $k")
    nt ≥ k || _cfg_error(cfg.path, "[contour] nt must be ≥ solve_order")
    ntau ≥ 2 || _cfg_error(cfg.path, "[contour] ntau must be ≥ 2")
    (h > 0.0 && β > 0.0) || _cfg_error(cfg.path, "[contour] h and beta must be positive")
    return (nt = nt, ntau = ntau, ntime = nt + 1, ntau_points = ntau + 1, h = h, beta = β,
            k = k, I = Integrator(k))
end

"""
    solver_params(cfg::ExampleConfig)

Solver parameters as a named tuple.
"""
function solver_params(cfg::ExampleConfig)
    s = cfg.solver
    v = (mats_max_iter = s["mats_max_iter"], mats_max_error = Float64(s["mats_max_error"]),
         bootstrap_max_iter = s["bootstrap_max_iter"],
         bootstrap_max_error = Float64(s["bootstrap_max_error"]),
         corrector_steps = s["corrector_steps"], output_every = s["output_every"])
    (v.mats_max_iter ≥ 0 && v.bootstrap_max_iter ≥ 0 && v.corrector_steps ≥ 1 &&
     v.mats_max_error > 0.0 && v.bootstrap_max_error > 0.0) ||
        _cfg_error(cfg.path, "[solver] invalid values")
    return v
end

"""
    load_tvector!(f::Cf, cfg::ExampleConfig, name)

Fill the single-time function `f` (1x1, `f.ntime = nt + 1`) with the drive
`[drive.<name>]`. A file holds `nt + 2` values in the order of the NESSi
time steps `-1, 0, ⋯, nt`: the first one is the Matsubara value `f[0]`,
the following ones are `f[1], ⋯, f[ntime]`. A relative path is resolved
from the directory of the TOML file.
"""
function load_tvector!(f::Cf{ComplexF64}, cfg::ExampleConfig, name::AbstractString)
    haskey(cfg.drive, name) || _cfg_error(cfg.path, "missing [drive.$name]")
    d = cfg.drive[name]
    ntime = contour_params(cfg).ntime
    f.ntime == ntime || throw(DimensionMismatch("f must have ntime = $ntime"))
    if d["mode"] == "constant"
        v = fill(Float64(d["value"]), ntime + 1)
    else
        p = String(d["path"])
        p = isabspath(p) ? p : joinpath(cfg.dir, p)
        isfile(p) || _cfg_error(cfg.path, "[drive.$name] no such file $p")
        v = Float64[]
        for line in eachline(p)
            s = strip(line)
            (isempty(s) || startswith(s, "#")) && continue
            for w in split(s)
                x = tryparse(Float64, w)
                (x === nothing || !isfinite(x)) &&
                    _cfg_error(cfg.path, "[drive.$name] bad value `$w` in $p")
                push!(v, x)
            end
        end
        length(v) == ntime + 1 ||
            _cfg_error(cfg.path, "[drive.$name] $p must hold nt + 2 = $(ntime + 1) values, " *
                                 "got $(length(v))")
    end
    for i = 0:ntime
        f[i] = fill(ComplexF64(v[i+1]), 1, 1)
    end
    return f
end

"""
    parse_example_args(args)

Parse the command line of an example entry: `<config.toml> [--profile
quick|release] [--validate-only]`.
"""
function parse_example_args(args)
    isempty(args) && throw(ArgumentError("usage: <config.toml> [--profile quick|release] " *
                                         "[--validate-only]"))
    cfgfile = args[1]
    profile = "quick"
    validate = false
    i = 2
    while i ≤ length(args)
        a = args[i]
        if a == "--profile"
            i < length(args) || throw(ArgumentError("--profile needs a value"))
            profile = args[i+1]
            i += 2
        elseif startswith(a, "--profile=")
            profile = split(a, "=")[2]
            i += 1
        elseif a == "--validate-only"
            validate = true
            i += 1
        else
            throw(ArgumentError("unknown argument `$a`"))
        end
    end
    return (cfgfile = cfgfile, profile = String(profile), validate_only = validate)
end
