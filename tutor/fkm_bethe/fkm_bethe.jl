#
# Project : Lavender
# Source  : fkm_bethe_quench.jl
# Author  : Li Huang (huangli@caep.cn)
# Status  : Unstable
#
# Last modified: 2026/11/10
#

#
# Usage:
#
#   julia --project=. tutor/fkm_bethe/fkm_bethe_quench.jl \
#         tutor/fkm_bethe/fkm_bethe_quench.toml [--profile quick|release] [--validate-only]
#
# Interaction quench U0 -> U1 of the Falicov-Kimball model on the Bethe
# lattice in DMFT (NESSi examples/programs/fkm_bethe_quench.cpp and
# libcntr/test/fkm_bethe_quench.cpp):
#
#   G0 = [i d/dt + mu - f0 - Delta]^{-1},  f0 = -U/2
#   G1 = [i d/dt + mu - f1 - Delta]^{-1},  f1 = +U/2
#   G  = (G0 + G1) / 2,  Delta = G,  mu = 0
#
# U = U0 on the Matsubara branch and U = U1 at t >= 0. The Matsubara Dyson
# equations use the Fourier method. Output: <prefix>_density.dat with
# t, den0, den1, total (real time steps).
#

using LinearAlgebra
using Printf
using KadanoffBaym

include(joinpath(@__DIR__, "..", "common", "example_config.jl"))
include(joinpath(@__DIR__, "..", "common", "text_output.jl"))
include(joinpath(@__DIR__, "..", "common", "contour_loop.jl"))

const FKM_MODEL_KEYS = ("U0", "U1")

"""
    FKMState

Green's functions `G0`, `G1`, `G`, the hybridization `Δ`, the previous
iterate `Gold`, and the single-time functions `f0`, `f1`.
"""
struct FKMState
    G0   :: ℱ{ComplexF64}
    G1   :: ℱ{ComplexF64}
    G    :: ℱ{ComplexF64}
    Δ    :: ℱ{ComplexF64}
    Gold :: ℱ{ComplexF64}
    f0   :: Cf{ComplexF64}
    f1   :: Cf{ComplexF64}
    β    :: Float64
    h    :: Float64
    k    :: Int
end

"""
    load_fkm_config(cfgfile, profile)

Read and validate the TOML file; `U0` and `U1` must be finite and
non-negative.
"""
function load_fkm_config(cfgfile::AbstractString, profile::AbstractString)
    cfg = load_example_config(cfgfile; profile = profile, model_required = FKM_MODEL_KEYS)
    for k in FKM_MODEL_KEYS
        (cfg.model[k] isa Real && cfg.model[k] ≥ 0.0) ||
            throw(ArgumentError("$cfgfile: [model] $k must be a non-negative number"))
    end
    solver_params(cfg)
    return cfg
end

"""
    fkm_state(cfg)

Allocate the Green's functions, `Δ` = Matsubara Green's function of the
Bethe lattice (NESSi `green_equilibrium_mat_bethe`), and `f0`, `f1`.
"""
function fkm_state(cfg::ExampleConfig)
    c = contour_params(cfg)
    C = Cn(c.ntime, c.ntau_points, 1, 1, (c.ntime - 1) * c.h, c.beta)
    F() = ℱ(C, FERMI)
    Δ = F()
    from_dos_mat_bethe!(Δ, c.beta)
    U0 = Float64(cfg.model["U0"])
    U1 = Float64(cfg.model["U1"])
    f0 = Cf(c.ntime, 1, 1, zero(ComplexF64))
    f1 = Cf(c.ntime, 1, 1, zero(ComplexF64))
    f0[0] = fill(ComplexF64(-0.5U0), 1, 1)
    f1[0] = fill(ComplexF64(0.5U0), 1, 1)
    for i = 1:c.ntime
        f0[i] = fill(ComplexF64(-0.5U1), 1, 1)
        f1[i] = fill(ComplexF64(0.5U1), 1, 1)
    end
    return FKMState(F(), F(), F(), Δ, F(), f0, f1, c.beta, c.h, c.k)
end

# G = (G0 + G1) / 2 at the KB time step n
function _fkm_average!(st::FKMState, n::Int)
    memcpy!(st.G0, st.G, n)
    smul!(st.G, n, 0.5)
    incr!(st.G, st.G1, n, 0.5)
end

"""
    solve_fkm!(st, cfg; log = _noop)

Matsubara iteration (FOURIER method, at least four iterations as in
NESSi), bootstrapping (exit at the first iteration below the threshold),
and propagation with an extrapolated `Δ` and `corrector_steps` correctors.
"""
function solve_fkm!(st::FKMState, cfg::ExampleConfig; log::Function = _noop)
    s = solver_params(cfg)
    c = contour_params(cfg)
    k, μ = st.k, 0.0
    # Matsubara: the previous iterate and Δ are updated only if the
    # iteration did not converge (NESSi :80-89)
    iterate_matsubara!(; max_iter = s.mats_max_iter, tol = s.mats_max_error,
                       converged = (iter, err) -> err < s.mats_max_error && iter > 2,
                       log = log) do iter
        dyson_mat!(st.G0, st.Δ, μ, st.f0, st.β; solve_order = k, method = :fourier)
        dyson_mat!(st.G1, st.Δ, μ, st.f1, st.β; solve_order = k, method = :fourier)
        _fkm_average!(st, 0)
        err = distance(st.G, st.Gold, 0)
        if !(err < s.mats_max_error && iter > 2)
            memcpy!(st.G, st.Gold, 0)
            memcpy!(st.G, st.Δ, 0)
        end
        err
    end
    # Bootstrapping
    iterate_bootstrap!(; max_iter = s.bootstrap_max_iter, tol = s.bootstrap_max_error,
                       min_iterations = 0, log = log) do iter
        dyson_start!(st.G0, μ, st.f0, st.Δ, st.β, st.h; solve_order = k)
        dyson_start!(st.G1, μ, st.f1, st.Δ, st.β, st.h; solve_order = k)
        err = 0.0
        for n = 1:k+1
            _fkm_average!(st, n)
            err += distance(st.G, st.Gold, n)
        end
        if !(err < s.bootstrap_max_error)
            for n = 1:k+1
                memcpy!(st.G, st.Gold, n)
                memcpy!(st.G, st.Δ, n)
            end
        end
        err
    end
    # Propagation
    propagate_timesteps!(n -> extrapolate_timestep!(n - 1, st.Δ, k),
                         (n, iter) -> begin
                             dyson_timestep!(n, st.G0, μ, st.f0, st.Δ, st.β, st.h; solve_order = k)
                             dyson_timestep!(n, st.G1, μ, st.f1, st.Δ, st.β, st.h; solve_order = k)
                             _fkm_average!(st, n)
                             memcpy!(st.G, st.Δ, n)
                         end;
                         first_step = k + 2, last_step = c.ntime,
                         corrector_steps = s.corrector_steps, h = st.h, log = log)
    return st
end

"""
    fkm_densities(st)

`(n, den0, den1)` for the KB time steps `n = 0, ⋯, ntime` (`n = 0` is the
Matsubara branch), `den = Re ρ`.
"""
fkm_densities(st::FKMState) =
    [(n, real(density_matrix(st.G0, n)[1,1]), real(density_matrix(st.G1, n)[1,1]))
     for n = 0:getntime(st.G)]

"""
    main(args; io = stdout)

Entry: `<config.toml> [--profile quick|release] [--validate-only]`.
Writes `<prefix>_density.dat` (`t, den0, den1, total` at the real time
steps). Returns the state and the densities.
"""
function main(args; io::IO = stdout)
    a = parse_example_args(args)
    cfg = load_fkm_config(a.cfgfile, a.profile)
    st = fkm_state(cfg)
    a.validate_only && return (config = cfg, state = st)
    logf(e) = e.stage in (:matsubara, :bootstrap) &&
        @printf(io, "%s iteration %d  error = %.6e\n", e.stage, e.iter, e.err)
    solve_fkm!(st, cfg; log = logf)
    dens = fkm_densities(st)
    dir = cfg.output["dir"]
    dir = isabspath(dir) ? dir : joinpath(cfg.dir, dir)
    mkpath(dir)
    rows = [[(n - 1) * st.h, d0, d1, d0 + d1] for (n, d0, d1) in dens if n ≥ 1]
    write_table(joinpath(dir, cfg.output["prefix"] * "_density.dat"),
                ["t", "den0", "den1", "total"], rows)
    return (config = cfg, state = st, densities = dens)
end

if abspath(PROGRAM_FILE) == (@__FILE__)
    try
        main(ARGS)
    catch e
        showerror(stderr, e)
        println(stderr)
        exit(1)
    end
end
