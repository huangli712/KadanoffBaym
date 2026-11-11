#
# Project : Lavender
# Source  : holstein_bethe_impl.jl
# Author  : Li Huang (huangli@caep.cn)
# Status  : Unstable
#
# Last modified: 2026/11/10
#

#
# Holstein model on the Bethe lattice in DMFT, normal state and Nambu
# (superconducting) form, with the self-consistent (Migdal) or the
# unrenormalized (uMig) Migdal self-energy. Port of NESSi
# examples/programs/Holstein_bethe_{Migdal,uMig,Nambu_Migdal,Nambu_uMig}.cpp
# (no HDF5). As in tutor/hubbard_chain, the mean-field terms use the real
# part of the density (the hermitian part of the density matrix).
#

using LinearAlgebra
using Printf
using KadanoffBaym

include(joinpath(@__DIR__, "..", "common", "example_config.jl"))
include(joinpath(@__DIR__, "..", "common", "text_output.jl"))
include(joinpath(@__DIR__, "..", "common", "contour_loop.jl"))
include(joinpath(@__DIR__, "holstein_nambu_impl.jl"))
include(joinpath(@__DIR__, "holstein_energy.jl"))

const HOLSTEIN_APPROX = ("Migdal", "uMig")
const HOLSTEIN_BETHE_KEYS = ("Hopping", "El_Ph_g", "Phfreq_w0", "MuChem_MF")
const HOLSTEIN_NAMBU_KEYS = ("Hopping", "El_Ph_g", "Phfreq_w0", "MuChem_MF", "sc_seed",
                             "sc_seed_iterations")

# Values of a drive at the KB time steps 0, ⋯, ntime
function holstein_drive(cfg::ExampleConfig, name::AbstractString)
    ntime = contour_params(cfg).ntime
    f = Cf(ntime, 1, 1, zero(ComplexF64))
    load_tvector!(f, cfg, name)
    return [real(f[n][1,1]) for n = 0:ntime]
end

"""
    load_holstein_bethe_config(cfgfile, profile; nambu = false)

Read and validate the TOML file of the Bethe Holstein examples.
"""
function load_holstein_bethe_config(cfgfile::AbstractString, profile::AbstractString;
                                    nambu::Bool = false)
    keys = nambu ? HOLSTEIN_NAMBU_KEYS : HOLSTEIN_BETHE_KEYS
    drives = nambu ? ("dHopping", "sc_field") : ("dHopping", "dEl_Ph_g")
    cfg = load_example_config(cfgfile; profile = profile, model_required = keys, drives = drives)
    solver_params(cfg)
    for k in keys
        cfg.model[k] isa Real || throw(ArgumentError("$cfgfile: [model] $k must be a number"))
    end
    cfg.model["Phfreq_w0"] > 0.0 || throw(ArgumentError("$cfgfile: [model] Phfreq_w0 must be > 0"))
    if nambu
        it = cfg.model["sc_seed_iterations"]
        (it isa Integer && it ≥ 0) ||
            throw(ArgumentError("$cfgfile: [model] sc_seed_iterations must be a non-negative integer"))
    end
    for d in drives
        holstein_drive(cfg, d)
    end
    return cfg
end

# Two-stage Matsubara iteration of the Bethe examples (NESSi :204-274).
# step!(method, iter) performs one iteration up to the mixing and the
# update of the hybridization and returns the error.
function _holstein_matsubara!(step!::Function, Gold::ℱ, G::ℱ, s, io::IO)
    stage = MatsubaraTwoStage(false, false)
    memcpy!(G, Gold, 0)
    iterate_matsubara!(; max_iter = s.mats_max_iter, tol = s.mats_max_error,
                       converged = (iter, err) -> stage.converged) do iter
        m = matsubara_method(stage)
        err = step!(m, iter)
        @printf(io, "matsubara (%s) iteration %d  error = %.6e\n", m, iter, err)
        matsubara_update!(stage, err, s.mats_max_error) && memcpy!(G, Gold, 0)
        err
    end
    return stage
end

# Bootstrapping with at least three iterations; step!(iter) returns the error
function _holstein_bootstrap!(step!::Function, Gold::ℱ, G::ℱ, k::Int, s, io::IO)
    for n = 1:k+1
        memcpy!(G, Gold, n)
    end
    iterate_bootstrap!(; max_iter = s.bootstrap_max_iter, tol = s.bootstrap_max_error,
                       min_iterations = 3) do iter
        err = step!(iter)
        @printf(io, "bootstrap iteration %d  error = %.6e\n", iter, err)
        if !(err < s.bootstrap_max_error && iter > 2)
            for n = 1:k+1
                memcpy!(G, Gold, n)
            end
        end
        err
    end
end

"""
    run_holstein_bethe(approx, cfg; io = stdout)

Normal-state Holstein model on the Bethe lattice (NESSi
Holstein_bethe_Migdal.cpp / Holstein_bethe_uMig.cpp). Returns the Green's
functions, the phonon state, and the observables at all KB time steps
(`n + 1` indexing): `rho`, `Xph`, `Pph`, `XX`, `Ekin`, `Enx_MF`,
`Enx_corr`, `Eph_cl`, `Eph_corr`, `Etot`.
"""
function run_holstein_bethe(approx::AbstractString, cfg::ExampleConfig; io::IO = stdout)
    approx in HOLSTEIN_APPROX || throw(ArgumentError("unknown approximation $approx"))
    mig = approx == "Migdal"
    c, s, m = contour_params(cfg), solver_params(cfg), cfg.model
    ntime, k, β, h = c.ntime, c.k, c.beta, c.h
    w0 = Float64(m["Phfreq_w0"])
    gv = ComplexF64.(m["El_Ph_g"] .+ holstein_drive(cfg, "dEl_Ph_g"))
    Jv = ComplexF64.(m["Hopping"] .+ holstein_drive(cfg, "dHopping"))
    g = holstein_scalar_cf(ntime, gv)
    J = holstein_scalar_cf(ntime, Jv)
    C = Cn(ntime, c.ntau_points, 1, 1, (ntime - 1) * h, β)
    F() = ℱ(C, FERMI)
    G, Hyb, Σ, HS, Gold = F(), F(), F(), F(), F()
    ph = holstein_phonon(ntime, c.ntau_points, 1, w0, β, h, k)
    h0MF = holstein_scalar_cf(ntime, fill(-Float64(m["MuChem_MF"]), ntime + 1))
    ntot = zeros(ComplexF64, ntime + 1)
    X = zeros(ComplexF64, ntime + 1)
    from_dos_mat_bethe!(G, β)
    holstein_hybridization!(0, Hyb, G, J)
    # Matsubara
    _holstein_matsubara!(Gold, G, s, io) do meth, iter
        mig ? sigma_mig!(0, G, Σ, g, ph; method = meth) : sigma_umig!(0, G, Σ, g, ph)
        _hyb_sigma!(0, HS, Hyb, Σ)
        dyson_mat!(G, HS, 0.0, h0MF, β; solve_order = k, method = meth)
        smul!(G, 0, 0.5)
        incr!(G, Gold, 0, 0.5)
        holstein_hybridization!(0, Hyb, G, J)
        distance(G, Gold, 0)
    end
    ntot[1] = 2.0 * real(density_matrix(G, 0)[1,1])
    phonon_displacement!(0, X, ntot, gv, ph)
    h0imp = -(Float64(m["MuChem_MF"]) + real(X[1] * gv[1]))
    # Start-up time steps from the Matsubara Green's function
    set_tk_from_mat!(G, k)
    set_tk_from_mat!(Hyb, k)
    _holstein_bootstrap!(Gold, G, k, s, io) do iter
        mig ? sigma_mig_start!(G, Σ, g, ph) : sigma_umig_start!(G, Σ, g, ph)
        for n = 1:k+1
            ntot[n+1] = 2.0 * real(density_matrix(G, n)[1,1])
        end
        phonon_displacement_start!(X, ntot, gv, ph)
        for n = 1:k+1
            h0MF[n] = fill(h0imp + X[n+1] * gv[n+1], 1, 1)
            _hyb_sigma!(n, HS, Hyb, Σ)
        end
        dyson_start!(G, 0.0, h0MF, HS, β, h; solve_order = k)
        err = 0.0
        for n = 1:k+1
            holstein_hybridization!(n, Hyb, G, J)
            err += distance(G, Gold, n)
        end
        err
    end
    # Propagation
    propagate_timesteps!(n -> (extrapolate_timestep!(n - 1, G, k);
                               extrapolate_timestep!(n - 1, Hyb, k)),
                         (n, iter) -> begin
                             mig ? sigma_mig!(n, G, Σ, g, ph) : sigma_umig!(n, G, Σ, g, ph)
                             ntot[n+1] = 2.0 * real(density_matrix(G, n)[1,1])
                             phonon_displacement!(n, X, ntot, gv, ph)
                             h0MF[n] = fill(h0imp + X[n+1] * gv[n+1], 1, 1)
                             _hyb_sigma!(n, HS, Hyb, Σ)
                             dyson_timestep!(n, G, 0.0, h0MF, HS, β, h; solve_order = k)
                             holstein_hybridization!(n, Hyb, G, J)
                         end;
                         first_step = k + 2, last_step = ntime,
                         corrector_steps = s.corrector_steps, h = h)
    # Energies and momentum
    Dout = mig ? ph.D : ph.D0
    if !mig
        for n = 0:ntime
            smul!(ph.Pi, n, 0.0)
        end
    end
    P = phonon_momentum(ntot, gv, ph)
    Ecl = phonon_energy_classical(X, P, w0)
    Equ = phonon_energy_quantum(Dout, ph.Pi, ph)
    Ekin = [4.0 * correlation_energy(n, G, Hyb, β, h, k) for n = 0:ntime]
    EMF = [gv[n+1] * X[n+1] * ntot[n+1] for n = 0:ntime]
    Ecorr = [4.0 * correlation_energy(n, G, Σ, β, h, k) for n = 0:ntime]
    Etot = Ekin .+ EMF .+ Ecorr .+ Ecl .+ Equ
    XX = [n == 0 ? zero(ComplexF64) : im * Dout.less[n, n][1,1] for n = 0:ntime]
    rho = [density_matrix(G, n) for n = 0:ntime]
    return (G = G, Σ = Σ, Hyb = Hyb, phonon = ph, ntot = ntot, rho = rho, Xph = X, Pph = P,
            XX = XX, Ekin = Ekin, Enx_MF = EMF, Enx_corr = Ecorr, Eph_cl = Ecl, Eph_corr = Equ,
            Etot = Etot)
end

"""
    run_holstein_nambu(approx, cfg; io = stdout)

Holstein model on the Bethe lattice in the Nambu form (NESSi
Holstein_bethe_Nambu_{Migdal,uMig}.cpp). The superconducting seed field
`sc_seed σ₁` is added to the Matsubara Hamiltonian in the first
`sc_seed_iterations` iterations. Returns the observables at all KB time
steps: `rho` (2x2), `XX`, `Ekin`, `Enx_MF` (zero, as in NESSi),
`Enx_corr`, `Eph`, `Etot`.
"""
function run_holstein_nambu(approx::AbstractString, cfg::ExampleConfig; io::IO = stdout)
    approx in HOLSTEIN_APPROX || throw(ArgumentError("unknown approximation $approx"))
    mig = approx == "Migdal"
    c, s, m = contour_params(cfg), solver_params(cfg), cfg.model
    ntime, k, β, h = c.ntime, c.k, c.beta, c.h
    w0 = Float64(m["Phfreq_w0"])
    σ₃ = ComplexF64[1 0; 0 -1]
    σ₁ = ComplexF64[0 1; 1 0]
    dJ = holstein_drive(cfg, "dHopping")
    scf = holstein_drive(cfg, "sc_field")
    g = Cf(ntime, 2, 2, zero(ComplexF64))
    J = Cf(ntime, 2, 2, zero(ComplexF64))
    H = Cf(ntime, 2, 2, zero(ComplexF64))
    h0imp = -Float64(m["MuChem_MF"]) .* σ₃
    for n = 0:ntime
        g[n] = Float64(m["El_Ph_g"]) .* σ₃
        J[n] = (Float64(m["Hopping"]) + dJ[n+1]) .* σ₃
        H[n] = h0imp .+ scf[n+1] .* σ₁
    end
    C = Cn(ntime, c.ntau_points, 2, 2, (ntime - 1) * h, β)
    F() = ℱ(C, FERMI)
    G, Hyb, Σ, HS, Gold = F(), F(), F(), F(), F()
    ph = holstein_phonon(ntime, c.ntau_points, 2, w0, β, h, k)
    from_dos_mat_bethe!(G, β)
    holstein_hybridization!(0, Hyb, G, J)
    seed = Float64(m["sc_seed"])
    nseed = m["sc_seed_iterations"]
    _holstein_matsubara!(Gold, G, s, io) do meth, iter
        mig ? sigma_mig_sc!(0, G, Σ, g, ph; method = meth) : sigma_umig_sc!(0, G, Σ, g, ph)
        _hyb_sigma!(0, HS, Hyb, Σ)
        H[0] = h0imp .+ (iter < nseed ? seed : 0.0) .* σ₁
        dyson_mat!(G, HS, 0.0, H, β; solve_order = k, method = meth)
        smul!(G, 0, 0.5)
        incr!(G, Gold, 0, 0.5)
        holstein_hybridization!(0, Hyb, G, J)
        distance(G, Gold, 0)
    end
    set_tk_from_mat!(G, k)
    set_tk_from_mat!(Hyb, k)
    _holstein_bootstrap!(Gold, G, k, s, io) do iter
        mig ? sigma_mig_sc_start!(G, Σ, g, ph) : sigma_umig_sc_start!(G, Σ, g, ph)
        for n = 1:k+1
            _hyb_sigma!(n, HS, Hyb, Σ)
        end
        dyson_start!(G, 0.0, H, HS, β, h; solve_order = k)
        err = 0.0
        for n = 1:k+1
            holstein_hybridization!(n, Hyb, G, J)
            err += distance(G, Gold, n)
        end
        err
    end
    propagate_timesteps!(n -> (extrapolate_timestep!(n - 1, G, k);
                               extrapolate_timestep!(n - 1, Hyb, k)),
                         (n, iter) -> begin
                             mig ? sigma_mig_sc!(n, G, Σ, g, ph) : sigma_umig_sc!(n, G, Σ, g, ph)
                             _hyb_sigma!(n, HS, Hyb, Σ)
                             dyson_timestep!(n, G, 0.0, H, HS, β, h; solve_order = k)
                             holstein_hybridization!(n, Hyb, G, J)
                         end;
                         first_step = k + 2, last_step = ntime,
                         corrector_steps = s.corrector_steps, h = h)
    # uMig: D = D0 and Pi = 0
    if !mig
        for n = 0:ntime
            memcpy!(ph.D0, ph.D, n)
            smul!(ph.Pi, n, 0.0)
        end
    end
    Ekin = [2.0 * correlation_energy(n, G, Hyb, β, h, k) for n = 0:ntime]
    EMF = zeros(ComplexF64, ntime + 1)
    Ecorr = [2.0 * correlation_energy(n, G, Σ, β, h, k) for n = 0:ntime]
    Eph = phonon_energy_quantum(ph.D, ph.Pi, ph)
    Etot = Ekin .+ EMF .+ Ecorr .+ Eph
    XX = [n == 0 ? zero(ComplexF64) : im * ph.D.less[n, n][1,1] for n = 0:ntime]
    rho = [density_matrix(G, n) for n = 0:ntime]
    return (G = G, Σ = Σ, Hyb = Hyb, phonon = ph, rho = rho, XX = XX, Ekin = Ekin,
            Enx_MF = EMF, Enx_corr = Ecorr, Eph = Eph, Etot = Etot)
end

# Output directory of a configuration
function _holstein_outdir(cfg::ExampleConfig)
    dir = cfg.output["dir"]
    dir = isabspath(dir) ? dir : joinpath(cfg.dir, dir)
    mkpath(dir)
    return dir
end

"""
    holstein_bethe_main(args, approx; io = stdout)

Entry of the normal-state Bethe examples. Writes `<prefix>_obs.dat` with
the NESSi columns `t, rho.re, rho.im, Xph, Pph, XX, Ekin, Exn_MF,
Exn_corr, Eph_cl, Eph_corr, Etot`.
"""
function holstein_bethe_main(args, approx::AbstractString; io::IO = stdout)
    a = parse_example_args(args)
    cfg = load_holstein_bethe_config(a.cfgfile, a.profile)
    approx in HOLSTEIN_APPROX || throw(ArgumentError("unknown approximation $approx"))
    a.validate_only && return (config = cfg,)
    r = run_holstein_bethe(approx, cfg; io = io)
    h = contour_params(cfg).h
    rows = [[(n - 1) * h, real(r.rho[n+1][1,1]), imag(r.rho[n+1][1,1]), real(r.Xph[n+1]),
             real(r.Pph[n+1]), real(r.XX[n+1]), real(r.Ekin[n+1]), real(r.Enx_MF[n+1]),
             real(r.Enx_corr[n+1]), real(r.Eph_cl[n+1]), real(r.Eph_corr[n+1]),
             real(r.Etot[n+1])] for n = 1:getntime(r.G)]
    write_table(joinpath(_holstein_outdir(cfg), cfg.output["prefix"] * "_obs.dat"),
                ["time", "rho(0,0).re", "rho(0,0).im", "Xph", "Pph", "XX", "Ekin", "Exn_MF",
                 "Exn_corr", "Eph_cl", "Eph_corr", "Etot"], rows)
    return merge((config = cfg,), r)
end

"""
    holstein_nambu_main(args, approx; io = stdout)

Entry of the Nambu Bethe examples. Writes `<prefix>_obs.dat` with the
NESSi columns `t, rho(a,b).re/im, XX, Ekin, Exn_MF, Exn_corr, Eph, Etot`.
"""
function holstein_nambu_main(args, approx::AbstractString; io::IO = stdout)
    a = parse_example_args(args)
    cfg = load_holstein_bethe_config(a.cfgfile, a.profile; nambu = true)
    approx in HOLSTEIN_APPROX || throw(ArgumentError("unknown approximation $approx"))
    a.validate_only && return (config = cfg,)
    r = run_holstein_nambu(approx, cfg; io = io)
    h = contour_params(cfg).h
    names = ["time"]
    for a = 0:1, b = 0:1
        push!(names, "rho($a,$b).re", "rho($a,$b).im")
    end
    append!(names, ["XX", "Ekin", "Exn_MF", "Exn_corr", "Eph", "Etot"])
    rows = map(1:getntime(r.G)) do n
        ρ = r.rho[n+1]
        v = [(n - 1) * h]
        for a = 1:2, b = 1:2
            push!(v, real(ρ[a,b]), imag(ρ[a,b]))
        end
        append!(v, real.([r.XX[n+1], r.Ekin[n+1], r.Enx_MF[n+1], r.Enx_corr[n+1], r.Eph[n+1],
                          r.Etot[n+1]]))
    end
    write_table(joinpath(_holstein_outdir(cfg), cfg.output["prefix"] * "_obs.dat"), names, rows)
    return merge((config = cfg,), r)
end

# Run an entry from the command line, a failure gives a nonzero exit code
function holstein_cli(f::Function, args, approx)
    try
        f(args, approx)
    catch e
        showerror(stderr, e)
        println(stderr)
        exit(1)
    end
end
