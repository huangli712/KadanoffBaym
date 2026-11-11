#
# Project : Lavender
# Source  : holstein_impurity_impl.jl
# Author  : Li Huang (huangli@caep.cn)
# Status  : Unstable
#
# Last modified: 2026/11/10
#

#
# Holstein impurity coupled to a single bath site, self-consistent
# (Migdal) or unrenormalized (uMig) Migdal self-energy. Port of NESSi
# examples/programs/Holstein_impurity_singlebath_{Migdal,uMig}.cpp (no
# HDF5). The hybridization Δ = J G0_11 J of the free bath site is fixed.
# As in tutor/holstein_bethe, the mean-field terms use the real part of
# the density (the hermitian part of the density matrix).
#

using LinearAlgebra
using Printf
using KadanoffBaym

include(joinpath(@__DIR__, "..", "common", "example_config.jl"))
include(joinpath(@__DIR__, "..", "common", "text_output.jl"))
include(joinpath(@__DIR__, "..", "common", "contour_loop.jl"))
include(joinpath(@__DIR__, "..", "holstein_bethe", "holstein_impl.jl"))
include(joinpath(@__DIR__, "..", "holstein_bethe", "holstein_energy.jl"))

const IMPURITY_APPROX = ("Migdal", "uMig")
const IMPURITY_KEYS = ("Hopping", "El_Ph_g", "Phfreq_w0", "Eps0_MF", "Eps1")

# Values of a drive at the KB time steps 0, ⋯, ntime
function impurity_drive(cfg::ExampleConfig, name::AbstractString)
    ntime = contour_params(cfg).ntime
    f = Cf(ntime, 1, 1, zero(ComplexF64))
    load_tvector!(f, cfg, name)
    return [real(f[n][1,1]) for n = 0:ntime]
end

"""
    load_impurity_config(cfgfile, profile)

Read and validate the TOML file of the impurity examples.
"""
function load_impurity_config(cfgfile::AbstractString, profile::AbstractString)
    cfg = load_example_config(cfgfile; profile = profile, model_required = IMPURITY_KEYS,
                              drives = ("dHopping", "dEl_Ph_g"))
    solver_params(cfg)
    for k in IMPURITY_KEYS
        cfg.model[k] isa Real || throw(ArgumentError("$cfgfile: [model] $k must be a number"))
    end
    cfg.model["Phfreq_w0"] > 0.0 || throw(ArgumentError("$cfgfile: [model] Phfreq_w0 must be > 0"))
    impurity_drive(cfg, "dHopping")
    impurity_drive(cfg, "dEl_Ph_g")
    return cfg
end

# Matsubara (FIXPOINT), start-up, and time steps of a Dyson equation, as
# the NESSi post-processing
function _dyson_all!(G::ℱ, H::Cf, Σ::ℱ, β::Float64, h::Float64, k::Int)
    dyson_mat!(G, Σ, 0.0, H, β; solve_order = k, method = :fixpoint)
    dyson_start!(G, 0.0, H, Σ, β, h; solve_order = k)
    for n = k+2:getntime(G)
        dyson_timestep!(n, G, 0.0, H, Σ, β, h; solve_order = k)
    end
    return G
end

"""
    run_holstein_impurity(approx, cfg; io = stdout)

Solve the impurity problem. Returns the Green's functions and the
observables at all KB time steps (`n + 1` indexing): `rho00`, `rho11`,
`Xph`, `Pph`, `XX`, `Ekin`, `Enx_MF`, `Enx_corr`, `Eph_cl`, `Eph_corr`,
`Etot`.
"""
function run_holstein_impurity(approx::AbstractString, cfg::ExampleConfig; io::IO = stdout)
    approx in IMPURITY_APPROX || throw(ArgumentError("unknown approximation $approx"))
    mig = approx == "Migdal"
    c, s, m = contour_params(cfg), solver_params(cfg), cfg.model
    ntime, k, β, h = c.ntime, c.k, c.beta, c.h
    w0 = Float64(m["Phfreq_w0"])
    Eps0MF, Eps1 = Float64(m["Eps0_MF"]), Float64(m["Eps1"])
    gv = ComplexF64.(m["El_Ph_g"] .+ impurity_drive(cfg, "dEl_Ph_g"))
    Jv = ComplexF64.(m["Hopping"] .+ impurity_drive(cfg, "dHopping"))
    g = holstein_scalar_cf(ntime, gv)
    J = holstein_scalar_cf(ntime, Jv)
    C = Cn(ntime, c.ntau_points, 1, 1, (ntime - 1) * h, β)
    F() = ℱ(C, FERMI)
    G00, Hyb, Σ, HS, Gold, G011, JG00J, G11 = F(), F(), F(), F(), F(), F(), F(), F()
    ph = holstein_phonon(ntime, c.ntau_points, 1, w0, β, h, k)
    h00MF = holstein_scalar_cf(ntime, fill(Eps0MF, ntime + 1))
    h11 = holstein_scalar_cf(ntime, fill(Eps1, ntime + 1))
    ntot = zeros(ComplexF64, ntime + 1)
    X = zeros(ComplexF64, ntime + 1)
    from_ham!(G011, 0.0, fill(ComplexF64(Eps1), 1, 1), β, h)
    from_ham!(G00, 0.0, fill(ComplexF64(Eps0MF), 1, 1), β, h)
    for n = 0:ntime
        holstein_hybridization!(n, Hyb, G011, J)
    end
    # Matsubara
    stage = MatsubaraTwoStage(false, false)
    memcpy!(G00, Gold, 0)
    iterate_matsubara!(; max_iter = s.mats_max_iter, tol = s.mats_max_error,
                       converged = (iter, err) -> stage.converged) do iter
        meth = matsubara_method(stage)
        mig ? sigma_mig!(0, G00, Σ, g, ph; method = meth) : sigma_umig!(0, G00, Σ, g, ph)
        _hyb_sigma!(0, HS, Hyb, Σ)
        dyson_mat!(G00, HS, 0.0, h00MF, β; solve_order = k, method = meth)
        smul!(G00, 0, 0.5)
        incr!(G00, Gold, 0, 0.5)
        err = distance(G00, Gold, 0)
        @printf(io, "matsubara (%s) iteration %d  error = %.6e\n", meth, iter, err)
        matsubara_update!(stage, err, s.mats_max_error) && memcpy!(G00, Gold, 0)
        err
    end
    ntot[1] = 2.0 * real(density_matrix(G00, 0)[1,1])
    phonon_displacement!(0, X, ntot, gv, ph)
    h00 = Eps0MF - real(X[1] * gv[1])
    set_tk_from_mat!(G00, k)
    # Bootstrapping
    for n = 1:k+1
        memcpy!(G00, Gold, n)
    end
    iterate_bootstrap!(; max_iter = s.bootstrap_max_iter, tol = s.bootstrap_max_error,
                       min_iterations = 3) do iter
        mig ? sigma_mig_start!(G00, Σ, g, ph) : sigma_umig_start!(G00, Σ, g, ph)
        for n = 1:k+1
            ntot[n+1] = 2.0 * real(density_matrix(G00, n)[1,1])
        end
        phonon_displacement_start!(X, ntot, gv, ph)
        for n = 1:k+1
            h00MF[n] = fill(h00 + X[n+1] * gv[n+1], 1, 1)
            _hyb_sigma!(n, HS, Hyb, Σ)
        end
        dyson_start!(G00, 0.0, h00MF, HS, β, h; solve_order = k)
        err = 0.0
        for n = 1:k+1
            err += distance(G00, Gold, n)
        end
        @printf(io, "bootstrap iteration %d  error = %.6e\n", iter, err)
        if !(err < s.bootstrap_max_error && iter > 2)
            for n = 1:k+1
                memcpy!(G00, Gold, n)
            end
        end
        err
    end
    # Propagation
    propagate_timesteps!(n -> extrapolate_timestep!(n - 1, G00, k),
                         (n, iter) -> begin
                             mig ? sigma_mig!(n, G00, Σ, g, ph) : sigma_umig!(n, G00, Σ, g, ph)
                             ntot[n+1] = 2.0 * real(density_matrix(G00, n)[1,1])
                             phonon_displacement!(n, X, ntot, gv, ph)
                             h00MF[n] = fill(h00 + X[n+1] * gv[n+1], 1, 1)
                             _hyb_sigma!(n, HS, Hyb, Σ)
                             dyson_timestep!(n, G00, 0.0, h00MF, HS, β, h; solve_order = k)
                         end;
                         first_step = k + 2, last_step = ntime,
                         corrector_steps = s.corrector_steps, h = h)
    # G'00 = [i∂t - h00_MF - Σ]^{-1}, G11 = [i∂t - h11 - J G'00 J]^{-1}
    _dyson_all!(JG00J, h00MF, Σ, β, h, k)
    for n = 0:ntime
        smul!(J, JG00J, n)
        smul!(JG00J, J, n)
    end
    _dyson_all!(G11, h11, JG00J, β, h, k)
    P = phonon_momentum(ntot, gv, ph)
    # Energies
    if !mig
        for n = 0:ntime
            smul!(ph.Pi, n, 0.0)
        end
    end
    Dout = mig ? ph.D : ph.D0
    Ekin = map(0:ntime) do n
        ρ10J = correlation_energy(n, G00, Hyb, β, h, k)
        n1 = 2.0 * density_matrix(G11, n)[1,1]
        4.0 * (ρ10J + conj(ρ10J)) + Eps1 * n1 + h00 * ntot[n+1]
    end
    EMF = [gv[n+1] * X[n+1] * ntot[n+1] for n = 0:ntime]
    Ecorr = [4.0 * correlation_energy(n, G00, Σ, β, h, k) for n = 0:ntime]
    Ecl = phonon_energy_classical(X, P, w0)
    Equ = phonon_energy_quantum(Dout, ph.Pi, ph)
    Etot = Ekin .+ EMF .+ Ecorr .+ Ecl .+ Equ
    XX = [n == 0 ? zero(ComplexF64) : im * Dout.less[n, n][1,1] for n = 0:ntime]
    return (G00 = G00, G11 = G11, Σ = Σ, phonon = ph,
            rho00 = [density_matrix(G00, n)[1,1] for n = 0:ntime],
            rho11 = [density_matrix(G11, n)[1,1] for n = 0:ntime],
            ntot = ntot, Xph = X, Pph = P, XX = XX, Ekin = Ekin, Enx_MF = EMF, Enx_corr = Ecorr,
            Eph_cl = Ecl, Eph_corr = Equ, Etot = Etot)
end

"""
    holstein_impurity_main(args, approx; io = stdout)

Entry of the impurity examples. Writes `<prefix>_obs.dat` with the NESSi
columns `t, rho00.re/im, rho11.re/im, Xph, Pph, XX, Ekin, Exn_MF,
Exn_corr, Eph_cl, Eph_corr, Etot`.
"""
function holstein_impurity_main(args, approx::AbstractString; io::IO = stdout)
    a = parse_example_args(args)
    cfg = load_impurity_config(a.cfgfile, a.profile)
    approx in IMPURITY_APPROX || throw(ArgumentError("unknown approximation $approx"))
    a.validate_only && return (config = cfg,)
    r = run_holstein_impurity(approx, cfg; io = io)
    h = contour_params(cfg).h
    rows = [[(n - 1) * h, real(r.rho00[n+1]), imag(r.rho00[n+1]), real(r.rho11[n+1]),
             imag(r.rho11[n+1]), real(r.Xph[n+1]), real(r.Pph[n+1]), real(r.XX[n+1]),
             real(r.Ekin[n+1]), real(r.Enx_MF[n+1]), real(r.Enx_corr[n+1]), real(r.Eph_cl[n+1]),
             real(r.Eph_corr[n+1]), real(r.Etot[n+1])] for n = 1:getntime(r.G00)]
    dir = cfg.output["dir"]
    dir = isabspath(dir) ? dir : joinpath(cfg.dir, dir)
    mkpath(dir)
    write_table(joinpath(dir, cfg.output["prefix"] * "_obs.dat"),
                ["time", "rho00.re", "rho00.im", "rho11.re", "rho11.im", "Xph", "Pph", "XX",
                 "Ekin", "Exn_MF", "Exn_corr", "Eph_cl", "Eph_corr", "Etot"], rows)
    return merge((config = cfg,), r)
end

# Run an entry from the command line, a failure gives a nonzero exit code
function impurity_cli(args, approx)
    try
        holstein_impurity_main(args, approx)
    catch e
        showerror(stderr, e)
        println(stderr)
        exit(1)
    end
end
