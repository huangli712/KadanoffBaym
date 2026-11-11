#
# Project : Lavender
# Source  : siam_ness2.jl
# Author  : Li Huang (huangli@caep.cn)
# Status  : Unstable
#
# Last modified: 2026/11/10
#

#
# Usage:
#
#   julia --project=. tutor/anderson_ness/siam_ness2.jl \
#         tutor/anderson_ness/siam_ness2.toml
#
# Steady state of the single impurity Anderson model. The impurity level
# `epsd` is coupled to the left and right leads (smooth box density of
# states, chemical potentials ±V/2). The noninteracting Green's function
# G₀ is obtained from the steady-state Dyson equation with
# Σ_bath = Σ_L + Σ_R, the interacting one with Σ = Σ_bath + Σ_U, where
#
#   Σ_U(t) = -U² i G₀(t) [i G₀(t) G₀(-t)],
#
# is the second order self-energy (Bubble1_ness, Bubble2_ness).
#
# Output (in the directory `dir`, file names start with `prefix`):
#
#   * prefix_observables.dat : density, interaction energy, current, and
#                              the difference of the two current formulas;
#   * prefix_G_time.dat      : t, G^R(t), G^<(t) (real and imaginary parts);
#   * prefix_G_freq.dat      : ω, G^R(ω), G^<(ω) (real and imaginary parts).
#
# Port of NESSi examples/programs/ness2_siam.cpp. HDF5 is not supported.
# NESSi computes the convolution density matrix [SL * G] into `SLG` and
# then overwrites `SLG` with [G * SL] (`GSL` stays empty), here `SLG` and
# `GSL` are computed separately.
#

using Printf
using KadanoffBaym

"""
    BoxDOS

Smooth box density of states of the leads,
``A(\\omega) = 1/[\\pi (1 + e^{\\nu(\\omega - \\omega_c)})
(1 + e^{-\\nu(\\omega_c + \\omega)})]``, with ``\\omega_c = 10`` and
``\\nu = 3``.
"""
struct BoxDOS
    lo :: Float64
    hi :: Float64
    wc :: Float64
    nu :: Float64
end

BoxDOS() = BoxDOS(-15.0, 15.0, 10.0, 3.0)

(d::BoxDOS)(w::Real) = 1.0 / ((1 + exp(d.nu * (w - d.wc))) * (1 + exp(-d.nu * (d.wc + w)))) / π

"""
    write_ness_gf(fname::String, G::NessHermMat, dom::FFTDomain)

Write the (1,1) elements of the retarded and lesser components of `G` in
the domain `dom` into the text file `fname`.
"""
function write_ness_gf(fname::String, G::NessHermMat, dom::FFTDomain)
    g = G.grid
    open(fname, "w") do io
        if dom == FFT_TIME
            println(io, "# t  Re G^R  Im G^R  Re G^<  Im G^<")
        else
            println(io, "# omega  Re G^R  Im G^R  Re G^<  Im G^<")
        end
        for n = 1:getnft(G)
            x = dom == FFT_TIME ? time_at(g, n) : freq_at(g, n)
            r = get_ret(G, n, dom)[1, 1]
            l = get_les(G, n, dom)[1, 1]
            @printf(io, "%.12e %.12e %.12e %.12e %.12e\n", x, real(r), imag(r), real(l), imag(l))
        end
    end
end

"""
    main(cfgfile::String)

Read the parameters from `cfgfile` (TOML), solve the steady state, and
write the results. Return the observables and the interacting Green's
function `G` (both domains) as a named tuple.
"""
function main(cfgfile::String)
    p = inp_toml(cfgfile, true)
    U = Float64(p["model"]["U"])
    β = Float64(p["model"]["beta"])
    V = Float64(p["model"]["V"])
    εd = Float64(p["model"]["epsd"])
    η = Float64(p["model"]["eta"])
    Nft = p["grid"]["Nft"]::Int
    h = Float64(p["grid"]["h"])
    dir = String(p["output"]["dir"])
    prefix = String(p["output"]["prefix"])
    @assert iseven(Nft) && Nft ≥ 4
    @assert h > 0.0 && β > 0.0 && η ≥ 0.0
    mkpath(dir)

    g = NessGrid(Nft, h)
    G₀ = NessHermMat(g, 1)
    G = NessHermMat(g, 1)
    SL = NessHermMat(g, 1)
    SR = NessHermMat(g, 1)
    SU = NessHermMat(g, 1)
    W = NessHermMat(g, 1)

    # Bath self-energies
    dos = BoxDOS()
    from_dos_ness!(SL, FERMI, dos, β, +0.5 * V)
    from_dos_ness!(SR, FERMI, dos, β, -0.5 * V)
    Sbath = deepcopy(SL)
    incr!(Sbath, SR, 1.0, FFT_TIME)

    # Noninteracting Dyson equation
    ε = fill(ComplexF64(εd), 1, 1)
    μ = 0.0
    dyson_ness!(G₀, μ, ε, Sbath; bath_type = BATH_GAUSS, bath_eta = η,
                bath_beta = β, bath_mu = μ)

    # Second order diagram
    bubble1_ness!(W, G₀, G₀)
    smul!(W, U * U, FFT_TIME)
    bubble2_ness!(SU, G₀, W)
    smul!(SU, -1.0, FFT_TIME)

    # Interacting Dyson equation
    S = deepcopy(Sbath)
    incr!(S, SU, 1.0, FFT_TIME)
    dyson_ness!(G, μ, ε, S; bath_type = BATH_GAUSS, bath_eta = η,
                bath_beta = β, bath_mu = μ)

    # Observables
    SLG = convolution_density_matrix(SL, G, FERMI)
    GSL = convolution_density_matrix(G, SL, FERMI)
    SUG = convolution_density_matrix(SU, G, FERMI)
    ρ = density_matrix(G, FERMI)
    current = 4.0 * imag(SLG[1, 1])
    current₁ = -4.0 * imag(GSL[1, 1])
    Eint = real(SUG[1, 1])
    dens = 2.0 * real(ρ[1, 1])

    open(joinpath(dir, prefix * "_observables.dat"), "w") do io
        println(io, "# density  Eint  current  current_GSL - current_SLG")
        @printf(io, "%.12e %.12e %.12e %.12e\n", dens, Eint, current, current₁ - current)
    end

    # Spectra
    integral_transform_to_freq!(G, FFT_TRAPEZ)
    write_ness_gf(joinpath(dir, prefix * "_G_time.dat"), G, FFT_TIME)
    write_ness_gf(joinpath(dir, prefix * "_G_freq.dat"), G, FFT_FREQ)

    @printf("density = %.8f, Eint = %.8f, current = %.8f, hermiticity of current: %.3e\n",
            dens, Eint, current, current₁ - current)

    return (density = dens, Eint = Eint, current = current, current_GSL = current₁, G = G)
end

if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) == 1 || error("usage: siam_ness2.jl <config.toml>")
    main(ARGS[1])
end
