#
# Project : Lavender
# Source  : hubbard_bethe_ness2.jl
# Author  : Li Huang (huangli@caep.cn)
# Status  : Unstable
#
# Last modified: 2026/11/10
#

#
# Usage:
#
#   julia --project=. tutor/hubbard_bethe/hubbard_bethe_ness2.jl \
#         tutor/hubbard_bethe/hubbard_bethe_ness2.toml
#
# Steady-state DMFT of the Hubbard model on the Bethe lattice with the
# bandwidth 4. The local Green's function solves
#
#   G = [i∂_t + μ - Σ - Δ]⁻¹,  Δ = G,
#
# where the second order self-energy is
#
#   Σ(t) = -U² i 𝒢(t) [i 𝒢(t) 𝒢(-t)],
#
# with 𝒢 = G (self-consistent PT, ipt_flag = 0) or the Weiss field
# 𝒢 = [i∂_t + μ - Δ]⁻¹ (IPT, ipt_flag = 1). The Dyson equation is solved
# with the kernel K = Σ + G, the iterations use linear mixing and stop if
# the error is below `errmax` after at least four iterations.
#
# Output (in the directory `dir`, file names start with `prefix`):
#
#   * prefix_iterations.dat  : iteration, error, density;
#   * prefix_observables.dat : density, interaction energy, iterations;
#   * prefix_G_time.dat, prefix_G_freq.dat, prefix_S_time.dat,
#     prefix_S_freq.dat      : final G and Σ (real and imaginary parts of
#                              the retarded and lesser components);
#   * prefix_G_itN.dat, prefix_S_itN.dat : G and Σ (time domain) of the
#                              iteration N, if out_every > 0.
#
# Port of NESSi examples/programs/ness2_bethe.cpp. HDF5 is not supported.
# NESSi writes G and Σ of every iteration if out_every ≠ 0.
#

using Printf
using KadanoffBaym

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

Read the parameters from `cfgfile` (TOML), run the DMFT loop, and write
the results. Return the observables, the error history, and the final
Green's function `G` (both domains) as a named tuple.
"""
function main(cfgfile::String)
    p = inp_toml(cfgfile, true)
    Nft = p["grid"]["Nft"]::Int
    h = Float64(p["grid"]["h"])
    β = Float64(p["model"]["beta"])
    U = Float64(p["model"]["U"])
    μ = Float64(p["model"]["mu"])
    Niter = p["solver"]["Niter"]::Int
    errmax = Float64(p["solver"]["errmax"])
    mix = Float64(p["solver"]["mixing"])
    ipt = p["solver"]["ipt_flag"]::Int
    out_every = p["solver"]["out_every"]::Int
    dir = String(p["output"]["dir"])
    prefix = String(p["output"]["prefix"])
    @assert iseven(Nft) && Nft ≥ 4
    @assert 0.0 < mix ≤ 1.0
    @assert h > 0.0 && β > 0.0 && Niter ≥ 1 && out_every ≥ 0
    mkpath(dir)

    g = NessGrid(Nft, h)
    G = NessHermMat(g, 1)
    SU = NessHermMat(g, 1)
    W = NessHermMat(g, 1)
    K = NessHermMat(g, 1)
    Gweiss = NessHermMat(g, 1)
    ham = zeros(ComplexF64, 1, 1)

    # Initial guess, the noninteracting Green's function
    from_dos_ness!(G, FERMI, BetheDOS(-2.0, 2.0), β, μ)

    # Second order self-energy of the field 𝒢
    function sigma!(𝒢)
        bubble1_ness!(W, 𝒢, 𝒢)
        smul!(W, U * U, FFT_TIME)
        bubble2_ness!(SU, 𝒢, W)
        smul!(SU, -1.0, FFT_TIME)
    end

    errs = Float64[]
    itermax = Niter
    fit = open(joinpath(dir, prefix * "_iterations.dat"), "w")
    println(fit, "# iteration  error  density")
    for iter = 1:Niter
        Gold = deepcopy(G)

        # Self-energy
        if ipt != 0
            dyson_ness!(Gweiss, μ, ham, G)
        else
            memcpy!(G, Gweiss)
        end
        sigma!(Gweiss)

        # Kernel K = Σ + G and the Dyson equation
        zeros!(K, FFT_TIME)
        incr!(K, SU, 1.0, FFT_TIME)
        incr!(K, G, 1.0, FFT_TIME)
        dyson_ness!(G, μ, ham, K)
        err = distance(G, Gold, FFT_TIME)

        # Linear mixing
        smul!(G, mix, FFT_TIME)
        incr!(G, Gold, 1.0 - mix, FFT_TIME)

        push!(errs, err)
        dens = 2.0 * real(density_matrix(G, FERMI)[1, 1])
        @printf(fit, "%d %.12e %.12e\n", iter, err, dens)
        @printf("iteration %d: error = %.6e\n", iter, err)
        if out_every > 0 && iter % out_every == 0
            write_ness_gf(joinpath(dir, prefix * "_G_it$iter.dat"), G, FFT_TIME)
            write_ness_gf(joinpath(dir, prefix * "_S_it$iter.dat"), SU, FFT_TIME)
        end
        if err < errmax && iter > 3
            itermax = iter
            println("converged after $itermax iterations")
            break
        end
    end
    close(fit)

    # Observables
    SUG = convolution_density_matrix(SU, G, FERMI)
    Eint = real(SUG[1, 1])
    dens = 2.0 * real(density_matrix(G, FERMI)[1, 1])
    open(joinpath(dir, prefix * "_observables.dat"), "w") do io
        println(io, "# density  Eint  iterations")
        @printf(io, "%.12e %.12e %d\n", dens, Eint, itermax)
    end

    # Spectra
    integral_transform_to_freq!(G, FFT_TRAPEZ)
    integral_transform_to_freq!(SU, FFT_TRAPEZ)
    write_ness_gf(joinpath(dir, prefix * "_G_time.dat"), G, FFT_TIME)
    write_ness_gf(joinpath(dir, prefix * "_G_freq.dat"), G, FFT_FREQ)
    write_ness_gf(joinpath(dir, prefix * "_S_time.dat"), SU, FFT_TIME)
    write_ness_gf(joinpath(dir, prefix * "_S_freq.dat"), SU, FFT_FREQ)

    @printf("density = %.8f, Eint = %.8f, iterations = %d\n", dens, Eint, itermax)

    return (density = dens, Eint = Eint, iterations = itermax, errors = errs, G = G,
            S = SU, Gweiss = Gweiss)
end

if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) == 1 || error("usage: bethe_ness2.jl <config.toml>")
    main(ARGS[1])
end
