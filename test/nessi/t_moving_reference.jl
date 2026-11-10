#
# t_moving_reference.jl
#
# Pointwise comparison of the Phase 8 moving (truncated) solvers and
# examples with the NESSi reference data (generators in .nessi/kbref,
# see .omo/plans/08-phase8-moving-parallel.md, Wave 7). The NESSi time
# step `step` of a series record is the KB time step `step + 1`. The
# primary condition is isapprox(actual, expected; atol = 1e-10,
# rtol = 1e-10) for every record.
#

isdefined(Main, :kb_test_profile) ||
    include(joinpath(@__DIR__, "..", "common", "profile.jl"))
isdefined(Main, :read_kbref) || include(joinpath(@__DIR__, "common.jl"))

using Test
using KadanoffBaym

module MovRefTB
include(joinpath(@__DIR__, "..", "..", "tutor", "trunc_bethe", "trunc_bethe_model.jl"))
end
module MovRefNBT
include(joinpath(@__DIR__, "..", "..", "tutor", "ness2_bethe_trunc", "ness2_bethe_trunc.jl"))
end
module MovRefGW
include(joinpath(@__DIR__, "..", "..", "tutor", "gw", "gw.jl"))
end

# Parameters of p8_gw_ref (see .nessi/kbref/p8_gw_ref.cpp), KB Ntau counts
# the points (NESSi Ntau + 1)
const MOVREF_GW_TOML = """
[model]
Nk = 4
HoppingT = 1.0
HubbardU = 1.0
V = 0.25
MuChem = 0.0
beta = 5.0

[contour]
Nt = 7
Ntau = 31
h = 0.02

[solver]
SolveOrder = 5
MatsMaxIter = 60
MatsMaxErr = 1e-9
BootstrapMaxIter = 40
BootstrapMaxErr = 1e-10
TimeMaxErr = 1e-10
CorrectorSteps = 6
parallel_axis = "serial"
current_mode = "nessi"

[pulse]
E0 = 1.0
omega = 2.0
t0 = 0.05
width = 0.1
"""

const MOVREF_DATA = joinpath(@__DIR__, "data")

movref_ok(a, b) = isapprox(a, b; atol = 1e-10, rtol = 1e-10)

# Inputs of p8_moving_solver_ref (see .nessi/kbref/p8_moving_solver_ref.cpp)
function movref_solver_case(size::Int, k::Int, md)
    nt = parse(Int, md["nt"])
    ntau = parse(Int, md["ntau"]) + 1
    tc = parse(Int, md["tc"])
    β, h, μ = 1.0, 0.02, -0.1
    ntime = nt + 1
    C = Cn(ntime, ntau, size, size, nt * h, β)
    H0 = size == 1 ? fill(ComplexF64(-0.3), 1, 1) :
         ComplexF64[-0.3 0.2im; -0.2im 0.4]
    Σ = ℱ(C, FERMI)
    green_from_H!(Σ, μ, H0, β, h)
    for n = 1:ntime, m = 1:n
        f = 0.3 * exp(-((n - m) * h)^2)
        Σ.ret[n, m] = f .* Σ.ret[n, m]
        Σ.less[m, n] = f .* Σ.less[m, n]
    end
    fill!(Σ.lmix.data, zero(ComplexF64))
    fill!(Σ.mat.data, zero(ComplexF64))
    Hf = Cf(ntime, size, size)
    for i = 0:ntime
        Hf[i] = H0
    end
    G = ℱ(C, FERMI)
    dyson!(G, μ, Hf, Σ, β, h; solve_order = k)
    mk(X) = (M = ℱᵐᵒᵛ(tc, size, getsign(X)); set_from_G_backward!(M, X, ntime); M)
    Hm = Cfᵐᵒᵛ(tc, size)
    set_from_function_backward!(Hm, Hf, ntime)
    return (Gm = mk(G), Sm = mk(Σ), Wm = mk(ℱ(C, BOSE)), Hm = Hm, ntime = ntime,
            tc = tc, μ = μ, h = h)
end

@testset verbose = true "KadanoffBaym: Phase 8 NESSi reference" begin
    @testset "moving data and algebra (p8_moving_case*.dat)" begin
        d₀ = read_kbref(joinpath(MOVREF_DATA, "p8_moving_case0.dat"))
        x = movref_solver_case(2, 3, merge(d₀.metadata, Dict("nt" => "30", "ntau" => "40")))
        G, S, n = x.Gm, x.Sm, x.ntime
        F(m) = ComplexF64[1+0.1*m*x.h 0.2im; -0.1 0.8-0.05*m*x.h]
        # Window of the full G after the four NESSi operations
        states = Any[deepcopy(G)]
        forward!(G)
        extrapolate_timestep!(G, 3)
        push!(states, deepcopy(G))
        cf = Cf(n + 1, 2, 2)
        for m = 1:n+1
            cf[m] = F(m - 1)
        end
        Fm = Cfᵐᵒᵛ(x.tc, 2)
        set_from_function_backward!(Fm, cf, n + 1)
        v = timestep_view(G, 0)
        smul!(Fm, v)
        smul!(v, Fm)
        push!(states, deepcopy(G))
        forward!(S)
        # The leading row of S is set to the row 1 (see p8_moving_ref.cpp)
        memcpy!(timestep_view(S, 1), timestep_view(S, 0))
        v = timestep_view(G, 0)
        smul!(v, 0.7)
        incr!(v, timestep_view(S, 0), 0.3)
        push!(states, deepcopy(G))
        ok = true
        used = 0
        for c = 0:3
            d = read_kbref(joinpath(MOVREF_DATA, "p8_moving_case$c.dat"))
            X = states[c+1]
            # NESSi time step tstp is the KB time step tstp + 1
            @test gettstp(X) == parse(Int, d.metadata["tstp"]) + 1
            for r in d.rows
                A = r.component == "ret" ? X.ret : X.less
                ok &= movref_ok(A[r.i, r.j][r.a+1, r.b+1], r.value)
                used += 1
            end
        end
        @test ok
        @test used == 4 * 2 * (x.tc + 1)^2 * 4
    end

    @testset "moving solvers (p8_moving_solver.dat)" begin
        d = read_kbref(joinpath(MOVREF_DATA, "p8_moving_solver.dat"))
        cases = ((1, 1), (1, 3), (1, 5), (2, 1), (2, 3), (2, 5))
        used = 0
        ok = true
        for (c, (size, k)) in enumerate(cases)
            x = movref_solver_case(size, k, d.metadata)
            n = x.ntime
            rows = filter(r -> r.i == c - 1, d.rows)
            # Bubbles of the orbitals a (G) and b (Σ), before the Dyson step
            B1 = Dict{Tuple{Int,Int},Any}()
            B2 = Dict{Tuple{Int,Int},Any}()
            for a = 1:size, b = 1:size
                W1 = deepcopy(x.Wm)
                W2 = deepcopy(x.Wm)
                bubble1!(n, W1, 1, 1, x.Gm, x.Gm, a, a, x.Sm, x.Sm, b, b)
                bubble2!(n, W2, 1, 1, x.Gm, x.Gm, a, a, x.Sm, x.Sm, b, b)
                B1[(a, b)] = W1
                B2[(a, b)] = W2
            end
            kin = kinetic_energy_trunc(n, x.Gm, x.Sm, Integrator(k), x.h)
            # Truncated VIE2 G + Σ * G = G (Q = the window of G), before the
            # Dyson step changes the leading row of x.Gm
            Gv = deepcopy(x.Gm)
            vie2_timestep!(n, Gv, x.Sm, x.Sm, x.Gm, Integrator(k), x.h)
            dyson_timestep!(n, x.Gm, x.μ, x.Hm, x.Sm, x.h; solve_order = k)
            for r in rows
                a, b = r.a + 1, r.b + 1
                if r.component == "kinetic"
                    v = kin
                elseif startswith(r.component, "dyson")
                    X = r.component == "dyson_ret" ? x.Gm.ret : x.Gm.less
                    v = X[0, r.j][a, b]
                elseif startswith(r.component, "vie2")
                    X = r.component == "vie2_ret" ? Gv.ret : Gv.less
                    v = X[0, r.j][a, b]
                else
                    W = startswith(r.component, "bubble1") ? B1[(a, b)] : B2[(a, b)]
                    X = endswith(r.component, "_ret") ? W.ret : W.less
                    v = X[0, r.j][1, 1]
                end
                ok &= movref_ok(v, r.value)
                used += 1
            end
        end
        @test ok
        @test used == length(d.rows)
    end

    @testset "trunc Bethe (p8_trunc_bethe.dat)" begin
        d = read_kbref(joinpath(MOVREF_DATA, "p8_trunc_bethe.dat"))
        md = d.metadata
        tc = parse(Int, md["tc"])
        # KB ntau counts the points (NESSi ntau + 1)
        p = (nt = parse(Int, md["nt"]), ntau = parse(Int, md["ntau"]) + 1, h = 0.02,
             beta = 5.0, tc = tc, tmax = parse(Int, md["nsteps"]) - 1, U = 1.0,
             epsk = Float64[], SolveOrder = 5, BootstrapMaxIter = 40,
             BootstrapMaxErr = 1e-10, CorrectorSteps = 3)
        start = MovRefTB.solve_bethe_start!(p)
        st = MovRefTB.initialize_moving_state(start, tc)
        # Densities of the window rows at the handoff (NESSi steps 0..tc)
        dens = Dict(tc - j => MovRefTB.moving_density(st.G, j) for j = 0:tc)
        out = MovRefTB.propagate_bethe_moving!(st, p)
        # Records of the propagation, keyed by the NESSi step
        for r in out
            dens[r.tstp - 1] = r.density
        end
        last = p.tmax
        used = 0
        ok = true
        for r in d.rows
            s = r.i
            if r.component == "density"
                v = dens[s]
            elseif r.component in ("error", "ekin")
                o = out[s - tc]
                @assert o.tstp == s + 1
                v = r.component == "error" ? o.err : o.ekin
            else
                # Two-time leading rows, only the last step is kept in memory
                s == last || continue
                X = startswith(r.component, "g_") ? st.G : st.Σ
                v = endswith(r.component, "_ret") ? X.ret[0, r.j][1] : X.less[0, r.j][1]
            end
            ok &= movref_ok(v, r.value)
            used += 1
        end
        @test ok
        # every scalar record and the leading rows of the last step
        @test used == count(r -> r.component in ("density", "error", "ekin") || r.i == last,
                            d.rows)
    end

    @testset "NESS2 trunc Bethe (p8_ness2_bethe_trunc.dat)" begin
        d = read_kbref(joinpath(MOVREF_DATA, "p8_ness2_bethe_trunc.dat"))
        dir = mktempdir()
        cfg = joinpath(dir, "nbt.toml")
        txt = read(joinpath(@__DIR__, "..", "..", "tutor", "ness2_bethe_trunc",
                            "ness2_bethe_trunc.toml"), String)
        txt = replace(txt, r"dir = \".*\"" => "dir = \"$(escape_string(dir))\"",
                      r"Nft = \d+" => "Nft = 256", r"Niter = \d+" => "Niter = 20",
                      r"tmax = \d+" => "tmax = $(parse(Int, d.metadata["nsteps"]) - 1)")
        write(cfg, txt)
        res = redirect_stdout(devnull) do
            MovRefNBT.main(cfg)
        end
        last = length(res.steps)
        dens = Dict(0 => MovRefNBT.moving_density(res.initial.G))
        for s in res.steps
            dens[s.rel] = s.density
        end
        ok = true
        used = 0
        for r in d.rows
            if r.component == "density"
                v = dens[r.i]
            elseif r.i == 0 || r.i == last
                X = r.i == 0 ? res.initial.G : res.G
                v = r.component == "g_ret" ? X.ret[0, r.j][1] : X.less[0, r.j][1]
            else
                continue
            end
            ok &= movref_ok(v, r.value)
            used += 1
        end
        @test ok
        @test used == count(r -> r.component == "density" || r.i in (0, last), d.rows)
        # NESS relative step 0 is the KB time step 1
        @test gettstp(res.initial.G) == 1
    end

    @testset "GW (p8_gw.dat)" begin
        d = read_kbref(joinpath(MOVREF_DATA, "p8_gw.dat"))
        dir = mktempdir()
        cfg = joinpath(dir, "gw.toml")
        write(cfg, MOVREF_GW_TOML * "\n[output]\ndir = \"$(escape_string(dir))\"\nprefix = \"gw\"\n")
        res = MovRefGW.main(cfg; io = devnull)
        kps, lat = res.state.kps, res.state.lat
        rows = res.observables
        ok = true
        for r in d.rows
            n = r.i + 1          # KB time step
            ob = rows[n+1]
            @assert ob.tstp == n
            c = r.component
            v = if c == "afield"
                lat.A[n+1]
            elseif c == "density"
                ob.density
            elseif c == "current_nessi"
                ob.current_nessi
            elseif c == "ekin"
                ob.Ekin
            elseif c == "eint"
                ob.Eint
            else
                kp = kps[r.j+1]
                if c == "rho"
                    kp.rho[n][1]
                elseif c == "sigma_hf"
                    (kp.SH[n] + kp.SF[n])[1]
                else
                    X = startswith(c, "g_") ? kp.G : startswith(c, "w_") ? kp.W :
                        startswith(c, "p_") ? kp.P : kp.Σ
                    endswith(c, "_ret") ? X.ret[n, 1][1] : X.less[1, n][1]
                end
            end
            ok &= movref_ok(v, r.value)
        end
        @test ok
        @test length(d.rows) == 8 * (5 + 4 * 10)
    end
end
