#
# t_kbref_schema.jl
#
# Contract tests of the kbref-v1 plain-text reference format. This is the
# only schema test file of the project; later phases must extend it
# instead of creating copies.
#
# It requires test/nessi/common.jl.
#

@testset verbose = true "kbref-v1 schema" begin
    mktempdir() do dir
        #
        # Helper: write a file, optionally edit its text, and expect that
        # read_kbref rejects it.
        #
        function bad(name, kind, meta, rows; edit = identity)
            q = joinpath(dir, name)
            write_kbref(q, kind, meta, rows)
            write(q, edit(read(q, String)))
            @test_throws ArgumentError read_kbref(q)
        end

        @testset "P1: herm_matrix and weights" begin
            hmeta = ["case" => "selftest", "nt" => "1", "ntau" => "1",
                     "size1" => "1", "size2" => "1"]
            hrows = [
                KbrefRow("mat", 0, -1, 0, 0, 1.0 + 2.0im),
                KbrefRow("mat", 1, -1, 0, 0, -1.0 / 3.0 + 0.0im),
                KbrefRow("ret", 0, 0, 0, 0, 0.0 - 1.0im),
                KbrefRow("ret", 1, 0, 0, 0, 0.1 + 0.2im),
                KbrefRow("ret", 1, 1, 0, 0, 0.0 - 1.0im),
                KbrefRow("tv", 0, 0, 0, 0, 0.5 + 0.5im),
                KbrefRow("tv", 0, 1, 0, 0, pi + 0.0im),
                KbrefRow("tv", 1, 0, 0, 0, 1e-300 + 0.0im),
                KbrefRow("tv", 1, 1, 0, 0, -1e300 + 0.0im),
                KbrefRow("les", 0, 0, 0, 0, 0.0 + 0.3im),
                KbrefRow("les", 0, 1, 0, 0, 0.7 - 0.1im),
                KbrefRow("les", 1, 1, 0, 0, 0.0 + 0.3im),
            ]
            #
            # Round trip, herm_matrix
            p = joinpath(dir, "ok_herm.dat")
            write_kbref(p, "herm_matrix", hmeta, hrows)
            d = read_kbref(p)
            @test d.metadata["data_kind"] == "herm_matrix"
            @test d.metadata["case"] == "selftest"
            @test length(d.rows) == length(hrows)
            @test all(d.rows[n].value == hrows[n].value for n in eachindex(hrows))
            @test all(d.rows[n].component == hrows[n].component for n in eachindex(hrows))
            #
            # Round trip, weights
            wrows = [
                KbrefRow("greg", 1, 0, 0, -1, 5/12 + 0im),
                KbrefRow("bd", 1, 2, -1, -1, 0.5 + 0im),
                KbrefRow("pint", 1, 1, 0, 1, 0.5 + 0im),
                KbrefRow("pdiff", 1, 1, 0, -1, -1.0 + 0im),
                KbrefRow("rcorr", 2, 1, 2, 0, 0.1 + 0im),
            ]
            p = joinpath(dir, "ok_weights.dat")
            write_kbref(p, "weights", ["case" => "selftest"], wrows)
            d = read_kbref(p)
            @test d.metadata["data_kind"] == "weights"
            @test all(d.rows[n].value == wrows[n].value for n in eachindex(wrows))
            #
            # Bad inputs
            bad("kind.dat", "herm_matrix", hmeta, hrows;
                edit = s -> replace(s, "data_kind herm_matrix" => "data_kind unknown"))
            bad("version.dat", "herm_matrix", hmeta, hrows;
                edit = s -> replace(s, "# kbref 1" => "# kbref 2"))
            bad("columns.dat", "herm_matrix", hmeta, hrows;
                edit = s -> replace(s, "re im" => "re im extra"))
            bad("comp.dat", "herm_matrix", hmeta,
                [hrows[1:end-1]; KbrefRow("gtr", 1, 1, 0, 0, 0.0im)])
            bad("ncol6.dat", "herm_matrix", hmeta, hrows;
                edit = s -> replace(s, r"\n(les 1 1 0 0 \S+) \S+\n" => s"\n\1\n"))
            bad("ncol8.dat", "herm_matrix", hmeta, hrows;
                edit = s -> replace(s, r"\n(les 1 1 0 0 \S+ \S+)\n" => s"\n\1 0\n"))
            bad("dup.dat", "herm_matrix", hmeta, [hrows; hrows[end]])
            bad("ret_upper.dat", "herm_matrix", hmeta,
                [hrows[1:2]; KbrefRow("ret", 0, 1, 0, 0, 0.0im); hrows[4:end]])
            bad("les_lower.dat", "herm_matrix", hmeta,
                [hrows[1:end-1]; KbrefRow("les", 1, 0, 0, 0, 0.0im)])
            bad("orbital.dat", "herm_matrix", hmeta,
                [hrows[1:end-1]; KbrefRow("les", 1, 1, 1, 0, 0.0im)])
            bad("mat_j.dat", "herm_matrix", hmeta,
                [KbrefRow("mat", 0, 0, 0, 0, 0.0im); hrows[2:end]])
            bad("nan.dat", "herm_matrix", hmeta,
                [hrows[1:end-1]; KbrefRow("les", 1, 1, 0, 0, complex(NaN, 0.0))])
            bad("inf.dat", "herm_matrix", hmeta,
                [hrows[1:end-1]; KbrefRow("les", 1, 1, 0, 0, complex(0.0, Inf))])
            bad("incomplete.dat", "herm_matrix", hmeta, hrows[1:end-1])
            bad("noheader.dat", "herm_matrix", hmeta, hrows;
                edit = s -> replace(s, "# kbref 1\n" => ""))
            bad("w_comp.dat", "weights", ["case" => "x"],
                [KbrefRow("mat", 1, 0, -1, 0, 0.0im)])
            bad("w_rcorr_k1.dat", "weights", ["case" => "x"],
                [KbrefRow("rcorr", 1, 0, 0, 0, 0.0im)])
            bad("w_rcorr_m0.dat", "weights", ["case" => "x"],
                [KbrefRow("rcorr", 3, 0, 0, 0, 0.0im)])
            bad("w_k6.dat", "weights", ["case" => "x"],
                [KbrefRow("bd", 6, 0, -1, -1, 1.0 + 0im)])
            bad("w_imag.dat", "weights", ["case" => "x"],
                [KbrefRow("bd", 1, 0, -1, -1, 1.0 + 1.0im)])
            #
            # P2 herm_matrix files must carry a non-negative case_id
            p2meta = [hmeta; "generator" => "p2_selftest"; "case_set" => "x"]
            p = joinpath(dir, "ok_herm_p2.dat")
            write_kbref(p, "herm_matrix", [p2meta; "case_id" => "3"], hrows)
            @test read_kbref(p).metadata["case_id"] == "3"
            bad("herm_p2_nocase.dat", "herm_matrix", p2meta, hrows)
            bad("herm_p2_negcase.dat", "herm_matrix", [p2meta; "case_id" => "-1"], hrows)
        end

        @testset "P2: scalar, function, element, labels" begin
            base = ["generator" => "p2_selftest", "case_set" => "selftest"]
            #
            # Round trips
            srows = [
                KbrefRow("distance_ret", 0, -1, -1, -1, 1.5 + 0.0im),
                KbrefRow("distance_tv", 0, -1, -1, -1, 0.0 + 0.0im),
                KbrefRow("distance_les", 0, -1, -1, -1, 2.5e-17 + 0.0im),
                KbrefRow("distance_total", 0, -1, -1, -1, 4.0 + 0.0im),
                KbrefRow("response", 1, -1, -1, -1, -0.3 + 0.7im),
            ]
            fmeta = [base; "nt" => "3"; "size1" => "2"; "size2" => "2"]
            frows = [KbrefRow("extrapolate", 2, j, a, b, complex(j, a + b))
                     for j = -1:3 for a = 0:1 for b = 0:1]
            imeta = [fmeta; "tinter" => "1.5"]
            irows = [KbrefRow("interpolate", 0, -1, a, b, complex(a, b))
                     for a = 0:1 for b = 0:1]
            erows = [[KbrefRow(c, 4, t, a, b, complex(t, 0.5))
                      for c in ("les_timediag", "density_matrix")
                      for t = -1:3 for a = 0:1 for b = 0:1];
                     [KbrefRow(c, 5, j, a, b, complex(j, -0.5))
                      for c in ("startup_less_adv", "startup_lmix_rmix")
                      for j = 0:3 for a = 0:1 for b = 0:1]]
            lrows = [
                KbrefRow("order", 0, -1, -1, -1, 0.0 + 0.0im),
                KbrefRow("order", 1, -1, -1, -1, 5.0 + 0.0im),
                KbrefRow("alpha", 1, -1, -1, -1, 0.5 - 0.25im),
            ]
            for (name, kind, meta, rows) in (
                    ("ok_scalar.dat", "scalar", base, srows),
                    ("ok_function.dat", "function", fmeta, frows),
                    ("ok_interp.dat", "function", imeta, irows),
                    ("ok_element.dat", "element", fmeta, erows),
                    ("ok_labels.dat", "labels", base, lrows))
                p = joinpath(dir, name)
                write_kbref(p, kind, meta, rows)
                d = read_kbref(p)
                @test d.metadata["data_kind"] == kind
                @test d.metadata["generator"] == "p2_selftest"
                @test length(d.rows) == length(rows)
                @test all(d.rows[n].value == rows[n].value for n in eachindex(rows))
                @test all(d.rows[n].component == rows[n].component for n in eachindex(rows))
                @test all((d.rows[n].i, d.rows[n].j, d.rows[n].a, d.rows[n].b) ==
                          (rows[n].i, rows[n].j, rows[n].a, rows[n].b) for n in eachindex(rows))
            end
            #
            # Labels joined with cases
            lab = read_kbref(joinpath(dir, "ok_labels.dat"))
            @test_throws ArgumentError labels_by_case(lab, [0, 1]; order = true, alpha = false)
            lab2 = KbrefData(lab.path, lab.metadata, lab.rows[1:2])
            m = labels_by_case(lab2, [0, 1]; order = true, alpha = false)
            @test m[0].order == 0 && m[1].order == 5
            @test m[0].alpha === nothing
            m = labels_by_case(KbrefData(lab.path, lab.metadata, lab.rows[2:3]), [1];
                               order = true, alpha = true)
            @test m[1].alpha == 0.5 - 0.25im
            # Missing order, missing alpha, unknown case, forbidden order
            @test_throws ArgumentError labels_by_case(lab2, [0, 1, 2]; order = true, alpha = false)
            @test_throws ArgumentError labels_by_case(lab2, [0, 1]; order = true, alpha = true)
            @test_throws ArgumentError labels_by_case(lab2, [0]; order = true, alpha = false)
            @test_throws ArgumentError labels_by_case(lab2, [0, 1]; order = false, alpha = false)
            #
            # Bad inputs
            bad("s_comp.dat", "scalar", base,
                [KbrefRow("distance", 0, -1, -1, -1, 1.0 + 0im)])
            bad("s_imag.dat", "scalar", base,
                [KbrefRow("distance_ret", 0, -1, -1, -1, 1.0 + 1.0im)])
            bad("s_neg.dat", "scalar", base,
                [KbrefRow("response", -1, -1, -1, -1, 1.0 + 0im)])
            bad("s_j.dat", "scalar", base,
                [KbrefRow("response", 0, 0, -1, -1, 1.0 + 0im)])
            bad("s_dup.dat", "scalar", base, [srows; srows[end]])
            bad("s_nan.dat", "scalar", base,
                [KbrefRow("response", 0, -1, -1, -1, complex(NaN, 0.0))])
            bad("s_inf.dat", "scalar", base,
                [KbrefRow("response", 0, -1, -1, -1, complex(0.0, -Inf))])
            bad("s_ncol.dat", "scalar", base, srows;
                edit = s -> replace(s, r"\n(response 1 -1 -1 -1 \S+ \S+)\n" => s"\n\1 0\n"))
            bad("s_nogen.dat", "scalar", ["case_set" => "x"], srows)
            bad("s_nocaseset.dat", "scalar", ["generator" => "p2_x"], srows)
            bad("f_orb.dat", "function", fmeta,
                [KbrefRow("extrapolate", 0, 0, 2, 0, 0.0im)])
            bad("f_t.dat", "function", fmeta,
                [KbrefRow("extrapolate", 0, 4, 0, 0, 0.0im)])
            bad("f_t2.dat", "function", fmeta,
                [KbrefRow("extrapolate", 0, -2, 0, 0, 0.0im)])
            bad("f_nont.dat", "function", base,
                [KbrefRow("extrapolate", 0, 0, 0, 0, 0.0im)])
            bad("f_notinter.dat", "function", fmeta, irows)
            bad("f_interp_j.dat", "function", imeta,
                [KbrefRow("interpolate", 0, 0, 0, 0, 0.0im)])
            bad("f_comp.dat", "function", fmeta,
                [KbrefRow("ret", 0, 0, 0, 0, 0.0im)])
            bad("e_comp.dat", "element", fmeta,
                [KbrefRow("density", 0, 0, 0, 0, 0.0im)])
            bad("e_orb.dat", "element", fmeta,
                [KbrefRow("les_timediag", 0, 0, 0, 2, 0.0im)])
            bad("e_t.dat", "element", fmeta,
                [KbrefRow("les_timediag", 0, 4, 0, 0, 0.0im)])
            bad("e_dup.dat", "element", fmeta, [erows; erows[1]])
            bad("e_startup_neg.dat", "element", fmeta,
                [KbrefRow("startup_less_adv", 0, -1, 0, 0, 0.0im)])
            bad("e_startup_big.dat", "element", fmeta,
                [KbrefRow("startup_lmix_rmix", 0, 4, 0, 0, 0.0im)])
            bad("l_order_frac.dat", "labels", base,
                [KbrefRow("order", 0, -1, -1, -1, 2.5 + 0im)])
            bad("l_order_big.dat", "labels", base,
                [KbrefRow("order", 0, -1, -1, -1, 6.0 + 0im)])
            bad("l_order_neg.dat", "labels", base,
                [KbrefRow("order", 0, -1, -1, -1, -1.0 + 0im)])
            bad("l_order_imag.dat", "labels", base,
                [KbrefRow("order", 0, -1, -1, -1, 1.0 + 1.0im)])
            bad("l_dup.dat", "labels", base, [lrows; lrows[1]])
            bad("l_comp.dat", "labels", base,
                [KbrefRow("beta", 0, -1, -1, -1, 1.0 + 0im)])
            bad("l_idx.dat", "labels", base,
                [KbrefRow("order", 0, 0, -1, -1, 1.0 + 0im)])
        end

        @testset "P4: iteration and single-case files" begin
            # FIXPOINT trajectory: g for iter = 0:niter
            fmeta = ["generator" => "phase4_vie2", "case_set" => "selftest",
                     "case" => "fixpoint_selftest", "sig" => "-1",
                     "algorithm" => "fixpoint", "niter" => "1", "ntau" => "1",
                     "size1" => "1", "size2" => "1"]
            frows = [KbrefRow("g", it, r, 0, 0, complex(0.1 * it, -0.2 * r))
                     for it = 0:1 for r = 0:1]
            # CG trajectory: g and five scalars for iter = 1:niter
            cmeta = ["generator" => "phase4_vie2", "case_set" => "selftest",
                     "case" => "cg_selftest", "sig" => "1",
                     "algorithm" => "cg", "niter" => "2", "ntau" => "1",
                     "size1" => "1", "size2" => "1"]
            crows = KbrefRow[]
            for it = 1:2
                for r = 0:1
                    push!(crows, KbrefRow("g", it, r, 0, 0, complex(it / 3, r / 7)))
                end
                for (n, c) in enumerate(KBREF_CG_SCALARS)
                    push!(crows, KbrefRow(c, it, -1, -1, -1, complex(n * it / 11, 0.0)))
                end
            end
            #
            # Byte-identical write/read/write round trips
            for (name, meta, rows) in (("ok_fix.dat", fmeta, frows),
                                       ("ok_cg.dat", cmeta, crows))
                p = joinpath(dir, name)
                q = joinpath(dir, "re_" * name)
                write_kbref(p, "iteration", meta, rows)
                d = read_kbref(p)
                @test d.metadata["data_kind"] == "iteration"
                write_kbref(q, "iteration", meta, d.rows)
                @test read(p) == read(q)
            end
            #
            # Reused herm_matrix and element/density_matrix, one case per file
            hmeta = ["generator" => "phase4_bubble", "case" => "bubble1_ff_size1",
                     "sig" => "-1", "nt" => "0", "ntau" => "0",
                     "size1" => "1", "size2" => "1"]
            hrows = [KbrefRow("mat", 0, -1, 0, 0, 0.5 + 0im),
                     KbrefRow("ret", 0, 0, 0, 0, -1im),
                     KbrefRow("tv", 0, 0, 0, 0, 0.2im),
                     KbrefRow("les", 0, 0, 0, 0, 0.3im)]
            p = joinpath(dir, "ok_p4_herm.dat")
            write_kbref(p, "herm_matrix", hmeta, hrows)
            @test read_kbref(p).metadata["case"] == "bubble1_ff_size1"
            emeta = ["generator" => "phase4_observables", "case_set" => "selftest",
                     "case" => "density_equilibrium", "sig" => "1", "nt" => "2",
                     "size1" => "1", "size2" => "1"]
            erows = [KbrefRow("density_matrix", 0, -1, 0, 0, 0.4 + 0im)]
            p = joinpath(dir, "ok_p4_elem.dat")
            write_kbref(p, "element", emeta, erows)
            @test read_kbref(p).rows[1].component == "density_matrix"
            #
            # Bad inputs
            without(meta, key) = filter(x -> x.first != key, meta)
            bad("i_kind.dat", "iteration", fmeta, frows;
                edit = s -> replace(s, "data_kind iteration" => "data_kind trajectory"))
            bad("i_comp.dat", "iteration", fmeta,
                [frows; KbrefRow("residual", 1, -1, -1, -1, 0im)])
            bad("i_casecomp.dat", "herm_matrix", hmeta,
                [hrows[1:3]; KbrefRow("les_bubble1_ff", 0, 0, 0, 0, 0.3im)])
            bad("i_densitycomp.dat", "element", emeta,
                [KbrefRow("density_state", 0, -1, 0, 0, 0.4 + 0im)])
            bad("i_twocases.dat", "herm_matrix", [hmeta; "case" => "bubble2_ff_size1"], hrows)
            bad("i_nocase.dat", "herm_matrix", without(hmeta, "case"), hrows)
            bad("i_nosig.dat", "iteration", without(cmeta, "sig"), crows)
            bad("i_badsig.dat", "iteration", [without(cmeta, "sig"); "sig" => "0"], crows)
            bad("i_noalg.dat", "iteration", without(cmeta, "algorithm"), crows)
            bad("i_ncol.dat", "iteration", fmeta, frows;
                edit = s -> replace(s, r"\n(g 1 1 0 0 \S+) \S+\n" => s"\n\1\n"))
            bad("i_dup.dat", "iteration", fmeta, [frows; frows[1]])
            bad("i_nan.dat", "iteration", fmeta,
                [frows[1:end-1]; KbrefRow("g", 1, 1, 0, 0, complex(NaN, 0.0))])
            bad("i_inf.dat", "iteration", cmeta,
                [crows[1:end-1]; KbrefRow("beta_cg", 2, -1, -1, -1, complex(Inf, 0.0))])
            bad("i_negiter.dat", "iteration", fmeta,
                [frows; KbrefRow("g", -1, 0, 0, 0, 0im)])
            bad("i_bigiter.dat", "iteration", fmeta,
                [frows; KbrefRow("g", 2, 0, 0, 0, 0im)])
            bad("i_cgiter0.dat", "iteration", cmeta,
                [crows; KbrefRow("g", 0, 0, 0, 0, 0im)])
            bad("i_gtau.dat", "iteration", fmeta,
                [frows; KbrefRow("g", 1, 2, 0, 0, 0im)])
            bad("i_gorb.dat", "iteration", fmeta,
                [frows; KbrefRow("g", 1, 0, 1, 0, 0im)])
            bad("i_scalar_idx.dat", "iteration", cmeta,
                [crows[1:end-1]; KbrefRow("beta_cg", 2, 0, -1, -1, 0.1 + 0im)])
            bad("i_scalar_imag.dat", "iteration", cmeta,
                [crows[1:end-1]; KbrefRow("beta_cg", 2, -1, -1, -1, 0.1 + 0.1im)])
            bad("i_scalar_fix.dat", "iteration", fmeta,
                [frows; KbrefRow("alpha", 1, -1, -1, -1, 0.1 + 0im)])
            bad("i_missing_scalar.dat", "iteration", cmeta,
                filter(r -> !(r.component == "c2" && r.i == 2), crows))
            bad("i_dup_scalar.dat", "iteration", cmeta,
                [crows; KbrefRow("rsnew", 1, -1, -1, -1, 0.5 + 0im)])
            bad("i_missing_g.dat", "iteration", fmeta, frows[1:end-1])
        end

        @testset "P6: pseudo" begin
            pmeta = ["generator" => "pseudo_reference", "case" => "selftest",
                     "sig" => "-1", "nt" => "1", "ntau" => "1", "size1" => "2",
                     "size2" => "2", "k" => "2"]
            prows = KbrefRow[]
            v(n) = complex(n / 7, -n / 11)
            n = 0
            for q = 0:1, a = 0:1, b = 0:1
                push!(prows, KbrefRow("mat", q, -1, a, b, v(n += 1)))
            end
            for i = 0:1, j = 0:i, a = 0:1, b = 0:1
                push!(prows, KbrefRow("gtr", i, j, a, b, v(n += 1)))
            end
            for i = 0:1, q = 0:1, a = 0:1, b = 0:1
                push!(prows, KbrefRow("tv", i, q, a, b, v(n += 1)))
            end
            for i = 0:1, j = i:1, a = 0:1, b = 0:1
                push!(prows, KbrefRow("les", i, j, a, b, v(n += 1)))
            end
            for i = -1:1, a = 0:1, b = 0:1
                push!(prows, KbrefRow("density", i, -1, a, b, v(n += 1)))
            end
            for i = 1:2, j = 0:i-1, a = 0:1, b = 0:1
                push!(prows, KbrefRow("les_start_source", i, j, a, b, v(n += 1)))
            end
            #
            # Round trip: metadata, rows, and the rebuilt PseudoGF
            p = joinpath(dir, "ok_pseudo.dat")
            q = joinpath(dir, "re_ok_pseudo.dat")
            write_kbref(p, "pseudo", pmeta, prows)
            d = read_kbref(p)
            @test d.metadata["data_kind"] == "pseudo"
            @test all(d.metadata[k] == v for (k, v) in pmeta)
            @test [(r.component, r.i, r.j, r.a, r.b, r.value) for r in d.rows] ==
                  [(r.component, r.i, r.j, r.a, r.b, r.value) for r in prows]
            G = kbref_to_pseudo(d)
            @test G isa PseudoGF
            @test all(r -> kb_value(G, r) == r.value,
                      filter(r -> r.component in ("mat", "gtr", "tv", "les"), d.rows))
            write_kbref(q, "pseudo", pmeta, d.rows)
            @test read(p) == read(q)
            # Storage rows are optional, but complete if present
            srows = filter(r -> r.component in ("density", "les_start_source"), prows)
            p = joinpath(dir, "ok_pseudo_src.dat")
            write_kbref(p, "pseudo", pmeta, srows)
            @test length(read_kbref(p).rows) == length(srows)
            #
            # Bad inputs
            without(meta, key) = filter(x -> x.first != key, meta)
            bad("p_ncol.dat", "pseudo", pmeta, prows;
                edit = s -> replace(s, r"\n(mat 0 -1 0 0 \S+ \S+)\n" => s"\n\1 0.0\n"))
            bad("p_comp.dat", "pseudo", pmeta, [prows; KbrefRow("ret", 0, 0, 0, 0, 0im)])
            bad("p_gtr_tri.dat", "pseudo", pmeta, [prows; KbrefRow("gtr", 0, 1, 0, 0, 0im)])
            bad("p_les_tri.dat", "pseudo", pmeta, [prows; KbrefRow("les", 1, 0, 0, 0, 0im)])
            bad("p_dens_j.dat", "pseudo", pmeta, [prows; KbrefRow("density", 0, 0, 0, 0, 0im)])
            bad("p_src_low.dat", "pseudo", pmeta,
                [prows; KbrefRow("les_start_source", 1, 1, 0, 0, 0im)])
            bad("p_src_high.dat", "pseudo", pmeta,
                [prows; KbrefRow("les_start_source", 3, 0, 0, 0, 0im)])
            bad("p_orb.dat", "pseudo", pmeta, [prows; KbrefRow("density", 0, -1, 2, 0, 0im)])
            bad("p_dup.dat", "pseudo", pmeta, [prows; prows[1]])
            bad("p_nan.dat", "pseudo", pmeta,
                [prows[1:end-1]; KbrefRow("les_start_source", 2, 1, 1, 1, complex(NaN, 0.0))])
            bad("p_incomplete.dat", "pseudo", pmeta, filter(r -> r !== prows[3], prows))
            bad("p_nok.dat", "pseudo", without(pmeta, "k"), prows)
            bad("p_nocase.dat", "pseudo", without(pmeta, "case"), prows)
            bad("p_badsig.dat", "pseudo", [without(pmeta, "sig"); "sig" => "0"], prows)
        end

        # Round trip of a P7 file: metadata and rows are reproduced exactly
        function roundtrip7(name, kind, meta, rows)
            p = joinpath(dir, name)
            q = joinpath(dir, "re_" * name)
            write_kbref(p, kind, meta, rows)
            d = read_kbref(p)
            @test d.metadata["data_kind"] == kind
            @test all(d.metadata[k] == v for (k, v) in meta)
            @test [(r.component, r.i, r.j, r.a, r.b, r.value) for r in d.rows] ==
                  [(r.component, r.i, r.j, r.a, r.b, r.value) for r in rows]
            write_kbref(q, kind, meta, d.rows)
            @test read(p) == read(q)
            return d
        end
        without7(meta, key) = filter(x -> x.first != key, meta)
        ncol7(s) = replace(s, r"\n(\S+ \S+ \S+ \S+ \S+ \S+ \S+)\n" => s"\n\1 0.0\n"; count = 1)

        @testset "phase7_fourier schema" begin
            fmeta = ["generator" => "phase7_fourier_ref", "case_set" => "selftest",
                     "size_max" => "2"]
            v(n) = complex(n / 3, -n / 5)
            frows = KbrefRow[]
            for c in ("fft_input", "fft_forward", "fft_backward"), j = 0:2
                push!(frows, KbrefRow(c, 3, j, 0, 0, v(j)))
            end
            push!(frows, KbrefRow("dftcorr", 0, 1, 0, 0, 0.7 + 0im))
            push!(frows, KbrefRow("dftcorr", 0, 1, 1, 0, 0.9 + 0im))
            push!(frows, KbrefRow("dftcorr", 0, 1, 3, 0, v(1)))
            push!(frows, KbrefRow("dftcorr", 0, 3, 5, 0, v(2)))
            push!(frows, KbrefRow("adaptive", 2, 4, 0, 0, -1.5 + 0im))
            push!(frows, KbrefRow("adaptive", 2, 4, 2, 0, v(3)))
            push!(frows, KbrefRow("matsubara_dft_result", 1, 5, 1, 0, v(4)))
            push!(frows, KbrefRow("matsubara_ft_result", 13, -25, 0, 1, v(5)))
            push!(frows, KbrefRow("vie2_pcf20", 0, 7, 1, 1, v(6)))
            push!(frows, KbrefRow("vie2_pcf10", 0, 7, 1, 1, v(7)))
            d = roundtrip7("ok_p7f.dat", "phase7_fourier", fmeta, frows)
            @test Set(r.component for r in d.rows) == Set(KBREF_COMPONENTS["phase7_fourier"])
            #
            # Bad inputs
            bad("p7f_kind.dat", "phase7_fft", fmeta, frows)
            bad("p7f_comp.dat", "phase7_fourier", fmeta, [frows; KbrefRow("fft", 3, 0, 0, 0, 0im)])
            bad("p7f_ncol.dat", "phase7_fourier", fmeta, frows; edit = ncol7)
            bad("p7f_nan.dat", "phase7_fourier", fmeta,
                [frows; KbrefRow("vie2_pcf10", 0, 8, 0, 0, complex(Inf, 0.0))])
            bad("p7f_fftj.dat", "phase7_fourier", fmeta, [frows; KbrefRow("fft_input", 3, 3, 0, 0, 0im)])
            bad("p7f_ffta.dat", "phase7_fourier", fmeta, [frows; KbrefRow("fft_forward", 2, 0, 1, 0, 0im)])
            bad("p7f_fftinc.dat", "phase7_fourier", fmeta, frows[2:end])
            bad("p7f_order.dat", "phase7_fourier", fmeta, [frows; KbrefRow("dftcorr", 1, 2, 0, 0, 0im)])
            bad("p7f_alpha.dat", "phase7_fourier", fmeta, [frows; KbrefRow("dftcorr", 1, 1, 4, 0, 0im)])
            bad("p7f_theta.dat", "phase7_fourier", fmeta, [frows; KbrefRow("dftcorr", 1, 3, 0, 0, 1im)])
            bad("p7f_fid.dat", "phase7_fourier", fmeta, [frows; KbrefRow("adaptive", 3, 0, 1, 0, 0im)])
            bad("p7f_ftord.dat", "phase7_fourier", fmeta,
                [frows; KbrefRow("matsubara_ft_result", 12, 0, 0, 0, 0im)])
            bad("p7f_orb.dat", "phase7_fourier", fmeta, [frows; KbrefRow("vie2_pcf20", 0, 0, 2, 0, 0im)])
            bad("p7f_dup.dat", "phase7_fourier", fmeta, [frows; frows[end]])
            bad("p7f_nosize.dat", "phase7_fourier", without7(fmeta, "size_max"), frows)
            bad("p7f_noset.dat", "phase7_fourier", without7(fmeta, "case_set"), frows)
        end

        @testset "phase7_ness schema" begin
            nmeta = ["generator" => "phase7_ness_ref", "case_set" => "selftest",
                     "nft_max" => "4", "size_max" => "2"]
            v(n) = complex(-n / 7, n / 9)
            nrows = KbrefRow[]
            n = 0
            for c in ("ret_time", "les_time", "ret_freq", "les_freq")
                push!(nrows, KbrefRow(c, 1, 3, 1, 0, v(n += 1)))
            end
            for c in ("bath_ret", "bath_les")
                push!(nrows, KbrefRow(c, 3, 12, 0, 0, v(n += 1)))
            end
            for c in ("density", "convolution_density")
                push!(nrows, KbrefRow(c, 0, 0, 1, 1, v(n += 1)))
            end
            for c in ("bubble1", "bubble2", "resample", "force_equilibrium", "cntr_conversion")
                push!(nrows, KbrefRow(c, 0, 1, 0, 1, v(n += 1)))
                push!(nrows, KbrefRow(c, 0, 7, 0, 1, v(n += 1)))
            end
            d = roundtrip7("ok_p7n.dat", "phase7_ness", nmeta, nrows)
            @test Set(r.component for r in d.rows) == Set(KBREF_COMPONENTS["phase7_ness"])
            #
            # Bad inputs
            bad("p7n_comp.dat", "phase7_ness", nmeta, [nrows; KbrefRow("gtr_time", 0, 0, 0, 0, 0im)])
            bad("p7n_ncol.dat", "phase7_ness", nmeta, nrows; edit = ncol7)
            bad("p7n_nan.dat", "phase7_ness", nmeta, [nrows; KbrefRow("ret_time", 0, 0, 0, 0, complex(0.0, NaN))])
            bad("p7n_mode.dat", "phase7_ness", nmeta, [nrows; KbrefRow("ret_freq", 0, 4, 0, 0, 0im)])
            bad("p7n_mode2.dat", "phase7_ness", nmeta, [nrows; KbrefRow("bubble1", 0, 8, 0, 0, 0im)])
            bad("p7n_orb.dat", "phase7_ness", nmeta, [nrows; KbrefRow("les_time", 0, 0, 0, 2, 0im)])
            bad("p7n_bath.dat", "phase7_ness", nmeta, [nrows; KbrefRow("bath_ret", 4, 0, 0, 0, 0im)])
            bad("p7n_batho.dat", "phase7_ness", nmeta, [nrows; KbrefRow("bath_les", 1, 0, 1, 0, 0im)])
            bad("p7n_dens.dat", "phase7_ness", nmeta, [nrows; KbrefRow("density", 0, 1, 0, 0, 0im)])
            bad("p7n_dup.dat", "phase7_ness", nmeta, [nrows; nrows[1]])
            bad("p7n_nonft.dat", "phase7_ness", without7(nmeta, "nft_max"), nrows)
        end

        @testset "phase7_example schema" begin
            emeta(c) = ["generator" => "phase7_examples_$(c)_ref", "case_set" => "selftest",
                        "case" => c]
            siam = [KbrefRow("gf_ret_time", 0, 5, 0, 0, 0.1 - 0.2im),
                    KbrefRow("gf_les_time", 0, 5, 0, 0, 0.3im),
                    KbrefRow("gf_ret_freq", 0, 9, 0, 0, -0.4im),
                    KbrefRow("gf_les_freq", 0, 9, 0, 0, 0.5im),
                    KbrefRow("density", 0, 0, 0, 0, 0.9 + 0im),
                    KbrefRow("current", 0, 0, 0, 0, 0.3 + 0im),
                    KbrefRow("current", 0, 1, 0, 0, 0.3 + 0im),
                    KbrefRow("eint", 0, 0, 0, 0, -0.1 + 0im)]
            bethe = [KbrefRow("gf_ret_time", 1, 0, 0, 0, -1im),
                     KbrefRow("gf_les_freq", 1, 2, 0, 0, 0.1im),
                     KbrefRow("density", 1, 0, 0, 0, 1.0 + 0im),
                     KbrefRow("eint", 1, 0, 0, 0, -0.2 + 0im),
                     KbrefRow("iteration_error", 1, 1, 0, 0, 2.0 + 0im),
                     KbrefRow("iteration_error", 1, 2, 0, 0, 1.0 + 0im)]
            prop = [KbrefRow("gf_ret_time", 2, 3, 0, 0, 0.1 - 0.9im),
                    KbrefRow("gf_les_time", 2, 3, 0, 0, 0.5im),
                    KbrefRow("density", 2, 0, 0, 0, 1.0 + 0im),
                    KbrefRow("density", 2, 9, 0, 0, 1.0 + 0im),
                    KbrefRow("iteration_error", 2, 0, 0, 0, 0.5 + 0im),
                    KbrefRow("iteration_error", 2, 4, 1, 4, 1e-9 + 0im),
                    KbrefRow("iteration_error", 2, 2, 2, 9, 1e-14 + 0im)]
            roundtrip7("ok_p7e_siam.dat", "phase7_example", emeta("siam"), siam)
            roundtrip7("ok_p7e_bethe.dat", "phase7_example", emeta("bethe"), bethe)
            roundtrip7("ok_p7e_prop.dat", "phase7_example", emeta("bethe_prop"), prop)
            @test Set(r.component for r in [siam; bethe; prop]) ==
                  Set(KBREF_COMPONENTS["phase7_example"])
            #
            # Bad inputs
            bad("p7e_case.dat", "phase7_example", emeta("hubbard"), siam)
            bad("p7e_nocase.dat", "phase7_example", without7(emeta("siam"), "case"), siam)
            bad("p7e_caseid.dat", "phase7_example", emeta("bethe"), siam)
            bad("p7e_comp.dat", "phase7_example", emeta("siam"), [siam; KbrefRow("gf_tv", 0, 0, 0, 0, 0im)])
            bad("p7e_ncol.dat", "phase7_example", emeta("siam"), siam; edit = ncol7)
            bad("p7e_nan.dat", "phase7_example", emeta("siam"),
                [siam; KbrefRow("gf_ret_time", 0, 6, 0, 0, complex(NaN, 0.0))])
            bad("p7e_dup.dat", "phase7_example", emeta("bethe"), [bethe; bethe[end]])
            # Quantities that the example does not compute are rejected
            bad("p7e_cur.dat", "phase7_example", emeta("bethe"), [bethe; KbrefRow("current", 1, 0, 0, 0, 0im)])
            bad("p7e_eint.dat", "phase7_example", emeta("bethe_prop"), [prop; KbrefRow("eint", 2, 0, 0, 0, 0im)])
            bad("p7e_freq.dat", "phase7_example", emeta("bethe_prop"), [prop; KbrefRow("gf_ret_freq", 2, 0, 0, 0, 0im)])
            bad("p7e_siamit.dat", "phase7_example", emeta("siam"),
                [siam; KbrefRow("iteration_error", 0, 1, 0, 0, 1.0 + 0im)])
            # Illegal iterations
            bad("p7e_it0.dat", "phase7_example", emeta("bethe"),
                [bethe; KbrefRow("iteration_error", 1, 0, 0, 0, 1.0 + 0im)])
            bad("p7e_stage.dat", "phase7_example", emeta("bethe_prop"),
                [prop; KbrefRow("iteration_error", 2, 0, 3, 0, 1.0 + 0im)])
            bad("p7e_itimag.dat", "phase7_example", emeta("bethe"),
                [bethe; KbrefRow("iteration_error", 1, 3, 0, 0, 1.0 + 1im)])
            bad("p7e_cur2.dat", "phase7_example", emeta("siam"), [siam; KbrefRow("current", 0, 2, 0, 0, 0im)])
            bad("p7e_dens.dat", "phase7_example", emeta("bethe"), [bethe; KbrefRow("density", 1, 1, 0, 0, 0im)])
        end

        @testset "P8: moving schema" begin
            mmeta = ["generator" => "p8_moving_ref", "case_set" => "moving",
                     "case_id" => "0", "source" => "selftest", "tc" => "1",
                     "size" => "2", "sign" => "-1", "tstp" => "3"]
            mrows = KbrefRow[]
            for c in ("ret", "less"), i = 0:1, j = 0:1, b = 0:1, a = 0:1
                push!(mrows, KbrefRow(c, i, j, a, b, complex(i + 0.5j, a - 0.25b)))
            end
            d = roundtrip7("ok_p8m.dat", "moving", mmeta, mrows)
            @test length(d.rows) == 32
            # Bad inputs
            bad("p8m_kind.dat", "movin", mmeta, mrows)
            bad("p8m_comp.dat", "moving", mmeta, [mrows[2:end]; KbrefRow("gtr", 0, 0, 0, 0, 0im)])
            bad("p8m_win.dat", "moving", mmeta, [mrows[2:end]; KbrefRow("ret", 2, 0, 0, 0, 0im)])
            bad("p8m_neg.dat", "moving", mmeta, [mrows[2:end]; KbrefRow("ret", 0, -1, 0, 0, 0im)])
            bad("p8m_orb.dat", "moving", mmeta, [mrows[2:end]; KbrefRow("less", 0, 0, 2, 0, 0im)])
            bad("p8m_dup.dat", "moving", mmeta, [mrows; mrows[1]])
            bad("p8m_inc.dat", "moving", mmeta, mrows[2:end])
            bad("p8m_nan.dat", "moving", mmeta, [mrows[2:end]; KbrefRow("ret", 0, 0, 0, 0, complex(NaN, 0.0))])
            bad("p8m_ncol.dat", "moving", mmeta, mrows; edit = ncol7)
            for key in ("tc", "size", "sign", "tstp", "source", "case_id")
                bad("p8m_no$key.dat", "moving", without7(mmeta, key), mrows)
            end
            bad("p8m_gen.dat", "moving", ["generator" => "p8_gw_ref"; mmeta[2:end]], mrows)
            bad("p8m_sign.dat", "moving", [mmeta[1:6]; "sign" => "2"; mmeta[8:end]], mrows)
            # The window size contradicts the data
            bad("p8m_tc.dat", "moving", [mmeta[1:4]; "tc" => "2"; mmeta[6:end]], mrows)
        end

        @testset "P8: series schema ($cs)" for cs in sort(collect(keys(KBREF_SERIES)))
            cls = KBREF_SERIES[cs]
            comps = sort(collect(keys(cls)))
            smeta = ["generator" => KBREF_SERIES_GENERATOR[cs], "case_set" => cs,
                     "components" => join(comps, ","), "nsteps" => "3", "nk" => "2",
                     "size" => "2", "source" => "selftest"]
            srows = KbrefRow[]
            for (n, c) in enumerate(comps)
                if cls[c] == 0
                    push!(srows, KbrefRow(c, 2, -1, -1, -1, complex(n, 0.5)))
                else
                    push!(srows, KbrefRow(c, 0, 1, 1, 0, complex(-n, 1.5)))
                end
            end
            d = roundtrip7("ok_p8s_$cs.dat", "series", smeta, srows)
            @test Set(r.component for r in d.rows) == Set(comps)
            # Matrix-block and scalar components
            cm = first(c for c in comps if cls[c] == 2)
            c0 = first(c for c in comps if cls[c] == 0)
            # Bad inputs
            bad("p8s_$(cs)_comp.dat", "series", smeta, [srows; KbrefRow("foo", 0, 0, 0, 0, 0im)])
            other = first(k for k in keys(KBREF_SERIES) if k != cs &&
                          any(c -> !haskey(cls, c), keys(KBREF_SERIES[k])))
            cx = first(c for c in keys(KBREF_SERIES[other]) if !haskey(cls, c))
            bad("p8s_$(cs)_foreign.dat", "series", smeta, [srows; KbrefRow(cx, 0, -1, -1, -1, 0im)])
            bad("p8s_$(cs)_step.dat", "series", smeta, [srows; KbrefRow(cm, 3, 0, 0, 0, 0im)])
            bad("p8s_$(cs)_k.dat", "series", smeta, [srows; KbrefRow(cm, 1, 2, 0, 0, 0im)])
            bad("p8s_$(cs)_orb.dat", "series", smeta, [srows; KbrefRow(cm, 1, 0, 2, 0, 0im)])
            bad("p8s_$(cs)_scal.dat", "series", smeta, [srows; KbrefRow(c0, 1, 0, -1, -1, 0im)])
            bad("p8s_$(cs)_dup.dat", "series", smeta, [srows; srows[1]])
            bad("p8s_$(cs)_nan.dat", "series", smeta, [srows; KbrefRow(c0, 0, -1, -1, -1, complex(Inf, 0.0))])
            bad("p8s_$(cs)_ncol.dat", "series", smeta, srows; edit = ncol7)
            # The component set is incomplete
            bad("p8s_$(cs)_inc.dat", "series", smeta, srows[2:end])
            for key in ("components", "nsteps", "nk", "size", "source")
                bad("p8s_$(cs)_no$key.dat", "series", without7(smeta, key), srows)
            end
            # Generator and case_set must match
            g = first(v for (k, v) in KBREF_SERIES_GENERATOR if k != cs)
            bad("p8s_$(cs)_gen.dat", "series", ["generator" => g; smeta[2:end]], srows)
            bad("p8s_$(cs)_cs.dat", "series", [smeta[1]; "case_set" => "bethe"; smeta[3:end]], srows)
            # The metadata lists a component that is not allowed
            bad("p8s_$(cs)_meta.dat", "series",
                [smeta[1:2]; "components" => join([comps; "foo"], ","); smeta[4:end]], srows)
        end

        @testset "P5: observable schema ($m)" for m in sort(collect(keys(KBREF_OBSERVABLE)))
            cls = KBREF_OBSERVABLE[m]
            comps = sort(collect(keys(cls)))
            size = m == "holstein_nambu" || m == "holstein_impurity" ? "2" : "1"
            ometa = ["generator" => "$(m)_text", "case_set" => "$(m)_quick", "model" => m,
                     "approximation" => KBREF_P5_APPROX[m][1], "profile" => "quick",
                     "nt" => "3", "ntau" => "10", "h" => "0.05", "beta" => "5",
                     "solve_order" => "3", "mats_max_iter" => "100",
                     "bootstrap_max_iter" => "50", "corrector_steps" => "3", "size" => size,
                     "params" => "U=1"]
            orows = KbrefRow[]
            for (n, c) in enumerate(comps), t = -1:3
                if cls[c] == 0
                    push!(orows, KbrefRow(c, t, -1, -1, -1, complex(n + 0.1t, 0.0)))
                else
                    for a = 0:parse(Int, size)-1, b = 0:parse(Int, size)-1
                        push!(orows, KbrefRow(c, t, -1, a, b, complex(n - a, 0.25b)))
                    end
                end
            end
            # write/read/write is byte identical
            d = roundtrip7("ok_p5o_$m.dat", "observable", ometa, orows)
            @test length(d.rows) == length(orows)
            c0 = first(c for c in comps if cls[c] == 0)
            # Bad inputs
            bad("p5o_$(m)_kind.dat", "observables", ometa, orows)
            bad("p5o_$(m)_comp.dat", "observable", ometa, [orows; KbrefRow("foo", 0, -1, -1, -1, 0im)])
            # A component of another model
            other = first(k for k in keys(KBREF_OBSERVABLE)
                          if any(c -> !haskey(cls, c), keys(KBREF_OBSERVABLE[k])))
            cx = first(c for c in keys(KBREF_OBSERVABLE[other]) if !haskey(cls, c))
            bad("p5o_$(m)_foreign.dat", "observable", ometa,
                [orows; KbrefRow(cx, 0, -1, -1, -1, 0im)])
            bad("p5o_$(m)_ncol.dat", "observable", ometa, orows; edit = ncol7)
            bad("p5o_$(m)_dup.dat", "observable", ometa, [orows; orows[1]])
            bad("p5o_$(m)_nan.dat", "observable", ometa,
                [orows; KbrefRow(c0, 3, -1, -1, -1, complex(NaN, 0.0))][2:end])
            bad("p5o_$(m)_inf.dat", "observable", ometa,
                [orows[2:end]; KbrefRow(orows[1].component, orows[1].i, -1, orows[1].a,
                                        orows[1].b, complex(Inf, 0.0))])
            bad("p5o_$(m)_tstp.dat", "observable", ometa, [orows; KbrefRow(c0, 4, -1, -1, -1, 0im)])
            bad("p5o_$(m)_tstp2.dat", "observable", ometa, [orows; KbrefRow(c0, -2, -1, -1, -1, 0im)])
            bad("p5o_$(m)_sent.dat", "observable", ometa,
                [orows[2:end]; KbrefRow(c0, -1, -1, 0, -1, 1.0 + 0im)])
            bad("p5o_$(m)_j.dat", "observable", ometa,
                [orows[2:end]; KbrefRow(c0, -1, 0, -1, -1, 1.0 + 0im)])
            bad("p5o_$(m)_im.dat", "observable", ometa,
                [orows[2:end]; KbrefRow(orows[1].component, orows[1].i, -1, -1, -1, 1.0 + 1im)])
            if haskey(cls, "rho")
                bad("p5o_$(m)_orb.dat", "observable", ometa,
                    [orows; KbrefRow("rho", 0, -1, parse(Int, size), 0, 0im)])
            end
            for key in KBREF_P5_META
                bad("p5o_$(m)_no$key.dat", "observable", without7(ometa, key), orows)
            end
            # model/approximation mismatch
            bad("p5o_$(m)_approx.dat", "observable",
                [ometa[1:3]; "approximation" => "RPA"; ometa[5:end]], orows)
            mo = first(k for k in keys(KBREF_P5_APPROX) if !(KBREF_P5_APPROX[m][1] in KBREF_P5_APPROX[k]))
            bad("p5o_$(m)_model.dat", "observable", [ometa[1:2]; "model" => mo; ometa[4:end]], orows)
            bad("p5o_$(m)_prof.dat", "observable", [ometa[1:4]; "profile" => "fast"; ometa[6:end]], orows)
        end

        @testset "P5: selected_green schema" begin
            gmeta = ["generator" => "hubbard_chain_text", "case_set" => "hubbard_2b_quick",
                     "model" => "hubbard", "approximation" => "2B", "profile" => "quick",
                     "nt" => "10", "ntau" => "40", "h" => "0.05", "beta" => "20",
                     "solve_order" => "3", "mats_max_iter" => "100",
                     "bootstrap_max_iter" => "50", "corrector_steps" => "3", "size" => "2",
                     "params" => "U=1"]
            grows = [KbrefRow(k..., complex(0.1n, -0.2n)) for (n, k) in
                     enumerate(KBREF_SELECTED_GREEN_KEYS)]
            d = roundtrip7("ok_p5g.dat", "selected_green", gmeta, grows)
            @test length(d.rows) == 4
            bad("p5g_key.dat", "selected_green", gmeta, [grows[1:3]; KbrefRow("tv", 3, 1, 0, 0, 0im)])
            bad("p5g_comp.dat", "selected_green", gmeta, [grows[1:3]; KbrefRow("gtr", 3, 2, 0, 0, 0im)])
            bad("p5g_inc.dat", "selected_green", gmeta, grows[1:3])
            bad("p5g_dup.dat", "selected_green", gmeta, [grows; grows[1]])
            bad("p5g_nan.dat", "selected_green", gmeta,
                [grows[1:3]; KbrefRow(KBREF_SELECTED_GREEN_KEYS[4]..., complex(0.0, NaN))])
            bad("p5g_ncol.dat", "selected_green", gmeta, grows; edit = ncol7)
            # Only defined for the Hubbard chain
            bad("p5g_model.dat", "selected_green",
                [gmeta[1:2]; "model" => "fkm"; "approximation" => "none"; gmeta[5:end]], grows)
            for key in KBREF_P5_META
                bad("p5g_no$key.dat", "selected_green", without7(gmeta, key), grows)
            end
        end

        @testset "P1 data files are still accepted" begin
            data = joinpath(@__DIR__, "data")
            files = filter(f -> endswith(f, ".dat") && !startswith(f, "p2_") &&
                                !startswith(f, "phase7_") && !startswith(f, "p8_"),
                           readdir(data))
            @test length(files) == 21
            for f in files
                d = read_kbref(joinpath(data, f))
                @test d.metadata["data_kind"] in ("weights", "herm_matrix")
            end
        end
    end
end
