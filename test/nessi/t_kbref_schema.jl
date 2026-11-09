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

        @testset "P1 data files are still accepted" begin
            data = joinpath(@__DIR__, "data")
            files = filter(f -> endswith(f, ".dat") && !startswith(f, "p2_"), readdir(data))
            @test length(files) == 21
            for f in files
                d = read_kbref(joinpath(data, f))
                @test d.metadata["data_kind"] in ("weights", "herm_matrix")
            end
        end
    end
end
