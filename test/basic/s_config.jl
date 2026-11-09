#
# s_config.jl
#
# To test whether the configuration parser works correctly.
#

@testset verbose = true "KadanoffBaym: config.jl" begin
    @testset "Configuration file parser" begin
        # Locate the configuration file relative to this test file, so that
        # the test does not depend on the current working directory.
        fn = joinpath(@__DIR__, "neq.toml")
        wfn = joinpath(@__DIR__, "wrong.toml")
        @test !isfile(wfn)
        #
        cfg = inp_toml(fn, true)
        fil_dict(cfg)
        chk_dict()
        see_dict()
        #
        @test get_c("ntime") == 101
        @test get_c("ntau") == 21
        @test get_c("ndim1") == 2
        @test get_c("ndim2") == 3
        @test get_c("tmax") == 3.0
        @test get_c("beta") == 4.0
        #
        @test get_m("system") == "Hubbard"
        #
        rev_dict_c(_PCONTOUR)
        @test get_c("ntime") == 201
        @test get_c("ntau") == 1001
        @test get_c("ndim1") == 1
        @test get_c("ndim2") == 1
        @test get_c("tmax") == 5.0
        @test get_c("beta") == 4.0
        #
        rev_dict_m(_PMODEL)
        @test get_m("system") == "unknown"
        #
        contour = inp_toml(fn, "contour", true)
        rev_dict_c(contour)
        @test get_c("ntime") == 101
        @test get_c("ntau") == 21
        @test get_c("ndim1") == 2
        @test get_c("ndim2") == 3
        @test get_c("tmax") == 3.0
        @test get_c("beta") == 4.0
        #
        model = inp_toml(fn, "model", true)
        rev_dict_m(model)
        @test get_m("system") == "Hubbard"
        #
        @test_throws ErrorException inp_toml(fn, "xxx", true)
        @test_throws ErrorException inp_toml(wfn, "xxx", true)
        @test inp_toml(wfn, "xxx", false) === nothing
        @test_throws ErrorException inp_toml(wfn, true)
        @test inp_toml(wfn, false) === nothing
        #
        @test_throws ErrorException rev_dict_c(Dict("A" => 1, "B" => "1"))
        @test_throws ErrorException rev_dict_c(Dict("A" => [1,"B"]))
        @test_throws ErrorException rev_dict_c(Dict("ntime" => ["A",100]))
        @test_throws ErrorException rev_dict_m(Dict("A" => 1, "B" => "1"))
        @test_throws ErrorException rev_dict_m(Dict("A" => [1,"B"]))
        @test_throws ErrorException rev_dict_m(Dict("system" => [100,"A"]))
        #
        @test_throws ErrorException get_c("xxx")
        @test_throws ErrorException get_m("xxx")
        #
        # Restore the default parameters, so that the global dicts are
        # not polluted for the subsequent tests.
        rev_dict_c(_PCONTOUR)
        rev_dict_m(_PMODEL)
        @test get_c("ntime") == 201
        @test get_m("system") == "unknown"
    end
end

println("All tests pass!\n")
