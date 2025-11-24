#
# config.jl
#
# To test whether the configuration parser works correctly.
#

@testset verbose = true "KadanoffBaym: config.jl" begin
    @testset "Configuration file parser" begin
        push!(ARGS, "neq.toml")
        cfg = inp_toml(query_args(), true)
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
        contour = inp_toml(query_args(), "contour", true)
        rev_dict_c(contour)
        @test get_c("ntime") == 101
        @test get_c("ntau") == 21
        @test get_c("ndim1") == 2
        @test get_c("ndim2") == 3
        @test get_c("tmax") == 3.0
        @test get_c("beta") == 4.0
        #
        model = inp_toml(query_args(), "model", true)
        rev_dict_m(model)
        @test get_m("system") == "Hubbard"
        #
        try
            inp_toml("wrong.toml", "xxx", true)
        catch ex
            catch_error()
        end
        #
        try
            inp_toml("wrong.toml", "xxx", false)
        catch ex
            catch_error()
        end
        #
        try
            inp_toml("wrong.toml", true)
        catch ex
            catch_error()
        end
        #
        try
            inp_toml("wrong.toml", false)
        catch ex
            catch_error()
        end
        #
        try
            rev_dict_c(Dict("A" => [1,"B"]))
        catch ex
            catch_error()
        end
        #
        try
            rev_dict_m(Dict("A" => [1,"B"]))
        catch ex
            catch_error()
        end
        #
        try
            get_c("xxx")
        catch ex
            catch_error()
        end
        #
        try
            get_m("xxx")
        catch ex
            catch_error()
        end
    end
end
