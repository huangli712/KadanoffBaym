#
# config.jl
#
# To test whether the configuration parser works correctly.
#

@testset verbose = true "KadanoffBaym: config.jl" begin
    @testset "Cf    Struct: getindex/setindex" begin
        push!(ARGS, "neq.toml")
        cfg = inp_toml(query_args(), true)
        fil_dict(cfg)
        chk_dict()
        #
        @test get_c("ntime") == 101
        @test get_c("ntau") == 21
        @test get_c("ndim1") == 2
        @test get_c("ndim2") == 3
        @test get_c("tmax") == 3.0
        @test get_c("beta") == 4.0
        #
        @test get_m("system") == "Hubbard"
    end
end