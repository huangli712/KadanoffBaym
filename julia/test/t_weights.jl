#
# t_weights.jl
#
# To compare all the integration weights (k = 1:5) with NESSi.
#
# NESSi reference: libcntr/cntr/integration.hpp:128-164,439-443 and the
# tables in libcntr/cntr/integration.cpp. Data: weights_k1_k5.dat.
#

@testset verbose = true "NESSi: integration weights" begin
    ref = read_kbref(joinpath(@__DIR__, "..", "..", "data", "test", "weights_k1_k5.dat"))
    #
    @testset "metadata" begin
        @test ref.metadata["data_kind"] == "weights"
        comps = Set(r.component for r in ref.rows)
        @test comps == Set(("greg", "bd", "pint", "pdiff", "rcorr"))
        # rcorr exists only for k = 2:5 and m = 1:k-1
        @test all(r -> r.i ≥ 2 && 1 ≤ r.j ≤ r.i - 1,
                  filter(r -> r.component == "rcorr", ref.rows))
        @test Set(r.i for r in ref.rows if r.component != "rcorr") == Set(1:5)
    end
    #
    @testset "point-wise comparison" begin
        errs = test_weight_kbref(ref; atol = 2e-12, rtol = 2e-12)
        for c in sort(collect(keys(errs)))
            @printf("  max |KB - NESSi| %-6s : %.3e\n", c, errs[c])
        end
        @test maximum(values(errs)) ≤ 2e-12
    end
    #
    @testset "k = 1 boundary convolution weights" begin
        @test size(Integrator(1).BCW.W) == (0, 2, 2)
    end
end
