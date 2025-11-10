haskey(ENV,"KADANOFF_BAYM_HOME") && pushfirst!(LOAD_PATH, ENV["KADANOFF_BAYM_HOME"])

using Test
using KadanoffBaym

@testset "Cn Struct" begin
    @test 1+1 == 2
    @test 2+2 == 4
end