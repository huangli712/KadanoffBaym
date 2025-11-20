haskey(ENV,"KADANOFF_BAYM_HOME") && pushfirst!(LOAD_PATH, ENV["KADANOFF_BAYM_HOME"])

using Test
using KadanoffBaym


#include("structs.jl")
#include("properties.jl")
#include("indexing.jl")
#include("inout.jl")
#include("weights.jl")
#include("traits.jl")
