#!/usr/bin/env julia

haskey(ENV,"KADANOFF_BAYM_HOME") && pushfirst!(LOAD_PATH, ENV["KADANOFF_BAYM_HOME"])

using Test
using KadanoffBaym

include("structs.jl")
include("properties.jl")
