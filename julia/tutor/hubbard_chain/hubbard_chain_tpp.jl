#
# Usage:
#
#   julia --project=. tutor/hubbard_chain/hubbard_chain_tpp.jl \
#         tutor/hubbard_chain/hubbard_chain.toml [--profile quick|release] [--validate-only]
#
# Hubbard chain in the particle-particle T-matrix approximation (NESSi
# examples/programs/hubbard_chain_tpp.cpp).
#

include(joinpath(@__DIR__, "hubbard_chain_impl.jl"))

main(args) = hubbard_main(args, "TPP")

abspath(PROGRAM_FILE) == (@__FILE__) && hubbard_cli(ARGS, "TPP")
