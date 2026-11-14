#
# Usage:
#
#   julia --project=. tutor/hubbard_chain/hubbard_chain_gw.jl \
#         tutor/hubbard_chain/hubbard_chain.toml [--profile quick|release] [--validate-only]
#
# Hubbard chain in the GW approximation (NESSi
# examples/programs/hubbard_chain_gw.cpp).
#

include(joinpath(@__DIR__, "hubbard_chain_impl.jl"))

main(args) = hubbard_main(args, "GW")

abspath(PROGRAM_FILE) == (@__FILE__) && hubbard_cli(ARGS, "GW")
