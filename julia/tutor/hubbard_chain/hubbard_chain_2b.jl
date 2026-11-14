#
# Usage:
#
#   julia --project=. tutor/hubbard_chain/hubbard_chain_2b.jl \
#         tutor/hubbard_chain/hubbard_chain.toml [--profile quick|release] [--validate-only]
#
# Hubbard chain in the second Born approximation (NESSi
# examples/programs/hubbard_chain_2b.cpp).
#

include(joinpath(@__DIR__, "hubbard_chain_impl.jl"))

main(args) = hubbard_main(args, "2B")

abspath(PROGRAM_FILE) == (@__FILE__) && hubbard_cli(ARGS, "2B")
