#
# Usage:
#
#   julia --project=. tutor/holstein_bethe/holstein_bethe_nambu_migdal.jl \
#         tutor/holstein_bethe/holstein_bethe_nambu_migdal.toml [--profile quick|release] [--validate-only]
#
# Holstein model on the Bethe lattice in the Nambu (superconducting) form,
# self-consistent Migdal approximation (NESSi
# examples/programs/holstein_bethe_nambu_migdal.cpp).
#

include(joinpath(@__DIR__, "holstein_bethe_impl.jl"))

main(args; io::IO = stdout) = holstein_nambu_main(args, "Migdal"; io = io)

abspath(PROGRAM_FILE) == (@__FILE__) && holstein_cli(holstein_nambu_main, ARGS, "Migdal")
