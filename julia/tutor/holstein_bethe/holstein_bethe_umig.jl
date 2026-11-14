#
# Usage:
#
#   julia --project=. tutor/holstein_bethe/holstein_bethe_umig.jl \
#         tutor/holstein_bethe/holstein_bethe_umig.toml [--profile quick|release] [--validate-only]
#
# Holstein model on the Bethe lattice, unrenormalized Migdal (uMig)
# approximation (NESSi examples/programs/holstein_bethe_umig.cpp).
#

include(joinpath(@__DIR__, "holstein_bethe_impl.jl"))

main(args; io::IO = stdout) = holstein_bethe_main(args, "uMig"; io = io)

abspath(PROGRAM_FILE) == (@__FILE__) && holstein_cli(holstein_bethe_main, ARGS, "uMig")
