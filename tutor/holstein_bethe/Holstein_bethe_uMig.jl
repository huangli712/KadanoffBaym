#
# Usage:
#
#   julia --project=. tutor/holstein_bethe/Holstein_bethe_uMig.jl \
#         tutor/holstein_bethe/Holstein_bethe_uMig.toml [--profile quick|release] [--validate-only]
#
# Holstein model on the Bethe lattice, unrenormalized Migdal (uMig)
# approximation (NESSi examples/programs/Holstein_bethe_uMig.cpp).
#

include(joinpath(@__DIR__, "holstein_bethe_impl.jl"))

main(args; io::IO = stdout) = holstein_bethe_main(args, "uMig"; io = io)

abspath(PROGRAM_FILE) == (@__FILE__) && holstein_cli(holstein_bethe_main, ARGS, "uMig")
