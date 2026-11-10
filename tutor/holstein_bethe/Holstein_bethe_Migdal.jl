#
# Usage:
#
#   julia --project=. tutor/holstein_bethe/Holstein_bethe_Migdal.jl \
#         tutor/holstein_bethe/Holstein_bethe_Migdal.toml [--profile quick|release] [--validate-only]
#
# Holstein model on the Bethe lattice, self-consistent Migdal
# approximation (NESSi examples/programs/Holstein_bethe_Migdal.cpp).
#

include(joinpath(@__DIR__, "holstein_bethe_impl.jl"))

main(args; io::IO = stdout) = holstein_bethe_main(args, "Migdal"; io = io)

abspath(PROGRAM_FILE) == (@__FILE__) && holstein_cli(holstein_bethe_main, ARGS, "Migdal")
