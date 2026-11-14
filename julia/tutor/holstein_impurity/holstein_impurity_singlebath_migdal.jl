#
# Usage:
#
#   julia --project=. tutor/holstein_impurity/holstein_impurity_singlebath_migdal.jl \
#         tutor/holstein_impurity/holstein_impurity_singlebath_migdal.toml [--profile quick|release] [--validate-only]
#
# Holstein impurity with a single bath site, self-consistent Migdal
# approximation (NESSi examples/programs/holstein_impurity_singlebath_migdal.cpp).
#

include(joinpath(@__DIR__, "holstein_impurity_impl.jl"))

main(args; io::IO = stdout) = holstein_impurity_main(args, "Migdal"; io = io)

abspath(PROGRAM_FILE) == (@__FILE__) && impurity_cli(ARGS, "Migdal")
