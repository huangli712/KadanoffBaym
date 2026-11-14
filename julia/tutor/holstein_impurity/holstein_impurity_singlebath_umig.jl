#
# Usage:
#
#   julia --project=. tutor/holstein_impurity/holstein_impurity_singlebath_umig.jl \
#         tutor/holstein_impurity/holstein_impurity_singlebath_umig.toml [--profile quick|release] [--validate-only]
#
# Holstein impurity with a single bath site, unrenormalized Migdal (uMig)
# approximation (NESSi examples/programs/holstein_impurity_singlebath_umig.cpp).
#

include(joinpath(@__DIR__, "holstein_impurity_impl.jl"))

main(args; io::IO = stdout) = holstein_impurity_main(args, "uMig"; io = io)

abspath(PROGRAM_FILE) == (@__FILE__) && impurity_cli(ARGS, "uMig")
