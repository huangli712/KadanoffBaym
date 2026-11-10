#
# Usage:
#
#   julia --project=. tutor/holstein_impurity/Holstein_impurity_singlebath_uMig.jl \
#         tutor/holstein_impurity/Holstein_impurity_singlebath_uMig.toml [--profile quick|release] [--validate-only]
#
# Holstein impurity with a single bath site, unrenormalized Migdal (uMig)
# approximation (NESSi examples/programs/Holstein_impurity_singlebath_uMig.cpp).
#

include(joinpath(@__DIR__, "holstein_impurity_impl.jl"))

main(args; io::IO = stdout) = holstein_impurity_main(args, "uMig"; io = io)

abspath(PROGRAM_FILE) == (@__FILE__) && impurity_cli(ARGS, "uMig")
