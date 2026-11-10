#
# Usage:
#
#   julia --project=. tutor/holstein_impurity/Holstein_impurity_singlebath_Migdal.jl \
#         tutor/holstein_impurity/Holstein_impurity_singlebath_Migdal.toml [--profile quick|release] [--validate-only]
#
# Holstein impurity with a single bath site, self-consistent Migdal
# approximation (NESSi examples/programs/Holstein_impurity_singlebath_Migdal.cpp).
#

include(joinpath(@__DIR__, "holstein_impurity_impl.jl"))

main(args; io::IO = stdout) = holstein_impurity_main(args, "Migdal"; io = io)

abspath(PROGRAM_FILE) == (@__FILE__) && impurity_cli(ARGS, "Migdal")
