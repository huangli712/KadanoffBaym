#
# Project : Lavender
# Source  : integrator.jl
# Author  : Li Huang (huangli@caep.cn)
# Status  : Unstable
#
# Last modified: 2026/11/10
#

#=
### *Public API*
=#

"""
    Integrator

TO_BE_DONE
"""
struct Integrator
    k   :: I64
    PIW :: PolynomialInterpolationWeights
    PDW :: PolynomialDifferentiationWeights
    XIW :: PolynomialIntegrationWeights
    BDW :: BackwardDifferentiationWeights
    GIW :: GregoryIntegrationWeights
    BCW :: BoundaryConvolutionWeights
end


"""
    Integrator(k::I64)

TO_BE_DONE
"""
function Integrator(k::I64)
    PIW = PolynomialInterpolationWeights(k)
    PDW = PolynomialDifferentiationWeights(k)
    XIW = PolynomialIntegrationWeights(k)
    BDW = BackwardDifferentiationWeights(k)
    GIW = GregoryIntegrationWeights(k)
    BCW = BoundaryConvolutionWeights(k)

    Integrator(k, PIW, PDW, XIW, BDW, GIW, BCW)
end
