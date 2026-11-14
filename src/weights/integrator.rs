// Project : KadanoffBaym
// File    : src/weights/integrator.rs
// Purpose : bundle of all weight tables for a given solver order
//           (blueprint julia/src/weights/integrator.jl)

use super::types::{
    BackwardDifferentiationWeights, BoundaryConvolutionWeights, GregoryIntegrationWeights,
    PolynomialDifferentiationWeights, PolynomialIntegrationWeights, PolynomialInterpolationWeights,
};

/// Bundle of all weight tables for a given solver order `k`
/// (blueprint `Integrator`; `k >= 1` because of the boundary
/// convolution weights, `k <= 10`).
pub struct Integrator {
    pub k: i64,
    pub piw: PolynomialInterpolationWeights,
    pub pdw: PolynomialDifferentiationWeights,
    pub xiw: PolynomialIntegrationWeights,
    pub bdw: BackwardDifferentiationWeights,
    pub giw: GregoryIntegrationWeights,
    pub bcw: BoundaryConvolutionWeights,
}

impl Integrator {
    pub fn new(k: i64) -> Self {
        assert!((1..=5).contains(&k), "Integrator: 1 <= k <= 5, got {k}");
        Integrator {
            k,
            piw: PolynomialInterpolationWeights::new(k),
            pdw: PolynomialDifferentiationWeights::new(k),
            xiw: PolynomialIntegrationWeights::new(k),
            bdw: BackwardDifferentiationWeights::new(k),
            giw: GregoryIntegrationWeights::new(k),
            bcw: BoundaryConvolutionWeights::new(k),
        }
    }
}
