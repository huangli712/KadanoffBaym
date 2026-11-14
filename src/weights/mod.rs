// Project : KadanoffBaym
// File    : src/weights/mod.rs
// Purpose : integration orders and the Integrator bundle

mod calc;
mod integrator;
mod rational;
mod types;

pub use types::{
    BackwardDifferentiationWeights, BoundaryConvolutionWeights, GregoryIntegrationWeights,
    PolynomialDifferentiationWeights, PolynomialIntegrationWeights, PolynomialInterpolationWeights,
};

pub use integrator::Integrator;
