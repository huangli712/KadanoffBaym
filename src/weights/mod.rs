// Project : KadanoffBaym
// File    : src/weights/mod.rs
// Purpose : integration orders and the Integrator bundle

pub mod calc;
pub mod integrator;
mod rational;
pub mod types;

pub use types::{
    BackwardDifferentiationWeights, BoundaryConvolutionWeights, GregoryIntegrationWeights,
    PolynomialDifferentiationWeights, PolynomialIntegrationWeights, PolynomialInterpolationWeights,
};

pub use integrator::Integrator;
