// Project : KadanoffBaym
// File    : src/weights/integrator.rs
// Purpose : bundle of all weight tables for a given solver order

//!
//! *Remarks* : *Weight Bundle*
//!
//! [`Integrator`] bundles the six weight tables of one solver order `k`:
//! polynomial interpolation and differentiation, polynomial integration,
//! backward differentiation, Gregory integration, and boundary convolution.

use std::sync::{Mutex, OnceLock};

use crate::weights::BackwardDifferentiationWeights;
use crate::weights::BoundaryConvolutionWeights;
use crate::weights::GregoryIntegrationWeights;
use crate::weights::PolynomialDifferentiationWeights;
use crate::weights::PolynomialIntegrationWeights;
use crate::weights::PolynomialInterpolationWeights;

/// Bundle of all weight tables for a given solver order `k`.
#[derive(Clone)]
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
    /// Build the weight tables of order `k` (`1 <= k <= 5`). The tables
    /// are a pure function of `k`, and the exact-rational construction
    /// costs far more than a clone of the few-KB result, so each order is
    /// built once and later calls return a clone of the cached bundle.
    pub fn new(k: i64) -> Self {
        assert!((1..=5).contains(&k), "Integrator: 1 <= k <= 5, got {k}");
        static CACHE: OnceLock<Mutex<[Option<Integrator>; 6]>> = OnceLock::new();
        let cache = CACHE.get_or_init(|| Mutex::new([None, None, None, None, None, None]));
        let mut slot = cache.lock().unwrap();
        if let Some(cached) = &slot[k as usize] {
            return cached.clone();
        }
        let it = Integrator {
            k,
            piw: PolynomialInterpolationWeights::new(k),
            pdw: PolynomialDifferentiationWeights::new(k),
            xiw: PolynomialIntegrationWeights::new(k),
            bdw: BackwardDifferentiationWeights::new(k),
            giw: GregoryIntegrationWeights::new(k),
            bcw: BoundaryConvolutionWeights::new(k),
        };
        slot[k as usize] = Some(it.clone());
        it
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cached_construction_returns_equal_tables() {
        let a = Integrator::new(5);
        let b = Integrator::new(5);
        assert_eq!(a.giw.sigma, b.giw.sigma);
        assert_eq!(a.giw.big_s, b.giw.big_s);
        assert_eq!(a.giw.omega, b.giw.omega);
        assert_eq!(a.bcw.w, b.bcw.w);
        assert_eq!(a.pdw.w, b.pdw.w);
        assert_eq!(a.bdw.w, b.bdw.w);
        // clones must not alias the cached storage
        assert_ne!(a.giw.sigma.as_ptr(), b.giw.sigma.as_ptr());
    }
}
