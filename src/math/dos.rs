// Project : KadanoffBaym
// File    : src/math/dos.rs
// Purpose : densities of states (blueprint: math/dos.jl)

use std::f64::consts::PI;

/// Semicircular (Bethe) density of states on `[lo, hi]`, centered at
/// `e0 = (lo + hi)/2` with `v = (hi - lo)/4` (bandwidth `4v`),
/// normalized to 1, zero outside `[lo, hi]`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BetheDos {
    pub lo: f64,
    pub hi: f64,
    pub v: f64,
    pub e0: f64,
}

impl BetheDos {
    /// Default constructor (semicircle on `[-2, 2]`).
    pub fn new(lo: f64, hi: f64) -> BetheDos {
        assert!(lo < hi, "BetheDos: lo must be smaller than hi");
        BetheDos {
            lo,
            hi,
            v: (hi - lo) / 4.0,
            e0: (hi + lo) / 2.0,
        }
    }

    /// Spectral weight at `x`.
    pub fn at(&self, x: f64) -> f64 {
        let arg = 4.0 * self.v * self.v - (x - self.e0) * (x - self.e0);
        let num = self.v * self.v * PI * 2.0;
        if arg < 0.0 {
            0.0
        } else {
            arg.sqrt() / num
        }
    }
}

impl Default for BetheDos {
    fn default() -> Self {
        BetheDos::new(-2.0, 2.0)
    }
}

/// Gaussian density of states, window `[mu - 6 sigma, mu + 6 sigma]`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GaussianDos {
    pub lo: f64,
    pub hi: f64,
    pub mu: f64,
    pub sigma: f64,
}

impl GaussianDos {
    pub fn new(mu: f64, sigma: f64) -> GaussianDos {
        assert!(sigma > 0.0, "GaussianDos: sigma must be positive");
        GaussianDos {
            lo: mu - 6.0 * sigma,
            hi: mu + 6.0 * sigma,
            mu,
            sigma,
        }
    }

    /// Spectral weight at `x` (not truncated at the window, like NESSi).
    pub fn at(&self, x: f64) -> f64 {
        let y = (x - self.mu) / self.sigma;
        (-0.5 * y * y).exp() / (self.sigma * 2.506623299999997)
    }
}

impl Default for GaussianDos {
    fn default() -> Self {
        GaussianDos::new(0.0, 1.0)
    }
}

/// Antisymmetric (bosonic) ohmic spectrum
/// `a(x) = sign(x) |x| exp(-|x|/omegac) / (2 omegac^2)`, window
/// `[-20 omegac, 20 omegac]` (function itself not truncated).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OhmicSymDos {
    pub lo: f64,
    pub hi: f64,
    pub omegac: f64,
}

impl OhmicSymDos {
    pub fn new(omegac: f64) -> OhmicSymDos {
        assert!(omegac > 0.0, "OhmicSymDos: omegac must be positive");
        OhmicSymDos {
            lo: -20.0 * omegac,
            hi: 20.0 * omegac,
            omegac,
        }
    }

    /// Spectral weight at `x`.
    pub fn at(&self, x: f64) -> f64 {
        let ax = x.abs();
        let s = if x < 0.0 {
            -1.0
        } else if x == 0.0 {
            0.0
        } else {
            1.0
        };
        s * ax * (-ax / self.omegac).exp() / (2.0 * self.omegac * self.omegac)
    }
}

impl Default for OhmicSymDos {
    fn default() -> Self {
        OhmicSymDos::new(1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bethe_normalized_and_window() {
        let d = BetheDos::default();
        assert_eq!((d.v, d.e0), (1.0, 0.0));
        assert_eq!(d.at(3.0), 0.0);
        assert_eq!(d.at(-3.0), 0.0);
        // peak value at e0: sqrt(4)/ (2*pi) = 1/pi
        assert!((d.at(0.0) - 1.0 / PI).abs() < 1e-15);
        // Riemann normalization ~ 1
        let n = 20000;
        let dx = (d.hi - d.lo) / n as f64;
        let sum: f64 = (0..n).map(|i| d.at(d.lo + (i as f64 + 0.5) * dx) * dx).sum();
        assert!((sum - 1.0).abs() < 1e-4, "sum = {sum}");
    }

    #[test]
    fn gaussian_peak_and_window() {
        let d = GaussianDos::default();
        // the blueprint normalizer is the literal 2.506623299999997
        assert!((d.at(0.0) - 1.0 / 2.506623299999997).abs() < 1e-15);
        assert_eq!((d.lo, d.hi), (-6.0, 6.0));
        // symmetry
        assert!((d.at(0.7) - d.at(-0.7)).abs() < 1e-15);
    }

    #[test]
    fn ohmic_sym_is_antisymmetric() {
        let d = OhmicSymDos::new(2.0);
        assert_eq!(d.at(0.0), 0.0);
        let p = d.at(1.3);
        let m = d.at(-1.3);
        assert!((p + m).abs() < 1e-16);
        // formula check
        let want = 1.3 * (-1.3f64 / 2.0).exp() / (2.0 * 4.0);
        assert!((p - want).abs() < 1e-15);
    }
}
