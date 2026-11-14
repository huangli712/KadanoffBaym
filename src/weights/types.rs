// Project : KadanoffBaym
// File    : src/weights/types.rs
// Purpose : weight structs (blueprint: julia/src/weights/types.jl)

use super::calc;

/// Weights for polynomial interpolation, `W` row-major `(k+1)^2`.
#[derive(Clone, Debug)]
pub struct PolynomialInterpolationWeights {
    /// Order (0 <= k <= 10).
    pub k: i64,
    /// Weight matrix.
    pub w: Vec<f64>,
}

impl PolynomialInterpolationWeights {
    pub fn new(k: i64) -> Self {
        assert!(
            (0..=10).contains(&k),
            "PolynomialInterpolationWeights: 0 <= k <= 10, got {k}"
        );
        Self {
            k,
            w: calc::calc_poly_interpolation(k),
        }
    }
}

/// Weights for polynomial differentiation, `W` row-major `(k+1)^2`.
#[derive(Clone, Debug)]
pub struct PolynomialDifferentiationWeights {
    pub k: i64,
    pub w: Vec<f64>,
}

impl PolynomialDifferentiationWeights {
    pub fn new(k: i64) -> Self {
        assert!(
            (0..=10).contains(&k),
            "PolynomialDifferentiationWeights: 0 <= k <= 10, got {k}"
        );
        let wi = calc::calc_poly_interpolation(k);
        Self {
            k,
            w: calc::calc_poly_differentiation(k, &wi),
        }
    }
}

/// Weights for polynomial integration, `W` row-major `(k+1)^3` indexed
/// as `(m * (k+1) + n) * (k+1) + l`.
#[derive(Clone, Debug)]
pub struct PolynomialIntegrationWeights {
    pub k: i64,
    pub w: Vec<f64>,
}

impl PolynomialIntegrationWeights {
    pub fn new(k: i64) -> Self {
        assert!(
            (0..=10).contains(&k),
            "PolynomialIntegrationWeights: 0 <= k <= 10, got {k}"
        );
        let wi = calc::calc_poly_interpolation(k);
        Self {
            k,
            w: calc::calc_poly_integration(k, &wi),
        }
    }

    /// Element access `W[m][n][l]`.
    #[inline]
    pub fn at(&self, m: usize, n: usize, l: usize) -> f64 {
        let s = (self.k + 1) as usize;
        self.w[(m * s + n) * s + l]
    }
}

/// Weights for backward differentiation, `W` of length `k + 2`
/// (the stored `k` is the user order plus one, like the blueprint).
#[derive(Clone, Debug)]
pub struct BackwardDifferentiationWeights {
    pub k: i64,
    pub w: Vec<f64>,
}

impl BackwardDifferentiationWeights {
    pub fn new(k: i64) -> Self {
        assert!(
            (0..=10).contains(&k),
            "BackwardDifferentiationWeights: 0 <= k <= 10, got {k}"
        );
        let k1 = k + 1;
        let wi = calc::calc_poly_interpolation(k1);
        Self {
            k: k1,
            w: calc::calc_backward_differentiation(k1, &wi),
        }
    }
}

/// Weights for Gregory integration: starting weights `sigma` and full
/// rows `big_s` (row-major `(k+1)^2`) plus the tail corrections `omega`
/// (length `k + 1`).
#[derive(Clone, Debug)]
pub struct GregoryIntegrationWeights {
    pub k: i64,
    pub sigma: Vec<f64>,
    pub big_s: Vec<f64>,
    pub omega: Vec<f64>,
}

impl GregoryIntegrationWeights {
    pub fn new(k: i64) -> Self {
        assert!(
            (0..=10).contains(&k),
            "GregoryIntegrationWeights: 0 <= k <= 10, got {k}"
        );
        let wi = calc::calc_poly_interpolation(k);
        let wt = calc::calc_poly_integration(k, &wi);
        let (sigma, big_s, omega) = calc::calc_gregory_integration(k, &wt);
        Self {
            k,
            sigma,
            big_s,
            omega,
        }
    }
}

/// Weights for boundary convolution, `W` row-major
/// `(k-1) * (k+1) * (k+1)` indexed as `((m-1) * (k+1) + i) * (k+1) + j`;
/// empty for `k = 1` (like NESSi's `rcorr`).
#[derive(Clone, Debug)]
pub struct BoundaryConvolutionWeights {
    /// Order (1 <= k <= 10).
    pub k: i64,
    pub w: Vec<f64>,
}

impl BoundaryConvolutionWeights {
    pub fn new(k: i64) -> Self {
        assert!(
            (1..=10).contains(&k),
            "BoundaryConvolutionWeights: 1 <= k <= 10, got {k}"
        );
        let wi = calc::calc_poly_interpolation(k);
        Self {
            k,
            w: calc::calc_boundary_convolution(k, &wi),
        }
    }

    /// Element access `W[m][i][j]` with `m = 1..k-1`.
    #[inline]
    pub fn at(&self, m: usize, i: usize, j: usize) -> f64 {
        let s = (self.k + 1) as usize;
        self.w[((m - 1) * s + i) * s + j]
    }
}

impl PolynomialInterpolationWeights {
    /// Element access `W[i][j]` (0-based, like the blueprint getindex).
    #[inline]
    pub fn at(&self, i: usize, j: usize) -> f64 {
        self.w[i * (self.k + 1) as usize + j]
    }
}

impl PolynomialDifferentiationWeights {
    /// Element access `W[i][j]` (0-based).
    #[inline]
    pub fn at(&self, i: usize, j: usize) -> f64 {
        self.w[i * (self.k + 1) as usize + j]
    }
}

impl BackwardDifferentiationWeights {
    /// Element access `W[l]` (0-based, `l = 0..=k+1` in user order).
    #[inline]
    pub fn at(&self, l: usize) -> f64 {
        self.w[l]
    }
}

impl GregoryIntegrationWeights {
    /// Full Omega-matrix element `Omega[n][j]` with the blueprint's
    /// five-case lookup (0-based `n`, `j`).
    pub fn at(&self, n: usize, j: usize) -> f64 {
        let k = self.k as usize;
        let s = k + 1;
        let mut val = 0.0;
        // 0 <= n, j <= k
        if n <= k && j <= k {
            val = self.sigma[n * s + j];
        }
        // k+1 <= n <= 2k+1 (Julia Sigma[n-k] 1-based -> big_s row n-k-1)
        if (k + 1..=2 * k + 1).contains(&n) {
            if j <= k {
                val = self.big_s[(n - k - 1) * s + j];
            } else if j <= n {
                val = self.omega[n - j];
            }
        }
        // n >= 2k+2
        if n >= 2 * k + 2 {
            if j <= k {
                val = self.omega[j];
            } else if j >= k + 1 && j <= n - k - 1 {
                val = 1.0;
            } else if j >= n - k && j <= n {
                val = self.omega[n - j];
            }
        }
        val
    }
}
