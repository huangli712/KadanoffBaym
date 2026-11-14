// Project : KadanoffBaym
// File    : src/weights/types.rs
// Purpose : weight structs (blueprint: julia/src/weights/types.jl)

use super::calc;

/// Weights for polynomial interpolation, `W` row-major `(k+1)^2`.
#[derive(Clone, Debug)]
///
/// *Remarks* : *Polynomial Interpolation*
///
/// Consider a function ``y(t)`` which takes the values ``y_j`` at the points
/// ``t_j = jh`` of an equidistant mesh ``j = 0, 1, \cdots, k`` with time
/// step ``h``. We denote the 𝑘th order polynomial ``\tilde{y}(t)`` passing
/// through the points ``y(jh) = y_j``,
///
/// ```math
/// \begin{equation}
/// \tilde{y}(jh) = y_j, \quad j = 0, \cdots, k,
/// \end{equation}
/// ```text
///
/// by ``\mathcal{P}^{(k)}[y_0,\cdots,y_k](t)``. The interpolation can be
/// cast into the matrix form,
///
/// ```math
/// \begin{equation}
/// \mathcal{P}^{(k)}[y_0,\cdots,y_k](t)
///     =
///     \sum^{k}_{a,l=0}h^{-a}t^{a}P^{(k)}_{al}y_l,
/// \end{equation}
/// ```text
///
/// where
///
/// ```math
/// \begin{equation}
/// P^{(k)}_{al} = (M^{-1})_{al},
/// \end{equation}
/// ```text
///
/// with
///
/// ```math
/// \begin{equation}
/// M_{ja} = j^a.
/// \end{equation}
/// ```text
///
/// *References* :
///
/// Please see [`NESSi`] Section `8.1` for more details.
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
///
/// *Remarks* : *Polynomial Differentiation*
///
/// An approximation for the derivative ``dy/dt`` of a function can be
/// obtained by taking the exact derivative of the polynomial approximant:
///
/// ```math
/// \begin{equation}
/// \frac{dy}{dt} \Big|_{t = mh}
///     \approx
///     \frac{d}{dt} \mathcal{P}^{(k)}[y_0,\cdots,y_k](mh).
/// \end{equation}
/// ```text
///
/// We thus arrive at an approximative formula for polynomial differentiation
///
/// ```math
/// \begin{equation}
/// \frac{dy}{dt}\Big |_{t = mh}
///     \approx
///     h^{-1} \sum^{k}_{l = 0} D^{(k)}_{ml}y_l,
/// \end{equation}
/// ```text
///
/// with
///
/// ```math
/// \begin{equation}
/// D^{(k)}_{ml} = \sum^{k}_{a=1} P^{(k)}_{al}am^{a-1}.
/// \end{equation}
/// ```text
///
/// *References* :
///
/// Please see [`NESSi`] Section `8.2` for more details.
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
///
/// *Remarks* : *Polynomial Integration*
///
/// In some cases, the polynomial interpolation formula is also used to get
/// the approximation to an integral. For 0 ≤ 𝑚 ≤ 𝑛 ≤ 𝑘,
///
/// ```math
/// \begin{equation}
/// \int^{nh}_{mh} dt\ y(t)
///     \approx
///     \int^{nh}_{mh} dt\ \mathcal{P}^{(k)}[y_0,\cdots,y_k](t).
/// \end{equation}
/// ```text
///
/// We thus use the following approximate relation for polynomial integration
///
/// ```math
/// \begin{equation}
/// \int^{nh}_{mh} dt\ y(t)
///     \approx
///     h \sum^{k}_{l=0} I^{(k)}_{mnl}y_l,
/// \end{equation}
/// ```text
///
/// with
///
/// ```math
/// \begin{equation}
/// I^{(k)}_{mnl}
///     =
///     \sum^{k}_{a=0} P^{(k)}_{al} \frac{n^{a+1} - m^{a+1}}{a+1}.
/// \end{equation}
/// ```text
///
/// *References* :
///
/// Please see [`NESSi`] Section `8.3` for more details.
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
///
/// *Remarks* : *Backward Differentiation*
///
/// Consider a function ``y(t)`` which takes the values ``y_j`` at the points
/// ``t_j = jh`` of an equidistant mesh ``j = 0, \cdots, n`` with 𝑛 ≥ 𝑘. The
/// backward differentiation formula (`BDF`) of order 𝑘 approximates the
/// derivative ``dy/dt`` at ``t = nh`` using the function values ``y_n,
/// y_{n-1}, \cdots, y_{n-k}``. It is defined via the linear relation:
///
/// ```math
/// \begin{equation}
/// \frac{dy}{dt}\Big |_{nh}
///     \approx
///     h^{-1} \sum^{k}_{j=0} a^{(k)}_{j} y_{n-j}.
/// \end{equation}
/// ```text
///
/// Here the coefficients for the 𝑘-th order formula are obtained by the
/// derivative ``\tilde{y}'(t=0)`` of the 𝑘-th order polynomial interpolation
/// ``\tilde{y}(t)`` defined by the values ``\tilde{y}(jh) = y_{n-j}``,
///
/// ```math
/// \begin{equation}
/// \frac{dy}{dt}\Big |_{nh}
///     \approx
///     -\frac{d}{dt} \mathcal{P}^{(k)}[y_n, y_{n-1}, \cdots, y_{n-k}](t=0).
/// \end{equation}
/// ```text
///
/// Note that the minus sign is due to the reversed order of the interpolated
/// points. Therefore, the coefficients of the `BDF` are directly related to
/// the coefficients for polynomial differentiation:
///
/// ```math
/// \begin{equation}
/// a^{(k)}_j = -D^{(k)}_{0j}.
/// \end{equation}
/// ```text
///
/// *References* :
///
/// Please see [`NESSi`] Section `8.4` for more details.
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
///
/// *Remarks* : *Gregory Integration*
///
/// The solution of the Volterra integral equations (`VIEs`) is based on a
/// combination of backward differentiation formulae with so-called Gregory
/// quadrature rules for the integration. The 𝑘th order Gregory quadrature
/// rule on a linear mesh is defined by the following equation:
///
/// ```math
/// \begin{equation}
/// \mathcal{I}_n
///     =
///     \int^{nh}_0 dt\ y(t)
///     \approx
///     h \sum^{\max(n,k)}_{j=0} \Omega^{(k)}_{nj}y_j.
/// \end{equation}
/// ```text
///
/// The matrix form of the integration weights ``\Omega^{(k)}_{nj}`` is as
/// follows:
///
/// ```math
/// \begin{equation}
/// \Omega^{(k)}_{nj} =
/// \begin{pmatrix}
/// \sigma^{(k)}_{00} & \cdots & \sigma^{(k)}_{0k} & 0 & 0 &&&&& \\
/// \vdots && \vdots &&&&&&& \\
/// \sigma^{(k)}_{k0} & \cdots & \sigma^{(k)}_{kk} & 0 & 0 &&&&& \\
/// \Sigma^{(k)}_{00} & \cdots & \Sigma^{(k)}_{0k} & \omega^{(k)}_0 & 0 &&&&& \\
/// \vdots && \vdots && \ddots & \ddots &&&& \\
/// \Sigma^{(k)}_{k0} & \cdots & \Sigma^{(k)}_{kk} & \omega^{(k)}_k & \cdots & \omega^{(k)}_0 & 0 &&& \\
/// \omega^{(k)}_0 & \cdots & \omega^{(k)}_k & 1 & \omega^{(k)}_k & \cdots & \omega^{(k)}_0 & 0 && \\
/// \omega^{(k)}_0 & \cdots & \omega^{(k)}_k & 1 & 1 &\omega^{(k)}_k & \cdots & \omega^{(k)}_0 & 0 & \\
/// \omega^{(k)}_0 & \cdots & \omega^{(k)}_k & 1 & 1 & 1 &\omega^{(k)}_k & \cdots & \omega^{(k)}_0 & 0 \\
/// \end{pmatrix}
/// \end{equation}
/// ```text
///
/// Clearly, there are five cases for ``\Omega^{k}_{nj}``. Later, we will
/// introduce how to calculate the 𝑘th order Gregory integration weights
/// ``\sigma^{(k)}``, ``\Sigma^{(k)}``, and ``\omega^{(k)}``.
///
/// *References* :
///
/// Please see [`NESSi`] Section `8.5` for more details.
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
///
/// *Remarks* : *Boundary Convolution*
///
/// Here, we introduce a 𝑘th order accurate approximation for a special kind
/// of convolution integral, which appears in the context of imaginary time
/// convolutions. Consider the following convolution integral:
///
/// ```math
/// \begin{equation}
/// c(t) = \int^{t}_0 dt'\ F(t-t')G(t')
/// \end{equation}
/// ```text
///
/// between two functions 𝐹 and 𝐺 which are only defined for 𝑡 > 0, and can
/// not be continued into a differentiable function on the domain 𝑡 < 0. On
/// an equidistant mesh with ``t = mh`` and ``m < k``, the integration range
/// includes less than 𝑘 points, and the functions must be continued outside
/// the given integration range in order to obtain an 𝑘th order accurate
/// approximation.
///
/// We use the approximation
///
/// ```math
/// \begin{equation}
/// c(mh) = \int^{mh}_0 dt'\
///     \mathcal{P}^{(k)}[F_0,\cdots,F_k](mh-t')
///     \mathcal{P}^{(k)}[G_0,\cdots,G_k](t').
/// \end{equation}
/// ```text
///
/// Finally, we have
///
/// ```math
/// \begin{equation}
/// \int^{mh}_0 dt'\ F(t-t')G(t') =
///     h\sum^{k}_{r,s = 0}F_r G_s R^{(k)}_{mrs},
/// \end{equation}
/// ```text
///
/// where
///
/// ```math
/// \begin{equation}
/// R^{(k)}_{mrs} = \sum^{k}_{a,b = 0} P^{(k)}_{ar} P^{(k)}_{bs}
///     \int^{m}_{0} dx\ (m-x)^a x^b.
/// \end{equation}
/// ```text
///
/// *References* :
///
/// Please see [`NESSi`] Section `8.6` for more details.
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
