// Project : KadanoffBaym
// File    : src/weights/types.rs
// Purpose : weight structs

use crate::weights::{backward, boundary, gregory, poly};

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
/// ```
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
/// ```
///
/// where
///
/// ```math
/// \begin{equation}
/// P^{(k)}_{al} = (M^{-1})_{al},
/// \end{equation}
/// ```
///
/// with
///
/// ```math
/// \begin{equation}
/// M_{ja} = j^a.
/// \end{equation}
/// ```
///
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
            w: poly::calc_poly_interpolation(k),
        }
    }

    /// Element access `W[i][j]` (0-based).
    #[inline]
    pub fn at(&self, i: usize, j: usize) -> f64 {
        self.w[i * (self.k + 1) as usize + j]
    }
}

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
/// ```
///
/// We thus arrive at an approximative formula for polynomial differentiation
///
/// ```math
/// \begin{equation}
/// \frac{dy}{dt}\Big |_{t = mh}
///     \approx
///     h^{-1} \sum^{k}_{l = 0} D^{(k)}_{ml}y_l,
/// \end{equation}
/// ```
///
/// with
///
/// ```math
/// \begin{equation}
/// D^{(k)}_{ml} = \sum^{k}_{a=1} P^{(k)}_{al}am^{a-1}.
/// \end{equation}
/// ```
///
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
        Self {
            k,
            w: poly::calc_poly_differentiation(k),
        }
    }

    /// Element access `W[i][j]` (0-based).
    #[inline]
    pub fn at(&self, i: usize, j: usize) -> f64 {
        self.w[i * (self.k + 1) as usize + j]
    }
}

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
/// ```
///
/// We thus use the following approximate relation for polynomial integration
///
/// ```math
/// \begin{equation}
/// \int^{nh}_{mh} dt\ y(t)
///     \approx
///     h \sum^{k}_{l=0} I^{(k)}_{mnl}y_l,
/// \end{equation}
/// ```
///
/// with
///
/// ```math
/// \begin{equation}
/// I^{(k)}_{mnl}
///     =
///     \sum^{k}_{a=0} P^{(k)}_{al} \frac{n^{a+1} - m^{a+1}}{a+1}.
/// \end{equation}
/// ```
///
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
        Self {
            k,
            w: poly::calc_poly_integration(k),
        }
    }

    /// Element access `W[m][n][l]`.
    #[inline]
    pub fn at(&self, m: usize, n: usize, l: usize) -> f64 {
        let s = (self.k + 1) as usize;
        self.w[(m * s + n) * s + l]
    }
}

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
/// ```
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
/// ```
///
/// Note that the minus sign is due to the reversed order of the interpolated
/// points. Therefore, the coefficients of the `BDF` are directly related to
/// the coefficients for polynomial differentiation:
///
/// ```math
/// \begin{equation}
/// a^{(k)}_j = -D^{(k)}_{0j}.
/// \end{equation}
/// ```
///
/// Weights for backward differentiation, `W` of length `k + 2`
/// (the stored `k` is the user order plus one).
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
        let wi = poly::calc_poly_interpolation(k1);
        Self {
            k: k1,
            w: backward::calc_backward_differentiation(k1, &wi),
        }
    }

    /// Element access `W[l]` (0-based, `l = 0..=k+1` in user order).
    #[inline]
    pub fn at(&self, l: usize) -> f64 {
        self.w[l]
    }
}

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
/// ```
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
/// ```
///
/// Clearly, there are five cases for ``\Omega^{k}_{nj}``. Later, we will
/// introduce how to calculate the 𝑘th order Gregory integration weights
/// ``\sigma^{(k)}``, ``\Sigma^{(k)}``, and ``\omega^{(k)}``.
///
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
        let wt = poly::calc_poly_integration(k);
        let (sigma, big_s, omega) = gregory::calc_gregory_integration(k, &wt);
        Self {
            k,
            sigma,
            big_s,
            omega,
        }
    }

    /// Full Omega-matrix element `Omega[n][j]` with the five-case
    /// lookup (0-based `n`, `j`); the cases are mutually
    /// exclusive, so the branches return early.
    pub fn at(&self, n: usize, j: usize) -> f64 {
        let k = self.k as usize;
        let s = k + 1;
        // 0 <= n, j <= k
        if n <= k && j <= k {
            return self.sigma[n * s + j];
        }
        // k+1 <= n <= 2k+1
        if (k + 1..=2 * k + 1).contains(&n) {
            if j <= k {
                return self.big_s[(n - k - 1) * s + j];
            } else if j <= n {
                return self.omega[n - j];
            }
        }
        // n >= 2k+2
        if n >= 2 * k + 2 {
            if j <= k {
                return self.omega[j];
            } else if j < n - k {
                return 1.0;
            } else if j <= n {
                return self.omega[n - j];
            }
        }
        0.0
    }
}

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
/// ```
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
/// ```
///
/// Finally, we have
///
/// ```math
/// \begin{equation}
/// \int^{mh}_0 dt'\ F(t-t')G(t') =
///     h\sum^{k}_{r,s = 0}F_r G_s R^{(k)}_{mrs},
/// \end{equation}
/// ```
///
/// where
///
/// ```math
/// \begin{equation}
/// R^{(k)}_{mrs} = \sum^{k}_{a,b = 0} P^{(k)}_{ar} P^{(k)}_{bs}
///     \int^{m}_{0} dx\ (m-x)^a x^b.
/// \end{equation}
/// ```
///
/// Weights for boundary convolution, `W` row-major
/// `(k-1) * (k+1) * (k+1)` indexed as `((m-1) * (k+1) + i) * (k+1) + j`;
/// empty for `k = 1`.
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
        Self {
            k,
            w: boundary::calc_boundary_convolution(k),
        }
    }

    /// Element access `W[m][i][j]` with `m = 1..k-1`.
    #[inline]
    pub fn at(&self, m: usize, i: usize, j: usize) -> f64 {
        let s = (self.k + 1) as usize;
        self.w[((m - 1) * s + i) * s + j]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::complex::Complex;
    use crate::weights::Integrator;

    use crate::math::fermi_tau;

    fn isclose(got: f64, want: f64, atol: f64, rtol: f64) -> bool {
        (got - want).abs() <= atol + rtol * want.abs()
    }

    #[test]
    fn polynomial_interpolation_weights() {
        let k = 5usize;
        let h = 0.1;
        let eps = 1.0e-4;
        let piw = PolynomialInterpolationWeights::new(k as i64);
        let ft: Vec<f64> = (0..=k).map(|x| (h * x as f64).cos()).collect();
        let mut err = 0.0;
        for i in 0..=k {
            let t = (i as f64 + 0.5) * h;
            let mut fint = 0.0;
            for (l, fl) in ft.iter().enumerate() {
                let mut t1 = 1.0;
                let mut weight = piw.at(0, l);
                for n in 1..=k {
                    t1 *= i as f64 + 0.5;
                    weight += t1 * piw.at(n, l);
                }
                fint += *fl * weight;
            }
            err += (t.cos() - fint).abs();
        }
        assert!(err < eps, "err = {err}");
    }

    #[test]
    fn polynomial_differentiation_weights() {
        let k = 5usize;
        let h = 0.1;
        let eps = 1.0e-4;
        let pdw = PolynomialDifferentiationWeights::new(k as i64);
        let ft: Vec<f64> = (0..=k).map(|x| (h * x as f64).cos()).collect();
        let mut err = 0.0;
        for i in 0..=k {
            let df_exact = -(h * i as f64).sin();
            let mut df_approx = 0.0;
            for (l, fl) in ft.iter().enumerate() {
                df_approx += (1.0 / h) * pdw.at(i, l) * *fl;
            }
            err += (df_exact - df_approx).abs();
        }
        assert!(err < eps, "err = {err}");
    }

    #[test]
    fn polynomial_integration_weights() {
        let k = 5usize;
        let h = 0.1;
        let eps = 1.0e-4;
        let xiw = PolynomialIntegrationWeights::new(k as i64);
        let ft: Vec<f64> = (0..=k).map(|x| (h * x as f64).cos()).collect();
        let mut err = 0.0;
        for i in 1..=k {
            for j in 0..i {
                let mut i_approx = 0.0;
                for (l, fl) in ft.iter().enumerate() {
                    i_approx += h * xiw.at(i, j, l) * *fl;
                }
                let i_exact = (h * j as f64).sin() - (h * i as f64).sin();
                err += (i_exact - i_approx).abs();
            }
        }
        assert!(err < eps, "err = {err}");
    }

    #[test]
    fn backward_differentiation_weights() {
        let k = 5usize;
        let h = 0.1;
        let eps = 1.0e-4;
        let bdw = BackwardDifferentiationWeights::new(k as i64);
        let t1 = 0.5;
        let mut df_approx = 0.0;
        for l in 0..=k + 1 {
            df_approx += bdw.at(l) * (t1 - l as f64 * h).cos() / h;
        }
        let df_exact = -t1.sin();
        assert!(
            (df_exact - df_approx).abs() < eps,
            "err = {}",
            (df_exact - df_approx).abs()
        );
    }

    #[test]
    fn gregory_integration_weights() {
        let k = 5usize;
        let nt = 100usize;
        let tmax = 10.0;
        let h = tmax / nt as f64;
        let eps = 1.0e-4;
        let giw = GregoryIntegrationWeights::new(k as i64);
        let ft1: Vec<f64> = (0..=nt).map(|x| (h * x as f64).cos()).collect();
        let ft2: Vec<Complex> = (0..=nt)
            .map(|i| Complex::from_polar(1.0, h * i as f64))
            .collect();
        let mut err1 = 0.0;
        let mut err2 = 0.0;
        for i in (k + 1)..=nt {
            let i_exact1 = (h * i as f64).sin();
            let i_exact2 = Complex::from_polar(1.0, h * i as f64) - Complex::ONE;
            let i_exact2 = Complex::new(0.0, -1.0) * i_exact2;
            let mut i_approx1 = 0.0;
            let mut i_approx2 = Complex::ZERO;
            for j in 0..=i {
                i_approx1 += giw.at(i, j) * ft1[j] * h;
                i_approx2 += ft2[j].scale(giw.at(i, j)).scale(h);
            }
            err1 += (i_exact1 - i_approx1).abs();
            err2 += (i_exact2 - i_approx2).abs();
        }
        assert!(err1 < eps, "err1 = {err1}");
        assert!(err2 < eps, "err2 = {err2}");
    }

    #[test]
    fn boundary_convolution_weights() {
        let k = 5usize;
        let ntau = 100usize;
        let eps = 1.0e-4;
        let eps1 = -0.1;
        let eps2 = 0.2;
        let beta = 5.0;
        let dtau = beta / ntau as f64;
        let bcw = BoundaryConvolutionWeights::new(k as i64);
        let mut a = vec![0.0; ntau + 1];
        let mut b = vec![0.0; ntau + 1];
        for m in 0..=ntau {
            a[m] = -fermi_tau(beta, m as f64 * dtau, -eps1);
            b[m] = -fermi_tau(beta, m as f64 * dtau, -eps2);
        }
        // R_exact[1..4] -> 0-based 0..3 (m = 1..4)
        let r_exact = [0.0, 0.0137659, 0.0274638, 0.0410948, 0.0546598];
        let mut err = 0.0;
        // for i = 1:k-1 -> m = i, BCW[i-1, j, l] (0-based m-1)
        for (i, re) in r_exact.iter().enumerate().take(k).skip(1) {
            let mut r_approx = 0.0;
            for (j, aj) in a.iter().enumerate().take(k + 1) {
                for (l, bl) in b.iter().enumerate().take(k + 1) {
                    r_approx += dtau * bcw.at(i, j, l) * *aj * *bl;
                }
            }
            err += (*re - r_approx).abs();
        }
        assert!(err < eps, "err = {err}");
    }

    /// The order `k` of `Integrator` maps to the same Gregory construction
    /// for every `k = 1, .., 5`; the expected tables are listed below.
    #[test]
    fn integrator_order_mapping_k_1_to_5() {
        let tol = 1e-14;
        // Gregory boundary weights omega_j
        let greg: Vec<Vec<f64>> = vec![
            vec![5.0 / 12.0, 13.0 / 12.0],
            vec![3.0 / 8.0, 7.0 / 6.0, 23.0 / 24.0],
            vec![251.0 / 720.0, 299.0 / 240.0, 211.0 / 240.0, 739.0 / 720.0],
            vec![
                95.0 / 288.0,
                317.0 / 240.0,
                23.0 / 30.0,
                793.0 / 720.0,
                157.0 / 160.0,
            ],
            vec![
                19087.0 / 60480.0,
                84199.0 / 60480.0,
                18869.0 / 30240.0,
                37621.0 / 30240.0,
                55031.0 / 60480.0,
                61343.0 / 60480.0,
            ],
        ];
        // Backward differentiation weights of order k + 1 (BDF(k+1))
        let bdf: Vec<Vec<f64>> = vec![
            vec![3.0 / 2.0, -2.0, 1.0 / 2.0],
            vec![11.0 / 6.0, -3.0, 3.0 / 2.0, -1.0 / 3.0],
            vec![25.0 / 12.0, -4.0, 3.0, -4.0 / 3.0, 1.0 / 4.0],
            vec![137.0 / 60.0, -5.0, 5.0, -10.0 / 3.0, 5.0 / 4.0, -1.0 / 5.0],
            vec![
                147.0 / 60.0,
                -6.0,
                15.0 / 2.0,
                -20.0 / 3.0,
                15.0 / 4.0,
                -6.0 / 5.0,
                1.0 / 6.0,
            ],
        ];
        for k in 1..=5i64 {
            let integ = Integrator::new(k);
            assert_eq!(integ.k, k);
            assert_eq!(integ.giw.k, k);
            assert_eq!(integ.bdw.k, k + 1);
            // BCW dims: (k-1, k+1, k+1)
            assert_eq!(integ.bcw.w.len(), ((k - 1) * (k + 1) * (k + 1)) as usize);
            // Gregory tail weights
            for (i, want) in greg[(k - 1) as usize].iter().enumerate() {
                let got = integ.giw.omega[i];
                assert!(
                    isclose(got, *want, tol, tol),
                    "k={k} omega[{i}] = {got}, want {want}"
                );
            }
            // Interior Gregory weights are unity
            let n = (4 * k + 4) as usize;
            let j = (2 * k + 2) as usize;
            assert!(isclose(integ.giw.at(n, j), 1.0, tol, tol));
            // BDW (round-off grows slightly with k, about 4e-14 for k = 5)
            let btol = if k <= 3 { tol } else { 1e-12 };
            for (l, want) in bdf[(k - 1) as usize].iter().enumerate() {
                let got = integ.bdw.at(l);
                assert!(
                    isclose(got, *want, btol, btol),
                    "k={k} bdw[{l}] = {got}, want {want}"
                );
            }
        }
        // k = 1: the boundary corrections are not defined, so W is empty.
        let integ = Integrator::new(1);
        assert_eq!(integ.bcw.w.len(), 0);
        // BCW.k stays 1, and the (k+1, k+1) block dims are implied by the
        // empty first dimension.
        assert_eq!(integ.bcw.k, 1);
    }
}
