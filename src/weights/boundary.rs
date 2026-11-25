// Project : KadanoffBaym
// File    : src/weights/boundary.rs
// Purpose : boundary convolution weight calculators: the analytic
//           R integral and the rcorr-style tables :

///
/// *Remarks* : ``\mathbf{R}`` *Integral*
///
/// In order to evaluate the following integration analytically
///
/// ```math
/// \begin{equation}
/// \mathbf{R} = \int^{m}_0 dx\ (m-x)^a x^b,
/// \end{equation}
/// ```
///
/// we execute the following python codes:
///
/// ```text
/// import sympy
/// x, a, b, m = sympy.symbols('x a b m')
/// c = sympy.integrate((m-x)**a*x**b,(x,0,m))
/// c = sympy.simplify(c)
/// sympy.pprint(c)
/// ```
///
/// The output is as follows:
///
/// ```math
/// \begin{equation}
/// \mathbf{R} = \frac{m^{a+b+1}\Gamma(a+1)\Gamma(b+1)}{\Gamma(a+b+2)},
/// \end{equation}
/// ```
///
/// where ``\Gamma`` is a special function.
///
/// *Remarks* : ``\Gamma`` *Function*
///
/// The definition for the ``\Gamma`` function is as follows:
///
/// ```math
/// \begin{equation}
/// \Gamma(z) = \int^{\infty}_0 e^{-t} t^{z - 1} dt,
/// \end{equation}
/// ```
///
/// where ``z = x + iy`` and 𝑥 > 0.
///
///
/// Analytic integral `R(m, a, b) = integral_0^m (m-x)^a x^b dx =
/// m^{a+b+1} a! b! / (a+b+1)!`. The Gamma functions of the analytic
/// result are replaced by factorials: `a` and `b` are non-negative
/// integers (the loop indices of `calc_boundary_convolution`), so
/// `Gamma(a+1) = a!`, `Gamma(b+1) = b!`, and `Gamma(a+b+2) = (a+b+1)!`,
/// and no Gamma function is needed.
fn r_integral(m: i64, a: i64, b: i64) -> f64 {
    (m as f64).powi((a + b + 1) as i32) * factorial_f64(a) * factorial_f64(b)
        / factorial_f64(a + b + 1)
}

fn factorial_f64(n: i64) -> f64 {
    let mut acc = 1.0;
    for i in 2..=n {
        acc *= i as f64;
    }
    acc
}

/// Boundary convolution weights `Wc[m-1][i][j]` for `m = 1..k-1`,
/// row-major `(k-1) * (k+1) * (k+1)`; empty for `k = 1`.
pub(super) fn calc_boundary_convolution(k: i64, wi: &[f64]) -> Vec<f64> {
    let n = (k + 1) as usize;
    assert_eq!(wi.len(), n * n);
    let m_dim = (k - 1) as usize;
    let mut wc = vec![0.0f64; m_dim * n * n];
    for m in 1..=(k - 1) {
        for i in 0..n {
            for j in 0..n {
                let mut val = 0.0;
                for a in 0..n {
                    for b in 0..n {
                        val += wi[a * n + i] * wi[b * n + j] * r_integral(m, a as i64, b as i64);
                    }
                }
                wc[((m as usize - 1) * n + i) * n + j] = val;
            }
        }
    }
    wc
}
