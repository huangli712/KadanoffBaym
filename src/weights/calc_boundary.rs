// Project : KadanoffBaym
// File    : src/weights/calc_boundary.rs
// Purpose : boundary convolution weight calculators: the analytic
//           R integral and the rcorr-style tables (blueprint:
//           julia/src/weights/calc.jl)

/// Analytic integral `R(m, a, b) = integral_0^m (m-x)^a x^b dx =
/// m^{a+b+1} a! b! / (a+b+1)!`, blueprint `𝐑()`.
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
pub(crate) fn r_integral(m: i64, a: i64, b: i64) -> f64 {
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

/// Boundary convolution weights `Wc[m-1][i][j]` for `m = 1..k-1`
/// (blueprint Eq. 104), row-major `(k-1) * (k+1) * (k+1)`; empty for
/// `k = 1` (like NESSi's `rcorr`).
///
/// *Remarks* : *Laplace Coefficients*
///
/// In previous *Remarks*, we introduce a variable ``\mathcal{L}_k``. Let us
/// recall it at first.
///
/// ```math
/// \begin{equation}
/// \mathcal{L}_k = \int^{1}_0 \binom{t}{k} dt,
/// \end{equation}
/// ```
///
/// It would be better to calculate the Laplace coefficients ``\Lambda_k``,
/// instead of calculating ``\mathcal{L}_k`` directly. The relation between
/// ``\mathcal{L}_k`` and ``\Lambda_k`` is as follows,
///
/// ```math
/// \begin{equation}
/// \Lambda_k = \mathcal{L}_k (-1)^{k-1}.
/// \end{equation}
/// ```
///
/// Thus, the definition of the Laplace coefficients ``\Lambda_k`` is:
///
/// ```math
/// \begin{equation}
/// \Lambda_k = (-1)^{k-1} \int^{1}_0 dt \binom{t}{k},
/// \end{equation}
/// ```
///
/// where 𝑘 ≥ 0, and the binomial function is defined as follows:
///
/// ```math
/// \begin{equation}
/// \binom{t}{k} = \frac{t!}{(t-k)!k!}.
/// \end{equation}
/// ```
///
/// Specially, for arbitary 𝑡, we have
///
/// ```math
/// \begin{equation}
/// \binom{t}{0} \equiv 1.
/// \end{equation}
/// ```
///
/// Here we apply the following recursive formulas to evaluate ``\Lambda_k``:
///
/// ```math
/// \begin{equation}
/// \Lambda_0 = -1,
/// \end{equation}
/// ```
///
/// and
///
/// ```math
/// \begin{equation}
/// \Lambda_k = \sum^{k-1}_{l=0} \frac{\Lambda_l}{l - k - 1}, \quad k \ge 1.
/// \end{equation}
/// ```
///
/// Then we get:
/// ```math
/// \begin{equation}
/// \Lambda_1 = \frac{1}{2},\
/// \Lambda_2 = \frac{1}{12},\
/// \Lambda_3 = \frac{1}{24},\
/// \Lambda_4 = \frac{19}{720},\
/// \Lambda_5 = \frac{3}{160},\
/// \Lambda_6 = \frac{863}{60480}.
/// \end{equation}
/// ```
///
/// The above values can be used as benchmark. This algorithm is implemented
/// in function `Λ()`.
pub(crate) fn calc_boundary_convolution(k: i64, wi: &[f64]) -> Vec<f64> {
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
