// Project : KadanoffBaym
// File    : src/weights/calc_gregory.rs
// Purpose : Gregory quadrature weight calculators: trapezoid rule,
//           Laplace coefficients, gamma corrections, and the
//           (sigma, Sigma, omega) tables (blueprint:
//           julia/src/weights/calc.jl)
//
// The Laplace coefficients and Gregory corrections use exact rational
// arithmetic (`Rational`, i128) like Julia's `Rational{I64}`, converted
// to `f64` only at the end.

use crate::weights::rational::Rational;

/// Trapezoid-rule weights for `n + 1` nodes (exact rationals).
pub(crate) fn trapezoid(n: i64) -> Vec<Rational> {
    let mut w = vec![Rational::from_int(1); (n + 1) as usize];
    w[0] = Rational::new(1, 2);
    w[n as usize] = Rational::new(1, 2);
    w
}

/// Laplace coefficient `Lambda_k` (`Lambda_0 = -1`, recursion
/// `Lambda_k = sum_{l<k} Lambda_l / (l - k - 1)`), blueprint `Λ()`.
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
pub(crate) fn lambda(k: i64) -> Rational {
    if k == 0 {
        return Rational::from_int(-1);
    }
    let mut acc = Rational::from_int(0);
    for l in 0..k {
        acc = acc.add(lambda(l).div(Rational::from_int((l - k - 1) as i128)));
    }
    acc
}

/// Integer binomial `C(n, r)` for small non-negative arguments.
fn binomial(n: i64, r: i64) -> i128 {
    if r < 0 || r > n {
        return 0;
    }
    let r = r.min(n - r);
    let mut c: i128 = 1;
    for i in 0..r {
        c = c * (n - i) as i128 / (i + 1) as i128;
    }
    c
}

/// Gregory correction coefficients `gamma_j`, `j = 0..k` (exact
/// rationals), blueprint `γⱼ()`.
///
/// *Remarks* : ``\gamma^{(k)}_j`` *Coefficients*
///
/// The 𝑘th order coefficients ``\gamma^{(k)}_j`` are used to calculate the
/// integration weights by Gregory quadrature rules. They are defined in the
/// previous *Remarks*. Here, we introduce two slightly different formulas.
///
/// ```math
/// \begin{equation}
/// \gamma_0 = -\sum^{k}_{i=1} \Lambda_{i+1},
/// \end{equation}
/// ```
///
/// and
///
/// ```math
/// \begin{equation}
/// \gamma_j = \sum^{k}_{i=j} \Lambda_{i+1} (-1)^{j+1}\binom{i}{j},
/// \end{equation}
/// ```
///
/// where ``j \in [1,k]``, and ``\Lambda_i`` are the Laplace coefficients.
pub(crate) fn gregory_gammas(k: i64) -> Vec<Rational> {
    let mut g = vec![Rational::from_int(0); (k + 1) as usize];
    // gamma_0 = -sum_{i=1..k} Lambda_{i+1}
    for i in 1..=k {
        g[0] = g[0].sub(lambda(i + 1));
    }
    // gamma_j (j = 1..k) = sum_{i=j..k} Lambda_{i+1} (-1)^{j+1} C(i, j)
    for j in 1..=k {
        let mut acc = Rational::from_int(0);
        for i in j..=k {
            let term = lambda(i + 1);
            let sign_flip = if (j + 1) % 2 == 0 { 1 } else { -1 };
            let scaled = Rational::new(term.num * sign_flip * binomial(i, j), term.den);
            acc = acc.add(scaled);
        }
        g[j as usize] = acc;
    }
    g
}

/// Gregory integration weights for `n > k` nodes (blueprint
/// `calc_gregory_weights`): trapezoid plus forward/backward `gamma`
/// corrections, exact rationals converted to `f64`, length `n + 1`.
pub(crate) fn calc_gregory_weights(k: i64, n: i64) -> Vec<f64> {
    assert!(n > k, "calc_gregory_weights requires n > k");
    let w = trapezoid(n);
    let g = gregory_gammas(k);
    let mut w: Vec<Rational> = w;
    for j in 0..=(k as usize) {
        // forward differences at the left end
        w[j] = w[j].add(g[j]);
        // backward differences at the right end
        let last = w.len() - 1 - j;
        w[last] = w[last].add(g[j]);
    }
    w.iter().map(|r| r.to_f64()).collect()
}

/// Gregory integration weight tables `(sigma, Sigma, omega)` from the
/// polynomial integration weights `wt` (blueprint
/// `calc_gregory_integration`).
///
/// *Remarks* : *Gregory Quadrature Rules*
///
/// Here, we would like to introduce how to calculate the 𝑘th order Gregory
/// integration weights ``\Omega^{(k)}_{nj}`` step by step. It is not a piece
/// of cake.
///
/// (1) 𝑛 ≤ 𝑘: ``\mathcal{I}_n`` is approximated by the exact integral over
/// the polynomial interpolation ``\mathcal{P}^{(k)}[y_0,\cdots,y_k](t)``,
///
/// ```math
/// \begin{equation}
/// \mathcal{I}_n \approx
///     \int^{nh}_0 dt\ \mathcal{P}^{(k)}[y_0,\cdots,y_k](t)
///     \equiv
///     h \sum^{k}_{j=0}\sigma^{(k)}_{nj}y_j.
/// \end{equation}
/// ```
///
/// Hence the weights for 𝑛 ≤ 𝑘, which are denoted as *starting weights*
/// ``\Omega^{(k)}_{nj} = \sigma^{(k)}_{nj}``, are equivalent to the
/// polynomial integration weights,
///
/// ```math
/// \begin{equation}
/// \Omega^{(k)}_{nj} = I^{(k)}_{0nj} = \sigma^{(k)}_{nj}, \quad 0 \le n \le k.
/// \end{equation}
/// ```
///
/// See [`NESSi`] Eq.~(94) and [95].
///
/// (2) 𝑛 > 𝑘: We adopt the formulism presented in [`QUADRATURE`].
///
/// The 𝑘th order Gregory quadrature formula for 𝑛 + 1 nodes reads:
///
/// ```math
/// \begin{equation}
/// 𝑄^{(k)}(f) =
///     \sum^{n}_{j=0} a_j f(t_j) +
///     \sum^{k}_{j=0} u_j f(t_j) +
///     \sum^{n}_{j=n-k} v_j f(t_j).
/// \end{equation}
/// ```
///
/// Here, ``a_j`` correspond to the weights for the `trapezoidal rule`, the
/// ``u_j`` correspond to the contribution of the forward differences, and
/// the ``v_j`` are the weights from the backward differences. Hence, the
/// final weights are a linear combination of ``a_j``, ``u_j``, and ``v_j``.
///
/// At first, ``a_j`` is easily to be computed.
///
/// Later, let us focus on ``u_j`` and ``v_j``. They fulfill the following
/// relation,
///
/// ```math
/// \begin{equation}
/// u_j = v_{n-j} = \gamma^{(k)}_{j}, \quad j = 0, 1, 2, \cdots, k.
/// \end{equation}
/// ```
///
/// For ``j = 0``,
///
/// ```math
/// \begin{equation}
/// \gamma^{(k)}_0 = \sum^{k}_{i=1} \mathcal{L}_{i+1}(-1)^{i+1}.
/// \end{equation}
/// ```
///
/// For ``j = 1, 2, \cdots, k``,
/// ```math
/// \begin{equation}
/// \gamma^{(k)}_j = \sum^{k}_{i=j}
///     \mathcal{L}_{i+1} (-1)^{i+j+1} \binom{i}{j},
/// \end{equation}
/// ```
///
/// where
///
/// ```math
/// \begin{equation}
/// \mathcal{L}_i = \int^{1}_0 \binom{t}{i} dt,
/// \end{equation}
/// ```
///
/// with
///
/// ```math
/// \begin{equation}
/// \binom{m}{n} = \frac{m!}{(m-n)!n!}.
/// \end{equation}
/// ```
/// ``\binom{m}{n}`` is the so-called binomial coefficients. From ``a_j``,
/// ``u_j``, and ``v_j``, we can easily calculate ``\Sigma^{(k)}``.
///
/// See [`QUADRATURE`] Eq.~(2.2)-(2.10).
///
/// (3) Tricks. Notice that there is a relation between ``\Sigma^{(k)}``
/// and ``\omega^{(k)}``,
///
/// ```math
/// \begin{equation}
/// \omega^{(k)}_j = \Sigma^{(k)}_{kj}, \quad j = 0, \cdots, k.
/// \end{equation}
/// ```
///
/// Actually, we always use this equation to deduce ``\omega^{(k)}`` from
/// ``\Sigma^{(k)}``.
///
/// See [`NESSi`] Table `9`.
///
/// (4) Implementations. The weights ``a_j`` are calculated in function
/// `trapezoid()`. In functions `γⱼ()` and `Λ()`, the calculations of
/// ``\gamma^{(k)}_j`` and ``\mathcal{L}_i`` are implemented, respectively.
/// In function `calc_gregory_weights()`, the weights ``a_j``, ``u_j``, and
/// ``v_j`` are computed and combined.
///
/// Finally, in function `calc_gregory_integration()`, the Gregory quadrature
/// weights ``\sigma^{(k)}``, ``\Sigma^{(k)}``, and ``\omega^{(k)}`` are ready.
pub(crate) fn calc_gregory_integration(k: i64, wt: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let n = (k + 1) as usize;
    assert_eq!(wt.len(), n * n * n);
    let mut sigma = vec![0.0f64; n * n];
    let mut big_s = vec![0.0f64; n * n];
    let mut omega = vec![0.0f64; n];

    // starting weights: sigma[i][j] = Wt[m=0][i][j]
    for j in 0..n {
        for i in 0..n {
            sigma[i * n + j] = wt[(0 * n + i) * n + j];
        }
    }
    // Sigma rows for n = k+1 .. 2k+1
    for row in 0..n {
        let nbig = k + 1 + row as i64;
        let w = calc_gregory_weights(k, nbig);
        for j in 0..n {
            big_s[row * n + j] = w[j];
        }
    }
    // omega = last row of Sigma
    omega.copy_from_slice(&big_s[(n - 1) * n..n * n]);
    (sigma, big_s, omega)
}
