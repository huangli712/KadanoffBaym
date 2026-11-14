// Project : KadanoffBaym
// File    : src/weights/calc.rs
// Purpose : weight calculators (blueprint: julia/src/weights/calc.jl)
//
// All formulas mirror the Julia blueprint exactly; the Laplace
// coefficients and Gregory corrections use exact rational arithmetic
// (`Rational`, i128) like Julia's `Rational{I64}`, converted to `f64`
// only at the end.

use crate::c64::C64;
use crate::mat::DenseMat;

/// Exact rational number (i128 numerator / denominator, always reduced,
/// denominator > 0).
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Rational {
    pub num: i128,
    pub den: i128,
}

impl Rational {
    pub fn new(num: i128, den: i128) -> Rational {
        assert!(den != 0, "Rational: zero denominator");
        let (mut n, mut d) = (num, den);
        if d < 0 {
            n = -n;
            d = -d;
        }
        let g = gcd(n.abs(), d);
        Rational {
            num: n / g,
            den: d / g,
        }
    }

    pub fn from_int(n: i128) -> Rational {
        Rational { num: n, den: 1 }
    }

    pub fn add(self, o: Rational) -> Rational {
        Rational::new(self.num * o.den + o.num * self.den, self.den * o.den)
    }

    pub fn sub(self, o: Rational) -> Rational {
        Rational::new(self.num * o.den - o.num * self.den, self.den * o.den)
    }

    pub fn div(self, o: Rational) -> Rational {
        Rational::new(self.num * o.den, self.den * o.num)
    }

    pub fn to_f64(self) -> f64 {
        self.num as f64 / self.den as f64
    }
}

fn gcd(a: i128, b: i128) -> i128 {
    if b == 0 {
        a
    } else {
        gcd(b, a % b)
    }
}

/// Trapezoid-rule weights for `n + 1` nodes (exact rationals).
pub(crate) fn trapezoid(n: i64) -> Vec<Rational> {
    let mut w = vec![Rational::from_int(1); (n + 1) as usize];
    w[0] = Rational::new(1, 2);
    w[n as usize] = Rational::new(1, 2);
    w
}

/// Laplace coefficient `Lambda_k` (`Lambda_0 = -1`, recursion
/// `Lambda_k = sum_{l<k} Lambda_l / (l - k - 1)`), blueprint `Λ()`.
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

/// Polynomial interpolation weights `Wi[i][j] = (M^{-1})_{ij}` with
/// `M_{ja} = j^a`, row-major `(k+1) * (k+1)` (blueprint Eq. 80).
pub fn calc_poly_interpolation(k: i64) -> Vec<f64> {
    let n = (k + 1) as usize;
    let mut m = DenseMat::zeros(n);
    for i in 0..n {
        for j in 0..n {
            // M[i][j] = i^j (Julia: i^j with 0^0 = 1)
            let v = (i as f64).powi(j as i32);
            m.set(i, j, C64::new(v, 0.0));
        }
    }
    let inv = crate::lu::mat_inv(&m);
    let mut out = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..n {
            out[i * n + j] = inv.get(i, j).re;
        }
    }
    out
}

/// Polynomial differentiation weights (blueprint Eq. 85), row-major
/// `(k+1) * (k+1)`, from the interpolation weights `wi`.
pub fn calc_poly_differentiation(k: i64, wi: &[f64]) -> Vec<f64> {
    let n = (k + 1) as usize;
    assert_eq!(wi.len(), n * n);
    let mut wd = vec![0.0f64; n * n];
    for j in 0..n {
        for i in 0..n {
            let mut acc = 0.0;
            for a in 1..n {
                acc += wi[a * n + j] * a as f64 * (i as f64).powi(a as i32 - 1);
            }
            wd[i * n + j] = acc;
        }
    }
    wd
}

/// Polynomial integration weights `Wt[m][n][l]` (blueprint Eq. 89),
/// row-major `(k+1)^3` indexed as `(m * (k+1) + n) * (k+1) + l`.
pub fn calc_poly_integration(k: i64, wi: &[f64]) -> Vec<f64> {
    let n = (k + 1) as usize;
    assert_eq!(wi.len(), n * n);
    let mut wt = vec![0.0f64; n * n * n];
    for l in 0..n {
        for j in 0..n {
            for i in 0..n {
                let mut acc = 0.0;
                for a in 0..n {
                    let af = a as f64;
                    acc += wi[a * n + l]
                        * ((j as f64).powi(a as i32 + 1) - (i as f64).powi(a as i32 + 1))
                        / (af + 1.0);
                }
                wt[(i * n + j) * n + l] = acc;
            }
        }
    }
    wt
}

/// Backward differentiation weights (blueprint Eq. 90):
/// `Wb[j] = -Wi[a=1][j]`, length `k + 1`.
pub fn calc_backward_differentiation(k: i64, wi: &[f64]) -> Vec<f64> {
    let n = (k + 1) as usize;
    assert_eq!(wi.len(), n * n);
    (0..n).map(|j| -wi[1 * n + j]).collect()
}

/// Gregory integration weights for `n > k` nodes (blueprint
/// `calc_gregory_weights`): trapezoid plus forward/backward `gamma`
/// corrections, exact rationals converted to `f64`, length `n + 1`.
pub fn calc_gregory_weights(k: i64, n: i64) -> Vec<f64> {
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
pub fn calc_gregory_integration(k: i64, wt: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
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

/// Analytic integral `R(m, a, b) = integral_0^m (m-x)^a x^b dx =
/// m^{a+b+1} a! b! / (a+b+1)!`, blueprint `𝐑()`.
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
pub fn calc_boundary_convolution(k: i64, wi: &[f64]) -> Vec<f64> {
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
