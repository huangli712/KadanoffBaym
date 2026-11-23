//
// Project : KadanoffBaym
// File    : src/base/seed.rs
// Purpose : frequency-domain Matsubara seed solver shared by the
//           Dyson and VIE2 families
//

use std::f64::consts::PI;

use crate::complex::Complex;
use crate::fourier::{Corr, MatsubaraTables};
use crate::linalg::DenseMat;
use crate::types::{MatComp, Sign};

use crate::fourier::first_order_tail;
use crate::linalg::{lu_factor_into, lu_solve_mat_into, mul_acc};

/// The seed solver scratch of the fixpoint and conjugate-gradient
/// Matsubara Dyson/VIE2 solvers (internal): the kernel `F` is fixed across
/// the iterations, so its raw transform and corrected frequency values
/// are computed once and reused (`dyson/mat.rs`).
pub(crate) struct MatsubaraSeed {
    d: usize,
    nel: usize,
    beta: f64,
    sign: Sign,
    tables: MatsubaraTables,
    mps: Vec<i64>,
    corrs: Vec<Corr>,
    f_ft: Vec<Vec<Complex>>,
    b_dft: Vec<Vec<Complex>>,
    b_ft: Vec<Vec<Complex>>,
    xmat: Vec<Vec<Complex>>,
    qmasy: Vec<Complex>,
    /// Cached LU factorizations of `[1 + F(i w_mp)]` for every
    /// frequency of `mps` (the kernel `F` is fixed for the lifetime of
    /// the seed, so the factors are computed once in `new`).
    lu: Vec<(DenseMat, Vec<usize>)>,
    z3m: DenseMat,
    sol: DenseMat,
    xout: DenseMat,
    fz3: DenseMat,
    col: Vec<Complex>,
}

impl MatsubaraSeed {
    /// Prepare the fixed kernel `f`.
    pub(crate) fn new(f: &MatComp, beta: f64, sign: Sign, pcf: i64, order: i64) -> MatsubaraSeed {
        let d = f.size1;
        assert_eq!(d, f.size2, "square matrices are expected");
        let n = f.ntau as i64;
        assert!(
            n % 2 == 0 && n >= 4,
            "the number of imaginary time intervals must be even and >= 4, got {n}"
        );
        assert!(pcf >= 0, "pcf must be nonnegative");
        assert!(beta > 0.0, "beta must be positive");
        let tables = MatsubaraTables::new(n, sign, order);
        // the (p, m) window enumeration: the corrections depend on the
        // actual frequency, so the lists are aligned with this order
        let mut mps = Vec::new();
        for p in -pcf..=pcf {
            let (mut m_low, mut m_high) = (0i64, 0i64);
            if sign == Sign::Bose {
                m_high = if p < 0 { 0 } else { 1 };
                m_low = if p > 0 { -1 } else { 0 };
            }
            let m2 = n / 2;
            for m in (-m2 - m_low)..=(m2 - 1 + m_high) {
                mps.push(m + p * n);
            }
        }
        let corrs: Vec<Corr> = mps.iter().map(|&mp| tables.corr_of_mp(mp)).collect();
        let mut f_dft = Vec::new();
        tables.dft_into(f, &mut f_dft);
        let mut f_ft = Vec::new();
        tables.ft_list(f, &f_dft, beta, &mps, &corrs, &mut f_ft);
        drop(f_dft);
        let mut seed = MatsubaraSeed {
            d,
            nel: d * d,
            beta,
            sign,
            tables,
            mps,
            corrs,
            f_ft,
            b_dft: Vec::new(),
            b_ft: Vec::new(),
            xmat: vec![vec![Complex::ZERO; d * d]; (n + 1) as usize],
            qmasy: vec![Complex::ZERO; d * d],
            lu: Vec::new(),
            z3m: DenseMat::zeros(d),
            sol: DenseMat::zeros(d),
            xout: DenseMat::zeros(d),
            fz3: DenseMat::zeros(d),
            col: Vec::new(),
        };
        // factor [1 + F(i w_mp)] once: the kernel is fixed across the
        // solve iterations, only the back substitution repeats
        for fiwo in &seed.f_ft {
            let mut z1 = DenseMat::identity(d);
            for i in 0..d {
                for j in 0..d {
                    let v = z1.get(i, j) + fiwo[i * d + j];
                    z1.set(i, j, v);
                }
            }
            let mut z1f = z1;
            let mut piv = Vec::new();
            lu_factor_into(&mut z1f, &mut piv);
            seed.lu.push((z1f, piv));
        }
        seed
    }

    /// Solve `[1 + F] X = B` for the source `b` (the per-iteration part
    /// of `_matsubara_seed!`).
    pub(crate) fn solve(&mut self, x: &mut MatComp, b: &MatComp) {
        assert!(x.is_compatible(b), "X and B are not compatible");
        let n = self.tables.n;
        let d = self.d;
        let nel = self.nel;
        let beta = self.beta;
        let sign = self.sign;
        let one = if sign == Sign::Fermi { 1i64 } else { 0 };
        // first order tail correction, q_inf = xi B(beta) - B(0)
        let qmasy = &mut self.qmasy;
        for (q, v) in qmasy.iter_mut().enumerate() {
            *v = Complex::new(sgn(sign), 0.0) * b.block(n as usize)[q] - b.block(0)[q];
        }
        first_order_tail(&mut self.xmat, qmasy, beta, sign);

        let mut b_dft = std::mem::take(&mut self.b_dft);
        let mut b_ft = std::mem::take(&mut self.b_ft);
        self.tables.dft_into(b, &mut b_dft);
        self.tables
            .ft_list(b, &b_dft, beta, &self.mps, &self.corrs, &mut b_ft);

        for (imp, &mp) in self.mps.iter().enumerate() {
            {
                let omega = (2 * mp + one) as f64 * PI / beta;

                // tail term in frequency space, q_inf / (i omega)
                let z3_is_zero = sign == Sign::Bose && mp == 0;
                if !z3_is_zero {
                    let f = Complex::new(0.0, -1.0 / omega);
                    for (z, &qm) in self.z3m.raw_mut().iter_mut().zip(qmasy.iter()) {
                        *z = f * qm;
                    }
                } else {
                    for q in 0..nel {
                        self.z3m.raw_mut()[q] = Complex::ZERO;
                    }
                }

                // [1 + F]^{-1} [B - (1 + F) q_inf / (i omega)] / beta
                let fiwo = &self.f_ft[imp];
                let qiwo = &b_ft[imp];
                let fz3 = &mut self.fz3;
                fz3.raw_mut().fill(Complex::ZERO);
                mul_acc(fz3.raw_mut(), 1.0, fiwo, self.z3m.raw(), d, d, d);
                let sol = &mut self.sol;
                let z3m = &self.z3m;
                for (q, s) in sol.raw_mut().iter_mut().enumerate() {
                    *s = qiwo[q] - z3m.raw()[q] - fz3.raw()[q];
                }
                let (z1f, piv) = &self.lu[imp];
                let mut col = std::mem::take(&mut self.col);
                lu_solve_mat_into(z1f, piv, sol, &mut self.xout, &mut col);
                self.col = col;
                for q in self.xout.raw_mut().iter_mut() {
                    *q /= Complex::new(beta, 0.0);
                }

                // accumulation: e^{-i omega tau_r} = conj(tw[k]),
                // k = (2 mp + one) r mod 2n
                let t = (2 * mp + one).rem_euclid(2 * n) as usize;
                let tw = &self.tables.tw;
                for (r, blk) in self.xmat.iter_mut().enumerate() {
                    let k = (t * r) % (2 * n as usize);
                    let w = tw[k].conj();
                    for (z, &sv) in blk.iter_mut().zip(self.xout.raw()) {
                        *z += w * sv;
                    }
                }
            }
        }

        for (dst, src) in x.raw_mut().chunks_mut(nel).zip(&self.xmat) {
            dst.copy_from_slice(src);
        }
        assert!(
            x.raw().iter().all(|z| z.re.is_finite() && z.im.is_finite()),
            "Matsubara seed is not finite"
        );
        self.b_dft = b_dft;
        self.b_ft = b_ft;
    }
}

#[inline]
fn sgn(sign: Sign) -> f64 {
    sign.as_i64() as f64
}

/// Solve ``[1 + F] * X = B`` for Matsubara functions in the frequency
/// domain, see *Remarks* : *Matsubara Seed*. The number of imaginary time
/// intervals must be even and at least 4; `pcf` is the number of the
/// oversampling copies on each side.
///
/// ### Arguments
/// * x     -> Matsubara function, the solution.
/// * f     -> Matsubara function, the kernel.
/// * b     -> Matsubara source term.
/// * beta  -> Inverse temperature.
/// * sign  -> FERMI or BOSE.
/// * pcf   -> Number of the oversampling copies on each side.
/// * order -> Order of the boundary correction.
///
/// ### Returns
/// * x -> The solution.
pub fn matsubara_seed(
    x: &mut MatComp,
    f: &MatComp,
    b: &MatComp,
    beta: f64,
    sign: Sign,
    pcf: i64,
    order: i64,
) {
    assert!(x.is_compatible(f), "X and F are not compatible");
    assert!(x.is_compatible(b), "X and B are not compatible");
    let mut solver = MatsubaraSeed::new(f, beta, sign, pcf, order);
    solver.solve(x, b);
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::complex::Complex;
    use crate::types::{Contour, ContourGreen, Sign};

    #[test]
    fn seed_identity_kernel_returns_source() {
        // F = 0: the seed must reproduce B
        let (beta, n, pcf) = (4.0f64, 16i64, 1);
        let c = Contour::new(1, n, 1, 1, 1.0, beta);
        let mut b = ContourGreen::new(&c, Sign::Fermi, Complex::ZERO);
        for r in 0..=n {
            let tau = r as f64 * beta / n as f64;
            b.set_mat(
                r,
                0,
                0,
                Complex::new((-(0.5 * tau)).exp() * (0.3 * tau).cos(), 0.0)
                    + Complex::new(0.0, 0.11 * tau),
            );
        }
        let f = ContourGreen::new(&c, Sign::Fermi, Complex::ZERO);
        let mut x = ContourGreen::new(&c, Sign::Fermi, Complex::ZERO);
        matsubara_seed(&mut x.mat, &f.mat, &b.mat, beta, Sign::Fermi, pcf, 3);
        let mut err = 0.0f64;
        for r in 0..=n {
            err = err.max((x.get_mat(r, 0, 0) - b.get_mat(r, 0, 0)).abs());
        }
        // the frequency window (N = 16 frequencies) truncates the 1/w^2
        // tails of the non-periodic source; ~5e-3 is the expected floor
        assert!(err < 8e-3, "seed differs from the source by {err}");
    }
}
