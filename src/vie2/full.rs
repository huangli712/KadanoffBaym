// Project : KadanoffBaym
// File    : src/vie2/full.rs
// Purpose : Volterra integral equation of the second kind: argument
//           checks and the full driver (blueprint: solvers/vie2.jl)
//
// The VIE2 reads `[1 + F] * G = Q` with the hermitian conjugate
// partner `Fcc` of the kernel. The (unrepresentable) alias checks of
// the blueprint are omitted; Rust ownership makes them impossible.

use crate::dyson::mat::MatsubaraMethod;
use crate::types::ContourGreen;
use crate::weights::Integrator;

/// Check the compatibility of the arguments of the VIE2 solvers
/// (blueprint `_check_vie2`).
pub(crate) fn check_vie2(g: &ContourGreen, f: &ContourGreen, fcc: &ContourGreen, q: &ContourGreen) {
    assert!(
        g.sign == crate::types::sign::Sign::Fermi || g.sign == crate::types::sign::Sign::Bose,
        "unknown statistics of G"
    );
    for (x, name) in [(f, "F"), (fcc, "Fcc"), (q, "Q")] {
        assert!(x.sign == g.sign, "G and {name} must have the same statistics");
        assert!(
            x.size1 == g.size1 && x.size2 == g.size2,
            "G and {name} have different sizes"
        );
        assert!(
            x.ntau == g.ntau,
            "G and {name} have different imaginary time grids"
        );
        assert!(
            x.nt == g.nt,
            "G and {name} have different real time grids"
        );
    }
    assert!(g.size1 == g.size2, "G must be square");
}

/// Check the order of the integrator (blueprint `_check_vie2_order`).
pub(crate) fn check_vie2_order(integ: &Integrator) {
    assert!(
        (1..=5).contains(&integ.k),
        "the order of the integrator must be in 1..=5, got {}",
        integ.k
    );
}

/// Solve the VIE2 `[1 + F] * G = Q` for all the components of `g`:
/// Matsubara, start-up, and then all the remaining real time steps
/// (blueprint `vie2!`).
#[allow(clippy::too_many_arguments)]
pub fn vie2(
    g: &mut ContourGreen,
    f: &ContourGreen,
    fcc: &ContourGreen,
    q: &ContourGreen,
    integ: &Integrator,
    beta: f64,
    h: f64,
    method: MatsubaraMethod,
) {
    let k = integ.k;
    super::mat::vie2_mat(g, f, fcc, q, beta, integ, method, 6, 40, 1e-12);
    super::start::vie2_start_impl(g, f, fcc, q, integ, beta, h);
    for tstp in (k + 1)..=(g.nt as i64) {
        super::timestep::vie2_timestep_core(tstp, g, f, fcc, q, integ, beta, h);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::complex::Complex;
    use crate::convolution::convolution_ab;
    use crate::equilibrium::from_ham;
    use crate::linalg::mat::DenseMat;
    use crate::dyson::full::dyson_default;
    use crate::types::contour::Contour;
    use crate::types::sign::Sign;
    use crate::types::step::GreenTimeStep;
    use crate::types::function::TimeFunction;

    fn setup(nt: usize, ntau: usize) -> (Contour, TimeFunction, DenseMat, crate::types::ContourGreen) {
        let (beta, hstep, _mu) = (4.0f64, 0.05, 0.0);
        let c = Contour::new(nt as i64, ntau as i64, 2, 2, hstep * nt as f64, beta);
        let mut h = TimeFunction::new(nt, 2, 2, Complex::ZERO);
        let h0 = DenseMat::from_rows(&[
            vec![Complex::new(-0.4, 0.0), Complex::new(0.0, 0.1)],
            vec![Complex::new(0.0, -0.1), Complex::new(0.6, 0.0)],
        ]);
        for t in -1..=(nt as i64) {
            for i in 0..2 {
                for j in 0..2 {
                    h.set(t, i, j, h0.get(i, j));
                }
            }
        }
        // a smooth self-energy profile on the Matsubara axis
        let mut sigma = crate::types::ContourGreen::new(&c, Sign::Fermi, Complex::ZERO);
        for m in 0..=(ntau as i64) {
            let tau = m as f64 * beta / ntau as f64;
            let f = (-(tau * 0.8)).exp();
            sigma.set_mat(m, 0, 0, Complex::new(0.3 * f, 0.0));
            sigma.set_mat(m, 1, 1, Complex::new(0.21 * f, 0.0));
            sigma.set_mat(m, 0, 1, Complex::new(0.0, -0.15 * f));
            sigma.set_mat(m, 1, 0, Complex::new(0.0, 0.15 * f));
        }
        (c, h, h0, sigma)
    }

    /// F = 0: the VIE2 reduces to G = Q in all components.
    #[test]
    fn zero_kernel_returns_source() {
        let (nt, ntau, beta, hstep, k) = (15usize, 100, 4.0f64, 0.05, 5i64);
        let (c, _h, _h0, _sigma) = setup(nt, ntau);
        let f = crate::types::ContourGreen::new(&c, Sign::Fermi, Complex::ZERO);
        let fcc = crate::types::ContourGreen::new(&c, Sign::Fermi, Complex::ZERO);
        // a physical source: the free propagator of _h
        let mut _unused = crate::types::ContourGreen::new(&c, Sign::Fermi, Complex::ZERO);
        let (_c2, h2, h02, _s2) = setup(nt, ntau);
        let mut q = crate::types::ContourGreen::new(&c, Sign::Fermi, Complex::ZERO);
        from_ham(&mut q, 0.0, &h02, beta, hstep);
        let _ = (&_unused, &h2);

        let mut g = crate::types::ContourGreen::new(&c, Sign::Fermi, Complex::ZERO);
        let integ = Integrator::new(k);
        vie2(&mut g, &f, &fcc, &q, &integ, beta, hstep, MatsubaraMethod::Fixpoint);

        let mut e_mat = 0.0f64;
        for kk in 0..q.mat.raw().len() {
            e_mat = e_mat.max((g.mat.raw()[kk] - q.mat.raw()[kk]).abs());
        }
        assert!(e_mat < 1e-7, "mat {e_mat}");
        let mut e = 0.0f64;
        for i in 0..=(nt as i64) {
            for j in 0..=i {
                for kk in 0..4 {
                    e = e.max((g.ret.raw_block_at(i, j)[kk] - q.ret.raw_block_at(i, j)[kk]).abs());
                    e = e
                        .max((g.less.raw_block_at(j, i)[kk] - q.less.raw_block_at(j, i)[kk]).abs());
                }
            }
            for m in 0..=(ntau as i64) {
                for kk in 0..4 {
                    e = e
                        .max((g.lmix.raw_block_at(i, m)[kk] - q.lmix.raw_block_at(i, m)[kk]).abs());
                }
            }
        }
        assert!(e < 1e-8, "components {e}");
    }

    /// The Dyson equation is a VIE2 with `F = -G0 * Sigma`,
    /// `Fcc = -Sigma * G0`, and `Q = G0`: both solvers must agree.
    #[test]
    fn vie2_matches_dyson() {
        let (nt, ntau, beta, hstep, mu, k) = (15usize, 100, 4.0f64, 0.05, 0.0, 5i64);
        let (c, h, _h0, sigma) = setup(nt, ntau);

        // dyson reference
        let mut g1 = crate::types::ContourGreen::new(&c, Sign::Fermi, Complex::ZERO);
        dyson_default(&mut g1, mu, &h, &sigma, beta, hstep, k);

        // the equivalent VIE2
        let mut g0 = crate::types::ContourGreen::new(&c, Sign::Fermi, Complex::ZERO);
        from_ham(&mut g0, mu, &_h0, beta, hstep);
        let integ = Integrator::new(k);
        let mut ff = crate::types::ContourGreen::new(&c, Sign::Fermi, Complex::ZERO);
        convolution_ab(&mut ff, &g0, &sigma, &integ, beta, hstep);
        ff.smul(-Complex::ONE);
        let mut fcc = crate::types::ContourGreen::new(&c, Sign::Fermi, Complex::ZERO);
        convolution_ab(&mut fcc, &sigma, &g0, &integ, beta, hstep);
        fcc.smul(-Complex::ONE);

        let mut g2 = crate::types::ContourGreen::new(&c, Sign::Fermi, Complex::ZERO);
        vie2(&mut g2, &ff, &fcc, &g0, &integ, beta, hstep, MatsubaraMethod::Fixpoint);

        let mut e = 0.0f64;
        for kk in 0..g1.mat.raw().len() {
            e = e.max((g1.mat.raw()[kk] - g2.mat.raw()[kk]).abs());
        }
        for i in 0..=(nt as i64) {
            for j in 0..=i {
                for kk in 0..4 {
                    e = e.max((g1.ret.raw_block_at(i, j)[kk] - g2.ret.raw_block_at(i, j)[kk]).abs());
                    e = e.max(
                        (g1.less.raw_block_at(j, i)[kk] - g2.less.raw_block_at(j, i)[kk]).abs(),
                    );
                }
            }
            for m in 0..=(ntau as i64) {
                for kk in 0..4 {
                    e = e.max(
                        (g1.lmix.raw_block_at(i, m)[kk] - g2.lmix.raw_block_at(i, m)[kk]).abs(),
                    );
                }
            }
        }
        let mut er = 0.0f64;
        for i in 0..=(nt as i64) {
            for j in 0..=i {
                for kk in 0..4 {
                    er = er.max((g1.ret.raw_block_at(i, j)[kk] - g2.ret.raw_block_at(i, j)[kk]).abs());
                }
            }
        }
        let mut el = 0.0f64;
        for i in 0..=(nt as i64) {
            for m in 0..=(ntau as i64) {
                for kk in 0..4 {
                    el = el.max((g1.lmix.raw_block_at(i, m)[kk] - g2.lmix.raw_block_at(i, m)[kk]).abs());
                }
            }
        }
        let mut es = 0.0f64;
        for i in 0..=(nt as i64) {
            for j in 0..=i {
                for kk in 0..4 {
                    es = es.max((g1.less.raw_block_at(j, i)[kk] - g2.less.raw_block_at(j, i)[kk]).abs());
                }
            }
        }
        assert!(e < 5e-6, "dyson vs vie2 differ by {e}");
    }

    /// A vanishing singular part reduces `vie2_timestep_sin` to the
    /// ordinary VIE2.
    #[test]
    fn sin_with_zero_singular_parts_matches_plain() {
        use crate::vie2::sin::{vie2_timestep_sin, Vie2SinWorkspace};
        let (nt, ntau, beta, hstep, k) = (15usize, 100, 4.0f64, 0.05, 5i64);
        let (c, _h, h0, sigma) = setup(nt, ntau);
        let mut g0 = crate::types::ContourGreen::new(&c, Sign::Fermi, Complex::ZERO);
        from_ham(&mut g0, 0.0, &h0, beta, hstep);
        let integ = Integrator::new(k);
        let mut ff = crate::types::ContourGreen::new(&c, Sign::Fermi, Complex::ZERO);
        convolution_ab(&mut ff, &g0, &sigma, &integ, beta, hstep);
        ff.smul(-Complex::ONE);
        let mut fcc = crate::types::ContourGreen::new(&c, Sign::Fermi, Complex::ZERO);
        convolution_ab(&mut fcc, &sigma, &g0, &integ, beta, hstep);
        fcc.smul(-Complex::ONE);

        // plain reference
        let mut g1 = crate::types::ContourGreen::new(&c, Sign::Fermi, Complex::ZERO);
        vie2(&mut g1, &ff, &fcc, &g0, &integ, beta, hstep, MatsubaraMethod::Fixpoint);

        // stepwise sin version with zero singular parts
        let zero_tf = TimeFunction::new(nt, 2, 2, Complex::ZERO);
        let mut gsin = TimeFunction::new(nt, 2, 2, Complex::ZERO);
        let mut g2 = crate::types::ContourGreen::new(&c, Sign::Fermi, Complex::ZERO);
        let mut ws = Vie2SinWorkspace::new(&g2);
        for tstp in -1..=(nt as i64) {
            vie2_timestep_sin(
                tstp,
                &mut g2,
                &mut gsin,
                &ff,
                &fcc,
                &zero_tf,
                &g0,
                &zero_tf,
                &integ,
                beta,
                hstep,
                MatsubaraMethod::Fixpoint,
                &mut ws,
            );
        }

        let mut e = 0.0f64;
        for kk in 0..g1.mat.raw().len() {
            e = e.max((g1.mat.raw()[kk] - g2.mat.raw()[kk]).abs());
        }
        for i in 0..=(nt as i64) {
            for j in 0..=i {
                for kk in 0..4 {
                    e = e.max((g1.ret.raw_block_at(i, j)[kk] - g2.ret.raw_block_at(i, j)[kk]).abs());
                    e = e.max(
                        (g1.less.raw_block_at(j, i)[kk] - g2.less.raw_block_at(j, i)[kk]).abs(),
                    );
                }
            }
        }
        assert!(e < 1e-9, "sin vs plain differ by {e}");
        let _ = GreenTimeStep::new(&c, 0, Sign::Fermi, Complex::ZERO);
    }
}
