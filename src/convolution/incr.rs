// Project : KadanoffBaym
// File    : src/convolution/incr.rs
// Purpose : incremental convolution driver (blueprint: convolution/incr.jl)

use super::less::incr_convolution_les;
use super::lmix::incr_convolution_lmix;
use super::mat::incr_convolution_mat;
use super::ret::incr_convolution_ret;
use crate::complex::Complex;
use crate::types::ContourGreen;
use crate::types::step::GreenTimeStep;
use crate::types::function::TimeFunction;
use crate::basics::MATSUBARA;
use crate::weights::Integrator;

/// Increment `C` at time step `tstp` by `alpha * [A * f * B]` (blueprint
/// `incr_convolution!`). If `tstp == MATSUBARA`, only the Matsubara
/// component is changed. Otherwise the retarded, left-mixing, and lesser
/// components at `tstp` are changed.
#[allow(clippy::too_many_arguments)]
pub fn incr_convolution(
    tstp: i64,
    c: &mut ContourGreen,
    a: &ContourGreen,
    acc: &ContourGreen,
    f: Option<&TimeFunction>,
    b: &ContourGreen,
    bcc: &ContourGreen,
    integ: &Integrator,
    beta: f64,
    h: f64,
    alpha: Complex,
) {
    assert!(beta > 0.0);
    assert!(h > 0.0);
    if tstp == MATSUBARA {
        incr_convolution_mat(c, a, f, b, integ, beta, alpha);
    } else {
        assert!(tstp >= 0);
        incr_convolution_ret(tstp, c, a, acc, f, b, bcc, integ, h, alpha);
        incr_convolution_lmix(tstp, c, a, acc, f, b, bcc, integ, beta, h, alpha);
        incr_convolution_les(tstp, c, a, acc, f, b, bcc, integ, beta, h, alpha);
    }
}

/// `incr_convolution!` with a timestep (𝒻) target.
#[allow(clippy::too_many_arguments)]
pub fn incr_convolution_step(
    tstp: i64,
    c: &mut GreenTimeStep,
    a: &ContourGreen,
    acc: &ContourGreen,
    f: Option<&TimeFunction>,
    b: &ContourGreen,
    bcc: &ContourGreen,
    integ: &Integrator,
    beta: f64,
    h: f64,
    alpha: Complex,
) {
    assert!(beta > 0.0);
    assert!(h > 0.0);
    if tstp == MATSUBARA {
        super::mat::incr_convolution_mat_step(c, a, f, b, integ, beta, alpha);
    } else {
        assert!(tstp >= 0);
        super::ret::incr_convolution_ret_step(tstp, c, a, acc, f, b, bcc, integ, h, alpha);
        super::lmix::incr_convolution_lmix_step(tstp, c, a, acc, f, b, bcc, integ, beta, h, alpha);
        super::less::incr_convolution_les_step(tstp, c, a, acc, f, b, bcc, integ, beta, h, alpha);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::equilibrium::from_ham;
    use crate::linalg::mat::DenseMat;
    use crate::types::contour::Contour;
    use crate::types::sign::Sign;
    use crate::basics::indexing::timestep_at;

    fn free(nt: i64, ntau: i64, beta: f64, h: f64) -> (Contour, ContourGreen) {
        let c = Contour::new(nt, ntau, 1, 1, nt as f64 * h, beta);
        let mut g = ContourGreen::new(&c, Sign::Fermi, Complex::ZERO);
        from_ham(&mut g, 0.0, &DenseMat::zeros(1), beta, h);
        (c, g)
    }

    #[test]
    fn incr_equals_full_convolution() {
        // accumulating incr over all steps (on a zero target) must
        // reproduce the plain convolution for ret (overwritten rows are
        // zero anyway) and mat
        let (nt, ntau) = (6i64, 20i64);
        let (beta, h) = (2.0, 0.05);
        let (c, g) = free(nt, ntau, beta, h);
        let integ = Integrator::new(3);
        let mut ci = ContourGreen::new(&c, Sign::Fermi, Complex::ZERO);
        incr_convolution(
            MATSUBARA,
            &mut ci,
            &g,
            &g,
            None,
            &g,
            &g,
            &integ,
            beta,
            h,
            Complex::ONE,
        );
        for n in 1..=nt {
            incr_convolution(n, &mut ci, &g, &g, None, &g, &g, &integ, beta, h, Complex::ONE);
        }
        let mut cf = ContourGreen::new(&c, Sign::Fermi, Complex::ZERO);
        super::super::convolution_ab(&mut cf, &g, &g, &integ, beta, h);
        for m in 0..=ntau {
            let d = (ci.get_mat(m, 0, 0) - cf.get_mat(m, 0, 0)).abs();
            assert!(d < 1e-13, "mat {m}: {d}");
        }
        for n in 1..=nt {
            for m in 0..=n {
                let d = (ci.get_ret(n, m, 0, 0) - cf.get_ret(n, m, 0, 0)).abs();
                assert!(d < 1e-13, "ret ({n},{m}): {d}");
            }
            for m in 0..=ntau {
                let d = (ci.get_lmix(n, m, 0, 0) - cf.get_lmix(n, m, 0, 0)).abs();
                assert!(d < 1e-13, "lmix ({n},{m}): {d}");
            }
            for m in 0..=n {
                let d = (ci.get_less(m, n, 0, 0) - cf.get_less(m, n, 0, 0)).abs();
                assert!(d < 1e-13, "less ({m},{n}): {d}");
            }
        }
    }

    #[test]
    fn step_target_matches_green_target() {
        let (nt, ntau) = (5i64, 10i64);
        let (beta, h) = (1.0, 0.1);
        let (c, g) = free(nt, ntau, beta, h);
        let integ = Integrator::new(2);
        let mut ci = ContourGreen::new(&c, Sign::Fermi, Complex::ZERO);
        incr_convolution(3, &mut ci, &g, &g, None, &g, &g, &integ, beta, h, Complex::ONE);
        let zero = ContourGreen::new(&c, Sign::Fermi, Complex::ZERO);
        let mut f = timestep_at(&zero, 3);
        incr_convolution_step(3, &mut f, &g, &g, None, &g, &g, &integ, beta, h, Complex::ONE);
        // the timestep target accumulates the same increments
        for m in 0..=3 {
            let d = (f.get_ret(3, m, 0, 0) - ci.get_ret(3, m, 0, 0)).abs();
            assert!(d < 1e-13, "ret ({m}): {d}");
        }
    }
}
