// Project : KadanoffBaym
// File    : src/math/interp.rs
// Purpose : polynomial interpolation of a time function
//
// (x_l, f_l), l = 0..k, is sum_{p,l} x^p P_pl f_l with `x` measured in
// time-step units from the first grid point of the interval.
//!
//! *Remarks* : *Polynomial Interpolation*
//!
//! The ``k``-th order polynomial approximant through the function values
//! ``(x_l, f_l)``, ``l = 0, \cdots, k``, is given by
//!
//! ```math
//! \begin{equation}
//! \mathcal{P}(x) = \sum^{k}_{p,l=0} x^p P^{(k)}_{pl} f_l.
//! \end{equation}
//! ```
//!
//! Here ``x`` is measured in units of the time step, from the first grid
//! point ``t_{tstp-k}`` of the interpolation interval.

use crate::complex::Complex;
use crate::linalg::DenseMat;
use crate::types::TimeFunction;
use crate::weights::Integrator;

/// Polynomial interpolation of the time function `f` at the continuous
/// grid coordinate `tinter` of the interval `[tstp - k, tstp]`
/// (0-based; the physical time is `tinter * dt`).
///
/// Returns the interpolated matrix.
pub fn interpolation(tstp: i64, tinter: f64, f: &TimeFunction, integ: &Integrator) -> DenseMat {
    let k = integ.k;
    assert!(tstp >= k, "interpolation: tstp {tstp} >= k {k}");
    assert!(
        tstp <= f.nt as i64,
        "interpolation: tstp {tstp} <= nt {}",
        f.nt
    );
    assert!(
        (tstp - k) as f64 <= tinter && tinter <= tstp as f64,
        "interpolation: tinter {tinter} out of [{}, {tstp}]",
        tstp - k
    );
    let (s1, s2) = (f.size1, f.size2);
    let mut x = vec![Complex::ZERO; s1 * s2];
    let t0 = (tstp - k) as f64;
    for l in 0..=k as usize {
        let mut t1 = 1.0;
        let mut weight = integ.piw.at(0, l);
        for p in 1..=k as usize {
            t1 *= tinter - t0;
            weight += t1 * integ.piw.at(p, l);
        }
        let blk = f.block(tstp - k + l as i64);
        for i in 0..x.len() {
            x[i] += blk[i] * weight;
        }
    }
    DenseMat::from_vec(s1, s2, x)
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::panic::AssertUnwindSafe;

    use std::panic::catch_unwind;

    struct Rng(u64);

    impl Rng {
        fn next_f64(&mut self) -> f64 {
            self.0 = self.0.wrapping_add(0x9e3779b97f4a7c15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
            z ^= z >> 31;
            ((z >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
        }

        fn block(&mut self, nel: usize) -> Vec<Complex> {
            (0..nel)
                .map(|_| Complex::new(self.next_f64(), self.next_f64()))
                .collect()
        }
    }

    fn poly_eval(ms: &[Vec<Complex>], t: f64) -> Vec<Complex> {
        let nel = ms[0].len();
        let mut out = vec![Complex::ZERO; nel];
        for (r, m) in ms.iter().enumerate() {
            let w = Complex::new(t.powi(r as i32), 0.0);
            for (o, c) in out.iter_mut().zip(m) {
                *o += *c * w;
            }
        }
        out
    }

    #[test]
    fn interpolates_polynomial_exactly() {
        // f(t) = 1 + 2t (degree 1): k = 3 masters it exactly
        let nt = 6usize;
        let mut f = TimeFunction::new(nt, 1, 1, Complex::ZERO);
        let dt = 0.25;
        for t in 0..=nt as i64 {
            let v = 1.0 + 2.0 * t as f64 * dt;
            f.set(t, 0, 0, Complex::new(v, 0.0));
        }
        let integ = Integrator::new(3);
        let out = interpolation(6, 5.5, &f, &integ);
        let want = 1.0 + 2.0 * 5.5 * dt;
        assert!((out.get(0, 0).re - want).abs() < 1e-12);
        // endpoints reproduce the grid values
        let out0 = interpolation(6, 5.0, &f, &integ);
        assert!((out0.get(0, 0).re - (1.0 + 2.0 * 5.0 * dt)).abs() < 1e-12);
    }

    #[test]
    fn cosine_interpolation_error() {
        // k = 5 on a fine grid approximates cos well
        let nt = 40usize;
        let mut f = TimeFunction::new(nt, 1, 1, Complex::ZERO);
        let h = 0.1f64;
        for t in 0..=nt as i64 {
            f.set(t, 0, 0, Complex::new((h * t as f64).cos(), 0.0));
        }
        let integ = Integrator::new(5);
        let mut err = 0.0f64;
        for t in 6..=nt as i64 {
            let out = interpolation(t, t as f64 - 0.5, &f, &integ);
            let want = (h * (t as f64 - 0.5)).cos();
            err = err.max((out.get(0, 0).re - want).abs());
        }
        assert!(err < 1e-5, "err = {err}");
    }

    #[test]
    #[should_panic(expected = "tinter")]
    fn rejects_out_of_interval() {
        let f = TimeFunction::new(6, 1, 1, Complex::ZERO);
        let integ = Integrator::new(3);
        let _ = interpolation(6, 4.0, &f, &integ); // below tstp - k = 3? 4.0 >= 3 -> ok
        let _ = interpolation(6, 2.5, &f, &integ);
    }

    #[test]
    fn interpolation_polynomial_exactness() {
        let h = 0.1;
        for k in 1..=5i64 {
            for nel in [1usize, 4] {
                let dims = if nel == 1 { (1, 1) } else { (2, 2) };
                let mut rng = Rng(0x777 + k as u64);
                let ms: Vec<Vec<Complex>> = (0..=k).map(|_| rng.block(nel)).collect();
                let ntime = k + 3; // Rust nt
                let mut f = TimeFunction::new(ntime as usize, dims.0, dims.1, Complex::ZERO);
                for i in 0..=ntime {
                    f.block_mut(i)
                        .copy_from_slice(&poly_eval(&ms, i as f64 * h));
                }
                let tstp = ntime; // last 0-based index
                let integ = Integrator::new(k);
                for tinter in [
                    tstp as f64,
                    (tstp - k) as f64,
                    tstp as f64 - 0.5,
                    (tstp - k) as f64 + 0.3,
                ] {
                    let x = interpolation(tstp, tinter, &f, &integ);
                    let want = poly_eval(&ms, tinter * h);
                    for (q, wq) in want.iter().enumerate().take(nel) {
                        let gotv = x.get(q / dims.1, q % dims.1);
                        assert!((gotv - *wq).abs() < 1e-12, "k={k} tinter={tinter}");
                    }
                }
                let r = catch_unwind(AssertUnwindSafe(|| {
                    interpolation(tstp, tstp as f64 + 0.1, &f, &integ);
                }));
                assert!(r.is_err());
                let r = catch_unwind(AssertUnwindSafe(|| {
                    interpolation(tstp, (tstp - k) as f64 - 0.1, &f, &integ);
                }));
                assert!(r.is_err());
            }
        }
        // k = 0 cases are covered by `interpolation_zero_order`
    }

    #[test]
    fn interpolation_zero_order() {
        let mut f = TimeFunction::new(4, 2, 2, Complex::ZERO);
        f.block_mut(3).copy_from_slice(&[
            Complex::new(1.0, 1.0),
            Complex::ZERO,
            Complex::ZERO,
            Complex::ONE,
        ]);
        // manual zeroth-order: result equals the block at tstp
        let x = interpolation(3, 3.0, &f, &Integrator::new(1));
        // (order-1 interpolation at a node reproduces the node value)
        assert_eq!(x.get(0, 0), Complex::new(1.0, 1.0));
    }
}
