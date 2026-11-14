// Project : KadanoffBaym
// File    : src/math/interp.rs
// Purpose : polynomial interpolation of a time function
//
// Blueprint: julia/src/math/interp.jl. The polynomial approximant through
// (x_l, f_l), l = 0..k, is sum_{p,l} x^p P_pl f_l with `x` measured in
// time-step units from the first grid point of the interval.

use crate::c64::C64;
use crate::mat::DenseMat;
use crate::types::green::TimeFunction;
use crate::weights::Integrator;

/// Polynomial interpolation of the time function `f` at the continuous
/// grid coordinate `tinter` of the interval `[tstp - k, tstp]`
/// (0-based; the physical time is `tinter * dt`).
///
/// Returns the interpolated matrix.
pub fn interpolation(
    tstp: i64,
    tinter: f64,
    f: &TimeFunction,
    integ: &Integrator,
) -> DenseMat {
    let k = integ.k;
    assert!(tstp >= k, "interpolation: tstp {tstp} >= k {k}");
    assert!(tstp <= f.nt as i64, "interpolation: tstp {tstp} <= nt {}", f.nt);
    assert!(
        (tstp - k) as f64 <= tinter && tinter <= tstp as f64,
        "interpolation: tinter {tinter} out of [{}, {tstp}]",
        tstp - k
    );
    let (s1, s2) = (f.size1, f.size2);
    let mut x = vec![C64::ZERO; s1 * s2];
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

    #[test]
    fn interpolates_polynomial_exactly() {
        // f(t) = 1 + 2t (degree 1): k = 3 masters it exactly
        let nt = 6usize;
        let mut f = TimeFunction::new(nt, 1, 1, C64::ZERO);
        let dt = 0.25;
        for t in 0..=nt as i64 {
            let v = 1.0 + 2.0 * t as f64 * dt;
            f.set(t, 0, 0, C64::new(v, 0.0));
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
        let mut f = TimeFunction::new(nt, 1, 1, C64::ZERO);
        let h = 0.1f64;
        for t in 0..=nt as i64 {
            f.set(t, 0, 0, C64::new((h * t as f64).cos(), 0.0));
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
        let f = TimeFunction::new(6, 1, 1, C64::ZERO);
        let integ = Integrator::new(3);
        let _ = interpolation(6, 4.0, &f, &integ); // below tstp - k = 3? 4.0 >= 3 -> ok
        let _ = interpolation(6, 2.5, &f, &integ);
    }
}
