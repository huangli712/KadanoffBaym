// Project : KadanoffBaym
// File    : src/convolution/response.rs
// Purpose : response convolution (blueprint: convolution/response.jl)

use crate::complex::Complex;
use crate::types::ContourGreen;
use crate::types::step::GreenTimeStep;
use crate::types::function::TimeFunction;
use crate::basics::MATSUBARA;
use crate::weights::Integrator;

#[allow(dead_code)]
/// Evaluate `sum_i Omega_{n0, i} g(i)`, the integral of `g` from 0 to
/// `n0` in units of the grid spacing (blueprint `_response_integrate`).
fn response_integrate(n0: i64, integ: &Integrator, g: impl Fn(i64) -> Complex) -> Complex {
    let mut res = Complex::ZERO;
    if n0 > 0 {
        let n1 = n0.max(integ.k);
        for i in 0..=n1 {
            res += Complex::new(integ.giw.at(n0 as usize, i as usize), 0.0) * g(i);
        }
    }
    res
}

#[allow(dead_code)]
fn check_response(
    w: &ContourGreen,
    a1: i64,
    a2: i64,
    f: &TimeFunction,
    b1: i64,
    b2: i64,
    integ: &Integrator,
) {
    assert!(w.size1 == w.size2, "response: W must be square");
    assert!((1..=(w.size1 as i64)).contains(&a1) && (1..=(w.size1 as i64)).contains(&a2));
    assert!((1..=(f.size1 as i64)).contains(&b1) && (1..=(f.size1 as i64)).contains(&b2));
    assert!(f.size1 == f.size2);
    assert!(integ.k <= w.ntau as i64 - 1);
}

#[allow(dead_code)]
/// Calculate the response `c(t) = Int_C W(t,t') f(t')` at time step
/// `tstp` for the matrix elements `W[a1,a2]` and `f[b1,b2]` (blueprint
/// `response_convolution`, ℱ version). Indices are 0-based.
#[allow(clippy::too_many_arguments)]
pub(crate) fn response_convolution(
    tstp: i64,
    w: &ContourGreen,
    a1: usize,
    a2: usize,
    f: &TimeFunction,
    b1: usize,
    b2: usize,
    integ: &Integrator,
    beta: f64,
    h: f64,
) -> Complex {
    check_response(
        w,
        a1 as i64 + 1,
        a2 as i64 + 1,
        f,
        b1 as i64 + 1,
        b2 as i64 + 1,
        integ,
    );
    assert!(
        (0..=(w.nt as i64 + 1) - 1 + 1).contains(&tstp)
            || (MATSUBARA..=(w.nt as i64)).contains(&tstp)
    );
    assert!(f.nt as i64 >= tstp);
    if 1 <= tstp && tstp <= integ.k {
        assert!(w.nt as i64 >= integ.k + 1);
        assert!(f.nt as i64 >= integ.k + 1);
    }
    assert!(beta > 0.0);
    assert!(h > 0.0);
    let ntau = w.ntau as i64;
    let dtau = beta / ntau as f64;
    let f0 = f.get(MATSUBARA, b1, b2);
    if tstp == MATSUBARA {
        let res = response_integrate(ntau, integ, |i| w.mat.block(i as usize)[a1 * w.size2 + a2]);
        return res * f0 * Complex::new(dtau, 0.0);
    }
    let res = response_integrate(ntau, integ, |i| {
        w.lmix.raw_block_at(tstp, i)[a1 * w.size2 + a2]
    });
    let mut c = res * f0 * Complex::new(0.0, -dtau);
    let g = |i: i64| -> Complex {
        if i <= tstp {
            let wv = w.ret.raw_block_at(tstp, i)[a1 * w.size2 + a2];
            wv * f.get(i, b1, b2)
        } else {
            assert!(i <= w.nt as i64);
            let wv = w.ret.raw_block_at(i, tstp)[a2 * w.size2 + a1];
            -wv.conj() * f.get(i, b1, b2)
        }
    };
    c += response_integrate(tstp, integ, g) * Complex::new(h, 0.0);
    c
}

#[allow(dead_code)]
/// `response_convolution` with a timestep (𝒻) version of `W`: only
/// `MATSUBARA` or `tstp - 1 >= k` is allowed (blueprint 𝒻 version).
#[allow(clippy::too_many_arguments)]
pub(crate) fn response_convolution_step(
    tstp: i64,
    w: &GreenTimeStep,
    a1: usize,
    a2: usize,
    f: &TimeFunction,
    b1: usize,
    b2: usize,
    integ: &Integrator,
    beta: f64,
    h: f64,
) -> Complex {
    assert!(tstp == MATSUBARA || tstp - 1 >= integ.k);
    assert!(f.nt as i64 >= tstp);
    assert!(beta > 0.0);
    assert!(h > 0.0);
    let ntau = w.ntau as i64;
    let dtau = beta / ntau as f64;
    let f0 = f.get(MATSUBARA, b1, b2);
    if tstp == MATSUBARA {
        let res = response_integrate(ntau, integ, |i| w.mat.block(i as usize)[a1 * w.size2 + a2]);
        return res * f0 * Complex::new(dtau, 0.0);
    }
    let res = response_integrate(ntau, integ, |i| {
        w.lmix.raw_block_at(0, i)[a1 * w.size2 + a2]
    });
    let mut c = res * f0 * Complex::new(0.0, -dtau);
    let g = |i: i64| -> Complex {
        if i <= tstp {
            let wv = w.ret.raw_block_at(tstp, i)[a1 * w.size2 + a2];
            wv * f.get(i, b1, b2)
        } else {
            assert!(i <= w.tstp as i64);
            let wv = w.ret.raw_block_at(i, tstp)[a2 * w.size2 + a1];
            -wv.conj() * f.get(i, b1, b2)
        }
    };
    c += response_integrate(tstp, integ, g) * Complex::new(h, 0.0);
    c
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::contour::Contour;
    use crate::types::sign::Sign;

    #[test]
    fn constant_response_exact() {
        // W = 1 (all components), f = 1: c(t_n) = t_n + (-i)(beta) and
        // c_0 = beta
        let nt = 6;
        let ntau = 20;
        let (beta, h) = (2.0, 0.1);
        let c = Contour::new(nt, ntau, 1, 1, 1.0, beta);
        let mut w = ContourGreen::new(&c, Sign::Bose, Complex::ZERO);
        for m in 0..=ntau as i64 {
            w.set_mat(m, 0, 0, Complex::ONE);
            for n in 0..=nt as i64 {
                w.set_lmix(n, m, 0, 0, Complex::ONE);
            }
        }
        for n in 0..=nt as i64 {
            for j in 0..=n {
                w.set_ret(n, j, 0, 0, Complex::ONE);
            }
        }
        let mut f = TimeFunction::new(nt as usize, 1, 1, Complex::ZERO);
        for t in -1..=nt as i64 {
            f.set(t, 0, 0, Complex::ONE);
        }
        let integ = Integrator::new(3);
        // Matsubara branch: beta
        let c0 = response_convolution(MATSUBARA, &w, 0, 0, &f, 0, 0, &integ, beta, h);
        assert!((c0 - Complex::new(beta, 0.0)).abs() < 1e-12, "{c0}");
        // real branch (n > k): t_n - i beta
        let cr = response_convolution(5, &w, 0, 0, &f, 0, 0, &integ, beta, h);
        let want = Complex::new(5.0 * h, -beta);
        assert!((cr - want).abs() < 1e-12, "{cr} vs {want}");
    }
}
