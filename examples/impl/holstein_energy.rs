//
// Phonon momentum and energies of the Holstein examples (the Rust port
// of julia/tutor/holstein_bethe/holstein_energy.jl, from NESSi
// examples/programs/Holstein_utils_impl.cpp).
//

#![allow(dead_code)]
// shared by several example binaries; not every binary uses every item

use crate::holstein_impl as hi;
use hi::HolsteinPhonon;
use kadanoffbaym::complex::Complex;
use kadanoffbaym::convolution::density_matrix::convolution_density_matrix;
use kadanoffbaym::convolution::timestep::convolution;
use kadanoffbaym::math::bose::bose;
use kadanoffbaym::types::contour::Contour;
use kadanoffbaym::types::sign::Sign;
use kadanoffbaym::types::green::ContourGreen;
use kadanoffbaym::types::layout::MATSUBARA;

/// Whole-object `dst += alpha * src` (blueprint `incr!(dst, src, alpha)`).
fn whole_add(dst: &mut ContourGreen, src: &ContourGreen, alpha: Complex) {
    for q in 0..dst.mat.raw().len() {
        dst.mat.raw_mut()[q] += alpha * src.mat.raw()[q];
    }
    for q in 0..dst.ret.raw().len() {
        dst.ret.raw_mut()[q] += alpha * src.ret.raw()[q];
    }
    for q in 0..dst.lmix.raw().len() {
        dst.lmix.raw_mut()[q] += alpha * src.lmix.raw()[q];
    }
    for q in 0..dst.less.raw().len() {
        dst.less.raw_mut()[q] += alpha * src.less.raw()[q];
    }
}

/// Phonon momentum `P(t) = int_0^t dD_0^R(t-t')/w0 [g n(t') - g n(0^-)]
/// dt'` at all KB time steps (NESSi `Hols::get_phonon_momentum`); `P[1]`
/// is zero on the Matsubara branch. The start-up steps use the integral
/// of the time step `k` with `g n(t') = g n(0^-)` for `t' < 0`.
pub fn phonon_momentum(ntot: &[Complex], g: &[Complex], ph: &HolsteinPhonon) -> Vec<Complex> {
    let nt_real = ph.d0.nt as i64;
    let (k, h, w0) = (ph.k, ph.h, ph.w0);
    // the arrays are indexed by the KB step number (0 = Matsubara)
    let ng = |m: usize| g[m] * ntot[m];
    let d1 = |m: i64| {
        let a = Complex::from_polar(1.0, -(m as f64) * h * w0);
        let b = Complex::from_polar(1.0, (m as f64) * h * w0);
        -(a + b)
    };
    let mut p = vec![Complex::ZERO; (nt_real + 2) as usize];
    // start-up steps: the code's tstp is my time step 0..k
    for tstp in 0..=k {
        let mut f = vec![Complex::ZERO; (k + 1) as usize];
        for it in 0..=k {
            // m is a real-time index (the value sits at the KB m + 1)
            let m = it + tstp - k;
            let ngm = if m >= 0 { ng(m as usize + 1) } else { ng(0) };
            f[it as usize] = d1(k - it) * (ngm - ng(0));
        }
        p[tstp as usize + 1] = hi::gregory_integrate(&ph.integ, &f, k) * Complex::new(h, 0.0);
    }
    // remaining steps
    for tstp in (k + 1)..=nt_real {
        let mut f = vec![Complex::ZERO; (tstp + 1) as usize];
        for it in 0..=tstp {
            f[it as usize] = d1(tstp - it) * (ng(it as usize + 1) - ng(0));
        }
        p[tstp as usize + 1] = hi::gregory_integrate(&ph.integ, &f, tstp) * Complex::new(h, 0.0);
    }
    p
}

/// `E_ph^cl = w0 (X^2 + P^2) / 4` (NESSi `evaluate_phonon_energy_cl`).
pub fn phonon_energy_classical(x: &[Complex], p: &[Complex], w0: f64) -> Vec<Complex> {
    (0..x.len())
        .map(|i| Complex::new(0.25, 0.0) * Complex::new(w0, 0.0) * (x[i] * x[i] + p[i] * p[i]))
        .collect()
}

/// Quantum phonon energy `w0 (<X^2> + <P^2>)/4` at all KB time steps
/// (NESSi `Hols::evaluate_phonon_energy_qu`), from the time derivatives
/// `D0^d1 = dD_0/dt / w0`, `D0^d2 = dD_0/dt' / w0` and the Dyson
/// equation of `D`.
pub fn phonon_energy_quantum(d: &ContourGreen, pi: &ContourGreen, ph: &HolsteinPhonon) -> Vec<Complex> {
    let (nt, ntau) = (d.nt as i64, d.ntau as i64);
    let (beta, h, w0) = (ph.beta, ph.h, ph.w0);
    let c = Contour::new(nt, ntau, 1, 1, nt as f64 * h, beta);
    let boson = || ContourGreen::new(&c, Sign::Bose, Complex::ZERO);
    let (mut d0_d1, mut d0_d2) = (boson(), boson());
    let (mut d_d1, mut d_d2) = (boson(), boson());
    let (mut d0_d1_pi, mut pi_d0_d2) = (boson(), boson());
    let dtau = beta / ntau as f64;
    let expp: Vec<Complex> = (0..=nt)
        .map(|it| Complex::from_polar(1.0, -(it as f64) * h * w0))
        .collect();
    let fb = bose(beta, w0);
    for m in 0..=ntau as i64 {
        let tau = m as f64 * dtau;
        let (c1, c2) = (
            Complex::new((-(tau * w0)).exp(), 0.0),
            Complex::new((tau * w0).exp(), 0.0),
        );
        let v = -(Complex::new(1.0 + fb, 0.0) * c1) + Complex::new(fb, 0.0) * c2;
        d0_d1.mat.block_mut(m as usize)[0] = v;
        d0_d2.mat.block_mut(m as usize)[0] = -v;
        for it in 0..=nt {
            let tv = -Complex::new(fb, 0.0) * expp[it as usize] * c2
                + Complex::new(1.0 + fb, 0.0) * expp[it as usize].conj() * c1;
            d0_d1.lmix.raw_block_at_mut(it, m)[0] = tv;
            d0_d2.lmix.raw_block_at_mut(it, m)[0] = -tv;
        }
    }
    for it1 in 0..=nt {
        for it2 in 0..=it1 {
            let e = expp[(it1 - it2) as usize];
            let r = -(e + e.conj());
            let l = -Complex::new(fb, 0.0) * e.conj() + Complex::new(1.0 + fb, 0.0) * e;
            d0_d1.ret.raw_block_at_mut(it1, it2)[0] = r;
            d0_d2.ret.raw_block_at_mut(it1, it2)[0] = -r;
            d0_d1.less.raw_block_at_mut(it2, it1)[0] = l;
            d0_d2.less.raw_block_at_mut(it2, it1)[0] = -l;
        }
    }
    convolution(&mut d0_d1_pi, &d0_d1, &d0_d2, None, pi, pi, &ph.integ, beta, h);
    convolution(&mut pi_d0_d2, pi, pi, None, &d0_d2, &d0_d1, &ph.integ, beta, h);
    convolution(&mut d_d1, &d0_d1_pi, &pi_d0_d2, None, d, d, &ph.integ, beta, h);
    whole_add(&mut d_d1, &d0_d1, Complex::ONE);
    convolution(&mut d_d2, d, d, None, &pi_d0_d2, &d0_d1_pi, &ph.integ, beta, h);
    whole_add(&mut d_d2, &d0_d2, Complex::ONE);
    let mut e = vec![Complex::ZERO; (nt + 2) as usize];
    for n in MATSUBARA + 1..=nt + 1 {
        let xx = if n == 0 {
            -d.mat.block(0)[0]
        } else {
            Complex::new(0.0, 1.0) * d.less.raw_block_at(n - 1, n - 1)[0]
        };
        let mut pp = -convolution_density_matrix(
            n - 1,
            &d0_d1_pi,
            &pi_d0_d2,
            None,
            &d_d2,
            &d_d1,
            &ph.integ,
            beta,
            h,
        )
        .get(0, 0);
        pp += Complex::new(1.0 + 2.0 * bose(beta, w0), 0.0);
        e[n as usize] = (xx + pp) * Complex::new(w0 / 4.0, 0.0);
    }
    e
}
