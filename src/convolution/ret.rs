// Project : KadanoffBaym
// File    : src/convolution/ret.rs
// Purpose : retarded convolution kernels (blueprint: convolution/ret.jl)

use super::{mul_acc_c, ConvWorkspace, FOp};
use crate::complex::Complex;
use crate::types::green::{ContourGreen, GreenTimeStep, RetComp, TimeFunction};
use crate::weights::Integrator;

/// Accumulate `s * [A^R * f * B^R](t_n, t_m)` into the target's ret row
/// `(n, m)` for `m = lo..=hi` (blueprint `_conv_ret_kernel!`, NESSi
/// Eq. (110a)-(110c)). All block accesses stay inside the stored
/// triangles.
#[allow(clippy::too_many_arguments)]
pub(crate) fn conv_ret_kernel(
    tgt: &mut RetComp,
    n: i64,
    a: &RetComp,
    acc: &RetComp,
    f: FOp,
    b: &RetComp,
    bcc: &RetComp,
    integ: &Integrator,
    s: Complex,
    ws: &mut ConvWorkspace,
    lo: i64,
    hi: i64,
) {
        let k = integ.k;
    let s1 = a.size1;
    let s2a = a.size2;
    let s2b = b.size2;

    if n >= k {
        // NESSi Eq. (110a)
        for j in lo..=n {
            let aj = super::af(ws, a.raw_block_at(n, j), &f, j, s1, s2a);
            for m in lo..=j.min(hi) {
                let w = integ.giw.at((n - m) as usize, (n - j) as usize);
                let blk = tgt.block_mut(tgt.idx(n, m) as usize);
                mul_acc_c(blk, s.scale(w), aj, b.raw_block_at(j, m), s1, s2a, s2b);
            }
        }
        // NESSi Eq. (110b)
        for j in 1..=k {
            let mlo = (n - j + 1).max(lo);
            if mlo > hi {
                continue;
            }
            let aj = super::af(ws, a.raw_block_at(n, n - j), &f, n - j, s1, s2a);
            for m in mlo..=hi {
                let w = integ.giw.at((n - m) as usize, j as usize);
                let badj = super::adjoint_block(bcc.raw_block_at(m, n - j), bcc.size1, bcc.size2);
                let blk = tgt.block_mut(tgt.idx(n, m) as usize);
                mul_acc_c(blk, -s.scale(w), aj, &badj, s1, s2a, s2b);
            }
        }
    } else {
        // NESSi Eq. (110c)
        for j in 0..=k {
            if n >= j {
                let aj = super::af(ws, a.raw_block_at(n, j), &f, j, s1, s2a);
                conv_ret_start(tgt, n, j, aj, s, b, bcc, integ, lo, hi);
            } else {
                let aadj = super::adjoint_block(acc.raw_block_at(j, n), acc.size1, acc.size2);
                let aj = super::af(ws, &aadj, &f, j, s1, s2a);
                conv_ret_start(tgt, n, j, aj, -s, b, bcc, integ, lo, hi);
            }
        }
    }
}

/// The `j`-th term of NESSi Eq. (110c) (blueprint `_conv_ret_start!`).
#[allow(clippy::too_many_arguments)]
fn conv_ret_start(
    tgt: &mut RetComp,
    n: i64,
    j: i64,
    a: &[Complex],
    s: Complex,
    b: &RetComp,
    bcc: &RetComp,
    integ: &Integrator,
    lo: i64,
    hi: i64,
) {
    let s1 = b.size1;
    let s2a = a.len() / s1; // dims of a: (s1, s2a)
    let s2b = b.size2;
    for m in lo..=hi {
        let w = integ.xiw.at(m as usize, n as usize, j as usize);
        if j >= m {
            let blk = tgt.block_mut(tgt.idx(n, m) as usize);
            mul_acc_c(blk, s.scale(w), a, b.raw_block_at(j, m), s1, s2a, s2b);
        } else {
            let badj = super::adjoint_block(bcc.raw_block_at(m, j), bcc.size1, bcc.size2);
            let blk = tgt.block_mut(tgt.idx(n, m) as usize);
            mul_acc_c(blk, -s.scale(w), a, &badj, s1, s2a, s2b);
        }
    }
}

/// Evaluate `C^R(t_n, t_m) = [A^R * f * B^R](t_n, t_m)` for the whole
/// row `n`, overwriting old values (blueprint `_conv_ret!`).
#[allow(clippy::too_many_arguments)]
pub(crate) fn conv_ret_row(
    c_ret: &mut RetComp,
    n: i64,
    a: &RetComp,
    acc: &RetComp,
    f: FOp,
    b: &RetComp,
    bcc: &RetComp,
    integ: &Integrator,
    h: f64,
    ws: &mut ConvWorkspace,
) {
    for m in 0..=n {
        let blk = c_ret.block_mut(c_ret.idx(n, m) as usize);
        blk.fill(Complex::ZERO);
    }
    conv_ret_kernel(
        c_ret,
        n,
        a,
        acc,
        f,
        b,
        bcc,
        integ,
        Complex::new(h, 0.0),
        ws,
        0,
        n,
    );
}

#[allow(dead_code)]
/// Public `conv_ret` with an optional time-dependent `f_t` (a slice of
/// blocks indexed by Rust time, blueprint `conv_ret` with `fₜ`).
#[allow(clippy::too_many_arguments)]
pub(crate) fn conv_ret(
    n: i64,
    c: &mut ContourGreen,
    a: &ContourGreen,
    acc: &ContourGreen,
    f_t: Option<&[Vec<Complex>]>,
    b: &ContourGreen,
    bcc: &ContourGreen,
    integ: &Integrator,
    h: f64,
) {
    assert!((0..=(c.nt as i64)).contains(&n));
    assert!(h > 0.0);
    let mut ws = ConvWorkspace::new(a.size1, a.size2, b.size1, b.size2);
    let f = match f_t {
        Some(v) => FOp::T(v),
        None => FOp::None,
    };
    conv_ret_row(
        &mut c.ret, n, &a.ret, &acc.ret, f, &b.ret, &bcc.ret, integ, h, &mut ws,
    );
}

/// Increment the retarded row `tstp` of `C` by
/// `alpha * [A * f * B]^R` (blueprint `incr_convolution_ret!`).
#[allow(clippy::too_many_arguments)]
pub fn incr_convolution_ret(
    tstp: i64,
    c: &mut ContourGreen,
    a: &ContourGreen,
    acc: &ContourGreen,
    f: Option<&TimeFunction>,
    b: &ContourGreen,
    bcc: &ContourGreen,
    integ: &Integrator,
    h: f64,
    alpha: Complex,
) {
    assert!(tstp >= 0);
    assert!(h > 0.0);
    let mut ws = ConvWorkspace::new(a.size1, a.size2, b.size1, b.size2);
    let fop = match f {
        Some(cf) => FOp::Cf(cf),
        None => FOp::None,
    };
    conv_ret_kernel(
        &mut c.ret,
        tstp,
        &a.ret,
        &acc.ret,
        fop,
        &b.ret,
        &bcc.ret,
        integ,
        alpha * Complex::new(h, 0.0),
        &mut ws,
        0,
        tstp,
    );
}

/// `incr_convolution_ret!` with a timestep (𝒻) target.
#[allow(clippy::too_many_arguments)]
pub(crate) fn incr_convolution_ret_step(
    tstp: i64,
    c: &mut GreenTimeStep,
    a: &ContourGreen,
    acc: &ContourGreen,
    f: Option<&TimeFunction>,
    b: &ContourGreen,
    bcc: &ContourGreen,
    integ: &Integrator,
    h: f64,
    alpha: Complex,
) {
    assert!(tstp >= 0);
    assert!(h > 0.0);
    let mut ws = ConvWorkspace::new(a.size1, a.size2, b.size1, b.size2);
    let fop = match f {
        Some(cf) => FOp::Cf(cf),
        None => FOp::None,
    };
    conv_ret_kernel(
        &mut c.ret,
        tstp,
        &a.ret,
        &acc.ret,
        fop,
        &b.ret,
        &bcc.ret,
        integ,
        alpha * Complex::new(h, 0.0),
        &mut ws,
        0,
        tstp,
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::contour::Contour;
    use crate::types::sign::Sign;


    fn dagger_partner(g: &ContourGreen) -> ContourGreen {
        // G-dagger partner: (G‡)^R(i,j) = -G^R(i,j) in the stored
        // triangle (tilde anti-hermiticity)
        let mut acc = g.clone();
        for i in 0..=(g.nt as i64) {
            for j in 0..=i {
                let v = g.get_ret(i, j, 0, 0);
                acc.set_ret(i, j, 0, 0, -v);
            }
        }
        acc
    }

    #[test]
    fn constant_retarded_kernel_exact() {
        // A^R = B^R = 1 (constant): C^R(t_n, t_m) = Int_{t_m}^{t_n} = (n-m) h
        let nt = 8;
        let c0 = Contour::new(nt as i64, 4, 1, 1, 1.0, 1.0);
        let mut g = ContourGreen::new(&c0, Sign::Fermi, Complex::ZERO);
        for i in 0..=nt as i64 {
            for j in 0..=i {
                g.set_ret(i, j, 0, 0, Complex::ONE);
            }
        }
        let integ = Integrator::new(3);
        let h = 0.1;
        let acc = dagger_partner(&g);
        let mut ws = ConvWorkspace::new(1, 1, 1, 1);
        let mut cgreen = ContourGreen::new(&c0, Sign::Fermi, Complex::ZERO);
        for n in 1..=nt as i64 {
            conv_ret_row(
                &mut cgreen.ret,
                n,
                &g.ret,
                &acc.ret,
                FOp::None,
                &g.ret,
                &acc.ret,
                &integ,
                h,
                &mut ws,
            );
        }
        for n in 1..=nt as i64 {
            for m in 0..=n {
                let want = Complex::new((n - m) as f64 * h, 0.0);
                let got = cgreen.get_ret(n, m, 0, 0);
                assert!((got - want).abs() < 1e-13, "({n},{m}): {got} vs {want}");
            }
        }
    }

    #[test]
    fn start_up_matches_gregory() {
        // for n <= k both branches must agree with the n > k formula
        // continued on small grids: use k = 2, compare n = 1, 2 rows
        // against direct quadrature of constants (exact anyway)
        let nt = 6;
        let c0 = Contour::new(nt as i64, 4, 1, 1, 1.0, 1.0);
        let mut g = ContourGreen::new(&c0, Sign::Fermi, Complex::ZERO);
        for i in 0..=nt as i64 {
            for j in 0..=i {
                g.set_ret(i, j, 0, 0, Complex::ONE);
            }
        }
        let integ = Integrator::new(2);
        let h = 0.1;
        let acc = dagger_partner(&g);
        let mut ws = ConvWorkspace::new(1, 1, 1, 1);
        let mut cgreen = ContourGreen::new(&c0, Sign::Fermi, Complex::ZERO);
        for n in 1..=2 {
            conv_ret_row(
                &mut cgreen.ret,
                n,
                &g.ret,
                &acc.ret,
                FOp::None,
                &g.ret,
                &acc.ret,
                &integ,
                h,
                &mut ws,
            );
        }
        for n in 1..=2 {
            for m in 0..=n {
                let want = Complex::new((n - m) as f64 * h, 0.0);
                let got = cgreen.get_ret(n, m, 0, 0);
                assert!((got - want).abs() < 1e-13, "({n},{m}): {got} vs {want}");
            }
        }
    }

    #[test]
    fn with_f_t_weight() {
        // f(t) = 2 constant: C = 2 * (n-m) h
        let nt = 8;
        let c0 = Contour::new(nt as i64, 4, 1, 1, 1.0, 1.0);
        let mut g = ContourGreen::new(&c0, Sign::Fermi, Complex::ZERO);
        for i in 0..=nt as i64 {
            for j in 0..=i {
                g.set_ret(i, j, 0, 0, Complex::ONE);
            }
        }
        let fblocks: Vec<Vec<Complex>> = (0..=nt as i64 + 1)
            .map(|_| vec![Complex::new(2.0, 0.0)])
            .collect();
        let integ = Integrator::new(3);
        let h = 0.1;
        let mut cgreen = ContourGreen::new(&c0, Sign::Fermi, Complex::ZERO);
        let acc = dagger_partner(&g);
        conv_ret(5, &mut cgreen, &g, &acc, Some(&fblocks), &g, &acc, &integ, h);
        for m in 0..=5 {
            let want = Complex::new(2.0 * (5 - m) as f64 * h, 0.0);
            let got = cgreen.get_ret(5, m, 0, 0);
            assert!((got - want).abs() < 1e-13);
        }
    }
}
