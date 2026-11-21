// Project : KadanoffBaym
// File    : src/convolution/full.rs
// Purpose : convolution entry points: the full contour drivers and
//           the single time step driver

use crate::complex::Complex;
use crate::convolution::{ConvWorkspace, FOp};
use crate::types::{ContourGreen, TimeFunction};
use crate::weights::Integrator;

use crate::types::MATSUBARA;

use crate::convolution::incr_less_diagcc;
use crate::convolution::mat::{cf_mat0, conv_mat, conv_mat_f0};
use crate::convolution::{conv_less_adv_kernel, conv_lmix_rmix_kernel, conv_ret_less_kernel};
use crate::convolution::{conv_lmix_mat_loop, conv_ret_lmix_kernel, conv_ret_row};
use crate::linalg::mul_acc;

/// Evaluate the retarded, left-mixing, and lesser components of
/// `C = A * f * B` at time step `n`.
/// The retarded component is OVERWRITTEN, while the left-mixing and
/// lesser components are ACCUMULATED (the Matsubara component is not
/// touched).
#[allow(clippy::too_many_arguments)]
fn convolution_time_step_inner(
    n: i64,
    c: &mut ContourGreen,
    a: &ContourGreen,
    acc: &ContourGreen,
    f: Option<&TimeFunction>,
    b: &ContourGreen,
    bcc: &ContourGreen,
    integ: &Integrator,
    beta: f64,
    h: f64,
    ws: &mut ConvWorkspace,
) {
    let sign = a.sign;
    let dtau = beta / a.ntau as f64;
    let fop = match f {
        Some(cf) => FOp::Cf(cf),
        None => FOp::None,
    };

    // C^R = A^R * f * B^R (overwrites the row)
    conv_ret_row(
        &mut c.ret, n, &a.ret, &acc.ret, fop, &b.ret, &bcc.ret, integ, h, ws,
    );

    // C^⌉ = A^R * f * B^⌉ + A^⌉ * f_0 * B^M (accumulates into the row)
    conv_ret_lmix_kernel(
        &mut c.lmix,
        n,
        n,
        &a.ret,
        &acc.ret,
        fop,
        &b.lmix,
        integ,
        Complex::new(h, 0.0),
        ws,
        (0, a.ntau as i64),
    );
    // f_0 * B^M(tau_m) is fixed for the workspace lifetime: build it
    // once and reuse across the time-step loop
    if ws.fbm.is_empty() {
        let nelb = a.size1 * b.size2;
        match f {
            Some(cf) => {
                let f0 = cf.block(MATSUBARA);
                let kdim = f0.len() / a.size2;
                ws.fbm = vec![Complex::ZERO; (a.ntau + 1) * nelb];
                for m in 0..=(a.ntau as i64) {
                    mul_acc(
                        &mut ws.fbm[m as usize * nelb..(m as usize + 1) * nelb],
                        1.0,
                        f0,
                        b.mat.block(m as usize),
                        a.size2,
                        kdim,
                        b.size2,
                    );
                }
            }
            None => ws.fbm = b.mat.raw().to_vec(),
        }
    }
    let bt = &ws.fbm;
    let rowlen = (a.ntau + 1) * a.size1 * b.size2;
    let off = n as usize * rowlen;
    let row = &mut c.lmix.raw_mut()[off..off + rowlen];
    conv_lmix_mat_loop(
        row,
        n,
        &a.lmix,
        bt,
        integ,
        sign,
        Complex::new(dtau, 0.0),
        (0, a.ntau as i64),
    );

    // C^< = A^R * f * B^< + A^< * f * B^A + A^⌉ * f_0 * B^⌈ (accumulates
    // into the column)
    let diagcc = incr_less_diagcc(n, integ);
    conv_ret_less_kernel(
        &mut c.less,
        (0, n),
        n,
        &a.ret,
        &acc.ret,
        fop,
        &b.less,
        &bcc.less,
        integ,
        Complex::new(h, 0.0),
        ws,
    );
    conv_less_adv_kernel(
        &mut c.less,
        (0, n),
        n,
        &a.less,
        &acc.less,
        fop,
        &b.ret,
        &bcc.ret,
        integ,
        Complex::new(h, 0.0),
        ws,
        diagcc,
    );
    let f0 = f.map(|cf| cf.block(MATSUBARA));
    conv_lmix_rmix_kernel(
        &mut c.less,
        (0, n),
        n,
        &a.lmix,
        f0,
        &bcc.lmix,
        integ,
        sign,
        Complex::new(0.0, -dtau),
        ws,
    );
}

/// Public `convolution_time_step`: `C(t,t') = A * f * B` at time step
/// `t = nh` for all components except the Matsubara one. Builds a
/// fresh workspace; use [`convolution_time_step_ws`] in time-step
/// loops.
#[allow(clippy::too_many_arguments)]
pub fn convolution_time_step(
    n: i64,
    c: &mut ContourGreen,
    a: &ContourGreen,
    acc: &ContourGreen,
    f: Option<&TimeFunction>,
    b: &ContourGreen,
    bcc: &ContourGreen,
    integ: &Integrator,
    beta: f64,
    h: f64,
) {
    assert!((0..=(c.nt as i64)).contains(&n));
    assert!(beta > 0.0);
    assert!(h > 0.0);
    let mut ws = ConvWorkspace::new(a.size1, a.size2, b.size1, b.size2);
    convolution_time_step_inner(n, c, a, acc, f, b, bcc, integ, beta, h, &mut ws);
}

/// Workspace-reusing variant of [`convolution_time_step`]: pass one
/// workspace built before the time-step loop. The `f_0 * B^M` blocks
/// cached in `ws` stay valid only while `f` and the Matsubara blocks
/// of `b` are unchanged; call `ws.reset()` when the inputs change.
#[allow(clippy::too_many_arguments)]
pub fn convolution_time_step_ws(
    n: i64,
    c: &mut ContourGreen,
    a: &ContourGreen,
    acc: &ContourGreen,
    f: Option<&TimeFunction>,
    b: &ContourGreen,
    bcc: &ContourGreen,
    integ: &Integrator,
    beta: f64,
    h: f64,
    ws: &mut ConvWorkspace,
) {
    assert!((0..=(c.nt as i64)).contains(&n));
    assert!(beta > 0.0);
    assert!(h > 0.0);
    convolution_time_step_inner(n, c, a, acc, f, b, bcc, integ, beta, h, ws);
}

/// Full-contour convolution `C = A * B` (Type 1) or `C = A * f * B`
/// (Type 2) over ALL time steps plus the Matsubara component. The
/// Matsubara and retarded components are overwritten; the lmix rows and
/// less columns are accumulated in step order, so a fresh (zeroed) `C`
/// yields the plain convolution.
///
/// *Remarks* : *Convolution*
///
/// **Convolution Type 1 : C = A ∗ B**
///
/// The convolution of two correlators ``A(t,t')`` and ``B(t,t')`` reads
///
/// ```math
/// \begin{equation}
/// C(t,t') = [A \ast B](t,t')
///         = \int_{\mathcal{C}} d\bar{t} \
///           A(t,\bar{t}) B(\bar{t},t').
/// \end{equation}
/// ```
///
/// It is one of the most basic operations on the contour ``\mathcal{C}``.
///
/// **Convolution Type 2 : C = A ∗ f ∗ B**
///
/// The most general convolution of two contour-ordered Green's functions `A`
/// and `B` and a time-dependent function `f` is given by the integral:
///
/// ```math
/// \begin{equation}
/// C(t,t') = [A \ast f \ast B](t,t')
///         = \int_{\mathcal{C}} d\bar{t} \
///           A(t,\bar{t}) f(\bar{t}) B(\bar{t},t').
/// \end{equation}
/// ```
///
/// ---
///
/// **Langreth Rules 1 : C = A ∗ B**
///
/// Using the Langreth rules, the convolution integral (`Convolution Type 1`)
/// is split into contributions from the Matsubara, retarded, left-mixing,
/// and lesser components:
///
/// ```math
/// \begin{equation}
/// C^{M}(\tau) = \int^{\beta}_0 d\bar{\tau} \
///     A^{M} (\tau - \bar{\tau}) B^{M}(\bar{\tau}).
/// \end{equation}
/// ```
///
/// ```math
/// \begin{equation}
/// C^{R}(t,t') = \int^{t}_{t'} d\bar{t} \
///     A^{R}(t,\bar{t}) B^{R}(\bar{t},t').
/// \end{equation}
/// ```
///
/// ```math
/// \begin{equation}
/// C^{\rceil}(t,\tau) = \int^t_0 d\bar{t} \
///     A^{R}(t,\bar{t}) B^{\rceil} (\bar{t},\tau)
///                    + \int^{\beta}_0 d\bar{\tau} \
///     A^{\rceil}(t,\bar{\tau}) B^{M}(\bar{\tau} - \tau).
/// \end{equation}
/// ```
///
/// ```math
/// \begin{equation}
/// C^{<}(t,t') = \int^t_0 d\bar{t} \
///     A^{R}(t,\bar{t}) B^{<}(\bar{t},t')
///             + \int^{t'}_0 d\bar{t}\
///     A^{<}(t,\bar{t}) B^{A}(\bar{t},t')
///             -i \int^{\beta}_0 d\bar{\tau} \
///     A^{\rceil}(t,\bar{\tau}) B^{\lceil}(\bar{\tau},t').
///
/// \end{equation}
/// ```
///
/// **Langreth Rules 2 : C = A ∗ f ∗ B**
///
/// Using the Langreth rules, the convolution integral (`Convolution Type 2`)
/// is split into contributions from the Matsubara, retarded, left-mixing,
/// and lesser components:
///
/// ```math
/// \begin{equation}
/// C^{M}(\tau) = \int^{\beta}_0 d\bar{\tau}\
///     A^{M} (\tau - \bar{\tau}) f(0^-) B^{M}(\bar{\tau}).
/// \end{equation}
/// ```
///
/// ```math
/// \begin{equation}
/// C^{R}(t,t') = \int^{t}_{t'} d\bar{t} \
///     A^{R}(t,\bar{t}) f(\bar{t}) B^{R}(\bar{t},t').
/// \end{equation}
/// ```
///
/// ```math
/// \begin{equation}
/// C^{\rceil}(t,\tau) = \int^t_0 d\bar{t}\
///     A^{R}(t,\bar{t}) f(\bar{t}) B^{\rceil} (\bar{t},\tau)
///                    + \int^{\beta}_0 d\bar{\tau}\
///     A^{\rceil}(t,\bar{\tau}) f(0^-) B^{M}(\bar{\tau} - \tau).
/// \end{equation}
/// ```
///
/// ```math
/// \begin{equation}
/// C^{<}(t,t') = \int^t_0 d\bar{t}\
///     A^{R}(t,\bar{t}) f(\bar{t}) B^{<}(\bar{t},t')
///             + \int^{t'}_0 d\bar{t}\
///     A^{<}(t,\bar{t}) f(\bar{t}) B^{A}(\bar{t},t')
///             -i \int^{\beta}_0 d\bar{\tau}\
///     A^{\rceil}(t,\bar{\tau}) f(0^-) B^{\lceil}(\bar{\tau},t').
///
/// \end{equation}
/// ```
///
/// ---
///
/// **Assumption**
///
/// In the evaluation of the above integrals we make in general no assumption
/// on the hermitian properties of `A` and `B`. The integrals constitute
/// different contributions to the convolution, which we separate into the
/// Matsubara, retarded, left-mixing, and lesser components of a contour
/// function `C`. All the equations are obtained in a straightforward way from
/// the `Gregory integration` if the integration interval includes more than
/// ``k + 1`` function values, and from the `polynomial integration` or the
/// `boundary convolution` otherwise.
#[allow(clippy::too_many_arguments)]
pub fn convolution(
    c: &mut ContourGreen,
    a: &ContourGreen,
    acc: &ContourGreen,
    f: Option<&TimeFunction>,
    b: &ContourGreen,
    bcc: &ContourGreen,
    integ: &Integrator,
    beta: f64,
    h: f64,
) {
    assert!(beta > 0.0);
    assert!(h > 0.0);
    // Matsubara component: f[0] is the value at the Matsubara branch
    match f {
        Some(cf) => conv_mat_f0(
            &mut c.mat,
            &a.mat,
            &cf_mat0(cf),
            &b.mat,
            integ,
            beta,
            a.sign,
        ),
        None => conv_mat(&mut c.mat, &a.mat, &b.mat, integ, beta, a.sign),
    }
    let mut ws = ConvWorkspace::new(a.size1, a.size2, b.size1, b.size2);
    // The loop runs over every real time step, 0..=nt
    for n in 0..=(c.nt as i64) {
        convolution_time_step_inner(n, c, a, acc, f, b, bcc, integ, beta, h, &mut ws);
    }
}

/// Type-1 convenience: `C = A * B` with `Acc = A` and `Bcc = B`.
pub fn convolution_ab(
    c: &mut ContourGreen,
    a: &ContourGreen,
    b: &ContourGreen,
    integ: &Integrator,
    beta: f64,
    h: f64,
) {
    convolution(c, a, a, None, b, b, integ, beta, h);
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::linalg::DenseMat;
    use crate::types::{Contour, Sign};

    use crate::equilibrium::from_ham;

    /// Free propagator with H = 0 (returns the contour as well).
    fn free_green(nt: i64, ntau: i64, beta: f64, h: f64) -> (Contour, ContourGreen) {
        let c = Contour::new(nt, ntau, 2, 2, nt as f64 * h, beta);
        let mut g = ContourGreen::new(&c, Sign::Fermi, Complex::ZERO);
        let h0 = DenseMat::zeros(2);
        from_ham(&mut g, 0.0, &h0, beta, h);
        (c, g)
    }

    #[test]
    fn free_retarded_component_analytic() {
        // C^R(t_n, t_m) = -(n-m) h I
        let (nt, ntau) = (10i64, 20i64);
        let (beta, h) = (2.0, 0.05);
        let (c, g) = free_green(nt, ntau, beta, h);
        let mut cgreen = ContourGreen::new(&c, Sign::Fermi, Complex::ZERO);
        let integ = Integrator::new(3);
        for n in 1..=nt {
            convolution_time_step(n, &mut cgreen, &g, &g, None, &g, &g, &integ, beta, h);
        }
        for n in 1..=nt {
            for m in 0..=n {
                let want = Complex::new(-(n - m) as f64 * h, 0.0);
                let got = cgreen.get_ret(n, m, 0, 0);
                assert!((got - want).abs() < 1e-13, "ret ({n},{m}): {got} vs {want}");
            }
        }
    }

    #[test]
    fn free_full_convolution_analytic_mat_and_ret() {
        let (nt, ntau) = (8i64, 40i64);
        let (beta, h) = (2.0, 0.05);
        let (c, g) = free_green(nt, ntau, beta, h);
        let mut cgreen = ContourGreen::new(&c, Sign::Fermi, Complex::ZERO);
        let integ = Integrator::new(3);
        convolution_ab(&mut cgreen, &g, &g, &integ, beta, h);
        // Matsubara: (1/4) (tau + xi (beta - tau)) I with xi = -1
        let dtau = beta / ntau as f64;
        for m in 0..=ntau {
            let tau = m as f64 * dtau;
            let want = 0.25 * (tau - (beta - tau));
            let got = cgreen.get_mat(m, 0, 0);
            assert!(
                (got.re - want).abs() < 1e-12,
                "mat {m}: {} vs {want}",
                got.re
            );
        }
        for n in 1..=nt {
            for m in 0..=n {
                let want = Complex::new(-(n - m) as f64 * h, 0.0);
                let got = cgreen.get_ret(n, m, 0, 0);
                assert!((got - want).abs() < 1e-12, "ret ({n},{m}): {got} vs {want}");
            }
        }
    }

    #[test]
    fn free_lmix_analytic() {
        // C^⌉(t,tau) = t/2 - (i/4)(beta - tau + xi tau), xi = -1
        let (nt, ntau) = (6i64, 40i64);
        let (beta, h) = (2.0, 0.05);
        let (c, g) = free_green(nt, ntau, beta, h);
        let mut cgreen = ContourGreen::new(&c, Sign::Fermi, Complex::ZERO);
        let integ = Integrator::new(3);
        for n in 1..=nt {
            convolution_time_step(n, &mut cgreen, &g, &g, None, &g, &g, &integ, beta, h);
        }
        let dtau = beta / ntau as f64;
        for n in 1..=nt {
            for m in 0..=ntau {
                let tau = m as f64 * dtau;
                let want = Complex::new(n as f64 * h / 2.0, -(beta - 2.0 * tau) / 4.0);
                let got = cgreen.get_lmix(n, m, 0, 0);
                assert!((got - want).abs() < 5e-3, "lmix ({n},{m}): {got} vs {want}");
            }
        }
    }

    #[test]
    fn convolution_time_step_ws_matches_fresh_workspace() {
        // one shared workspace across the loop == a fresh workspace
        // per step (the `f_0 * B^M` cache must not change the numbers)
        let (nt, ntau) = (6i64, 40i64);
        let (beta, h) = (2.0, 0.05);
        let (c, g) = free_green(nt, ntau, beta, h);
        let integ = Integrator::new(3);
        let mut c1 = ContourGreen::new(&c, Sign::Fermi, Complex::ZERO);
        let mut c2 = ContourGreen::new(&c, Sign::Fermi, Complex::ZERO);
        let mut ws = ConvWorkspace::new(g.size1, g.size2, g.size1, g.size2);
        for n in 1..=nt {
            convolution_time_step(n, &mut c1, &g, &g, None, &g, &g, &integ, beta, h);
            convolution_time_step_ws(n, &mut c2, &g, &g, None, &g, &g, &integ, beta, h, &mut ws);
        }
        assert!(c1.equal(&c2));
    }
    #[test]
    fn convolution_time_step_ws_pair_switch_with_reset() {
        // two DIFFERENT (a, b) pairs on one shared workspace with a
        // reset in between (the Holstein step_D pattern) must agree
        // with fresh workspaces
        let (nt, ntau) = (6i64, 40i64);
        let (beta, h) = (2.0, 0.05);
        let (c, g) = free_green(nt, ntau, beta, h);
        let (c2, g2) = free_green(nt, ntau, beta, h);
        let integ = Integrator::new(3);
        let mut c1 = ContourGreen::new(&c, Sign::Bose, Complex::ZERO);
        let mut c2g = ContourGreen::new(&c, Sign::Bose, Complex::ZERO);
        let mut ws = ConvWorkspace::new(g.size1, g.size2, g.size1, g.size2);
        for n in 1..=nt {
            // pair 1: (g, g2)
            convolution_time_step(n, &mut c1, &g, &g, None, &g2, &g2, &integ, beta, h);
            // pair 2 on the SAME ws after reset: (g2, g)
            let mut tmp = ContourGreen::new(&c2, Sign::Bose, Complex::ZERO);
            ws.reset();
            convolution_time_step_ws(n, &mut tmp, &g2, &g2, None, &g, &g, &integ, beta, h, &mut ws);
            let mut ref2 = ContourGreen::new(&c2, Sign::Bose, Complex::ZERO);
            convolution_time_step(n, &mut ref2, &g2, &g2, None, &g, &g, &integ, beta, h);
            assert!(tmp.equal(&ref2), "pair 2 differs at n = {n}");
        }
        let _ = &mut c2g;
    }

}
