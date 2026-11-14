// Project : KadanoffBaym
// File    : src/convolution/density_matrix.rs
// Purpose : equal-time lesser convolution and density matrix of a
//           convolution (blueprint: convolution/density_matrix.jl)

use super::less::{conv_less_adv_kernel, conv_lmix_rmix_kernel, conv_ret_less_kernel};
use super::mat::{cf_mat0, conv_mat_mat_1};
use super::{ConvWorkspace, FOp};
use crate::complex::c64::C64;
use crate::mat::DenseMat;
use crate::types::green::{ContourGreen, GreenTimeStep, LessComp, TimeFunction};
use crate::types::MATSUBARA;
use crate::weights::Integrator;

/// Calculate the equal-time lesser component `[A * f * B]^<(t,t)` at
/// time step `tstp >= 1`, or `-i [A * f * B]^M(beta)` if
/// `tstp == MATSUBARA` (blueprint `convolution_les_timediag`).
#[allow(clippy::too_many_arguments)]
pub fn convolution_les_timediag(
    tstp: i64,
    a: &ContourGreen,
    acc: &ContourGreen,
    f: Option<&TimeFunction>,
    b: &ContourGreen,
    bcc: &ContourGreen,
    integ: &Integrator,
    beta: f64,
    h: f64,
) -> DenseMat {
    assert!(
        a.size1 == a.size2,
        "equal-time kernels require square blocks"
    );
    assert!(beta > 0.0);
    assert!(h > 0.0);
    let s1 = a.size1;
    let s2 = a.size2;
    if tstp == MATSUBARA {
        let ntau = a.ntau as i64;
        let dtau = beta / ntau as f64;
        let mut tmp = crate::types::green::MatComp::new(ntau as usize, s1, s2, C64::ZERO);
        match f {
            Some(cf) => {
                let mut bt = b.mat.clone();
                bt.lmul(&cf_mat0(cf), crate::ops::Mode::Normal);
                conv_mat_mat_1(ntau, &mut tmp, &a.mat, &bt, integ, a.sign);
            }
            None => conv_mat_mat_1(ntau, &mut tmp, &a.mat, &b.mat, integ, a.sign),
        }
        let mut x = DenseMat::zeros_shape(s1, s2);
        let blk = tmp.block(ntau as usize);
        for q in 0..(s1 * s2) {
            x.raw_mut()[q] = C64::new(0.0, -dtau) * blk[q];
        }
        x
    } else {
        assert!(tstp >= 0);
        let mut ws = ConvWorkspace::new(a.size1, a.size2, b.size1, b.size2);
        let fop = match f {
            Some(cf) => FOp::Cf(cf),
            None => FOp::None,
        };
        // one-block target: a scratch less triangle, written at the
        // diagonal (m = tstp) only
        let mut scratch = LessComp::new(tstp as usize, s1, s2, C64::ZERO);
        // The blueprint calls the *partial* `_incr_less_adv!` kernel
        // with its default `diagcc = false`: the density matrix always
        // uses `A^<(t, t)` (no start-up substitution).
        conv_ret_less_kernel(
            &mut scratch,
            (tstp, tstp),
            tstp,
            &a.ret,
            &acc.ret,
            fop,
            &b.less,
            &bcc.less,
            integ,
            C64::new(h, 0.0),
            &mut ws,
        );
        conv_less_adv_kernel(
            &mut scratch,
            (tstp, tstp),
            tstp,
            &a.less,
            &acc.less,
            fop,
            &b.ret,
            &bcc.ret,
            integ,
            C64::new(h, 0.0),
            &mut ws,
            false,
        );
        let f0 = f.map(|cf| cf.block(MATSUBARA));
        conv_lmix_rmix_kernel(
            &mut scratch,
            (tstp, tstp),
            tstp,
            &a.lmix,
            f0,
            &bcc.lmix,
            integ,
            a.sign,
            C64::new(0.0, -(beta / a.ntau as f64)),
            &mut ws,
        );
        DenseMat::from_vec(s1, s2, scratch.raw_block_at(tstp, tstp).to_vec())
    }
}

/// Hermitian convenience of `convolution_les_timediag`.
#[allow(clippy::too_many_arguments)]
pub fn convolution_les_timediag_herm(
    tstp: i64,
    a: &ContourGreen,
    f: Option<&TimeFunction>,
    b: &ContourGreen,
    integ: &Integrator,
    beta: f64,
    h: f64,
) -> DenseMat {
    convolution_les_timediag(tstp, a, a, f, b, b, integ, beta, h)
}

/// Density matrix of the convolution `A * f * B` in the NESSi
/// convention (blueprint `convolution_density_matrix`): `-i` times
/// `convolution_les_timediag`, i.e. `-i C^<(t,t)` or `-C^M(beta)`.
#[allow(clippy::too_many_arguments)]
pub fn convolution_density_matrix(
    tstp: i64,
    a: &ContourGreen,
    acc: &ContourGreen,
    f: Option<&TimeFunction>,
    b: &ContourGreen,
    bcc: &ContourGreen,
    integ: &Integrator,
    beta: f64,
    h: f64,
) -> DenseMat {
    let x = convolution_les_timediag(tstp, a, acc, f, b, bcc, integ, beta, h);
    let mut out = DenseMat::zeros_shape(x.nrows(), x.ncols());
    for q in 0..x.raw().len() {
        out.raw_mut()[q] = C64::new(0.0, -1.0) * x.raw()[q];
    }
    out
}

/// Hermitian convenience of `convolution_density_matrix`.
#[allow(clippy::too_many_arguments)]
pub fn convolution_density_matrix_herm(
    tstp: i64,
    a: &ContourGreen,
    f: Option<&TimeFunction>,
    b: &ContourGreen,
    integ: &Integrator,
    beta: f64,
    h: f64,
) -> DenseMat {
    convolution_density_matrix(tstp, a, a, f, b, b, integ, beta, h)
}

/// Unused-import guard for the timestep type referenced by docs.
#[allow(dead_code)]
type _Step = GreenTimeStep;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::equilibrium::from_ham;
    use crate::types::contour::{Contour, Sign};

    #[test]
    fn free_density_matrix_diagonal_matches_mat() {
        // H = 0, A = B = G: rho(t) = -i [G*G]^<(t,t). Cross-check the
        // real-time branch against the full convolution's diagonal.
        let (nt, ntau) = (6i64, 20i64);
        let (beta, h) = (2.0, 0.05);
        let c = Contour::new(nt, ntau, 1, 1, 1.0, beta);
        let mut g = ContourGreen::new(&c, Sign::Fermi, C64::ZERO);
        from_ham(&mut g, 0.0, &DenseMat::zeros(1), beta, h);
        let integ = Integrator::new(3);
        let mut cf = ContourGreen::new(&c, Sign::Fermi, C64::ZERO);
        crate::convolution::convolution_ab(&mut cf, &g, &g, &integ, beta, h);
        for n in 1..=nt {
            let rho = convolution_density_matrix_herm(n, &g, None, &g, &integ, beta, h);
            let want = -C64::new(0.0, 1.0) * cf.get_less(n, n, 0, 0);
            let got = rho.get(0, 0);
            assert!((got - want).abs() < 1e-13, "n {n}: {got} vs {want}");
        }
    }
}
