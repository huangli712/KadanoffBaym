// Project : KadanoffBaym
// File    : src/solvers/vie2_sin.rs
// Purpose : VIE2 with a singular kernel term (blueprint:
//           solvers/vie2_sin.jl)
//
// With `F(t,t') + F_delta(t) delta_C(t,t')` and
// `Q(t,t') + Q_delta(t) delta_C(t,t')`, the singular part of the
// solution is `G_delta = D^{-1} Q_delta` with `D = 1 + F_delta`, and
// the regular part satisfies the ordinary VIE2 with the transformed
// `F' = D^{-1} F`, `F'^{cc} = F^{cc} D^{-1}`, and
// `Q' = D^{-1} [Q - F G_delta]`. Only the changed time steps are
// re-transformed in the reusable workspace.

use super::dyson_mat::MatsubaraMethod;
use super::vie2::check_vie2;
use crate::complex::Complex;
use crate::linalg::lu::mat_inv;
use crate::linalg::mat::DenseMat;
use crate::types::contour::Contour;
use crate::types::green::{ContourGreen, TimeFunction};
use crate::types::MATSUBARA;
use crate::weights::Integrator;

/// Reusable workspace of [`vie2_timestep_sin`] (blueprint
/// `VIE2SinWorkspace`): it keeps the transformed kernel and source term
/// at the time steps up to `ndone` (`-1` for none) together with
/// `D^{-1}` and the identity of the input objects.
pub struct Vie2SinWorkspace {
    /// Transformed kernel `F'`.
    pub f: ContourGreen,
    /// Transformed kernel `F'^{cc}`.
    pub fcc: ContourGreen,
    /// Transformed source term `Q'`.
    pub q: ContourGreen,
    /// Work space for `F' G_delta`.
    pub fg: ContourGreen,
    /// `D^{-1}`.
    pub finv: TimeFunction,
    /// Last time step kept (-1 for none, -2 the Matsubara axis only).
    pub ndone: i64,
    /// Object addresses of the last call (Gsin, F, Fcc, Q, Fsin, Qsin).
    src: Vec<usize>,
}

impl Vie2SinWorkspace {
    /// Create a workspace compatible with the contour-ordered Green's
    /// function `g` (blueprint `VIE2SinWorkspace(G)`).
    pub fn new(g: &ContourGreen) -> Vie2SinWorkspace {
        let c = Contour::new(g.nt as i64, g.ntau as i64, g.size1 as i64, g.size2 as i64, 1.0, 1.0);
        let s = g.sign;
        Vie2SinWorkspace {
            f: ContourGreen::new(&c, s, Complex::ZERO),
            fcc: ContourGreen::new(&c, s, Complex::ZERO),
            q: ContourGreen::new(&c, s, Complex::ZERO),
            fg: ContourGreen::new(&c, s, Complex::ZERO),
            finv: TimeFunction::new(g.nt, g.size1, g.size2, Complex::ZERO),
            ndone: -1,
            src: Vec::new(),
        }
    }
}

/// Copy the time step `tstp` of `src` into `dst` (the per-step
/// `memcpy!` of the blueprint): the Matsubara component at
/// `MATSUBARA`, otherwise the ret/lmix rows and the less column.
fn copy_step(dst: &mut ContourGreen, src: &ContourGreen, tstp: i64) {
    let b = src.size1 * src.size2;
    if tstp == MATSUBARA {
        dst.mat.raw_mut().copy_from_slice(src.mat.raw());
        return;
    }
    for j in 0..=tstp {
        dst.ret.raw_block_at_mut(tstp, j)
            .copy_from_slice(src.ret.raw_block_at(tstp, j));
        dst.less.raw_block_at_mut(j, tstp)
            .copy_from_slice(src.less.raw_block_at(j, tstp));
    }
    for m in 0..=(src.ntau as i64) {
        dst.lmix.raw_block_at_mut(tstp, m)
            .copy_from_slice(src.lmix.raw_block_at(tstp, m));
    }
    let _ = b;
}

/// Transform the time step `i` of the kernel and the source term
/// (blueprint `_vie2_sin_transform!`); `ws.finv` and `gsin` must be
/// known at the time steps `<= i`.
fn vie2_sin_transform(
    ws: &mut Vie2SinWorkspace,
    i: i64,
    gsin: &TimeFunction,
    f: &ContourGreen,
    fcc: &ContourGreen,
    q: &ContourGreen,
) {
    let d = ws.f.size1;
    // F' = D^{-1} F
    copy_step(&mut ws.f, f, i);
    let dinv = DenseMat::from_vec(d, d, ws.finv.block(i).to_vec());
    ws.f.lmul_at(i, &dinv, crate::basics::Mode::Normal);
    // Fcc' = Fcc D^{-1}
    copy_step(&mut ws.fcc, fcc, i);
    ws.fcc.rmul_at(i, &dinv, crate::basics::Mode::Normal);
    // Q' = D^{-1} Q - F' Gsin
    copy_step(&mut ws.q, q, i);
    ws.q.lmul_at(i, &dinv, crate::basics::Mode::Normal);
    copy_step(&mut ws.fg, &ws.f, i);
    let gsin_m = DenseMat::from_vec(d, d, gsin.block(i).to_vec());
    // Q' = D^{-1} Q - F' Gsin: F' is RIGHT-multiplied by Gsin
    // (blueprint smul!(tF1, Gsin, i) = tF1 * Gsin)
    ws.fg.rmul_at(i, &gsin_m, crate::basics::Mode::Normal);
    let fg_step = crate::basics::indexing::timestep_at(&ws.fg, i.max(0));
    ws.q.incr_by_step(&fg_step, i, -Complex::ONE);
}

/// Solve the VIE2 with singular kernel at the time step `tstp`
/// (blueprint `vie2_timestep_sin!` with the KB convention `tstp + 1`).
/// The singular part of the solution is written into `gsin` at the time
/// steps which are transformed in this call. `fsin` must be hermitian.
/// For time stepping pass the same workspace `ws` to all the calls.
#[allow(clippy::too_many_arguments)]
pub fn vie2_timestep_sin(
    tstp: i64,
    g: &mut ContourGreen,
    gsin: &mut TimeFunction,
    f: &ContourGreen,
    fcc: &ContourGreen,
    fsin: &TimeFunction,
    q: &ContourGreen,
    qsin: &TimeFunction,
    integ: &Integrator,
    beta: f64,
    h: f64,
    method: MatsubaraMethod,
    ws: &mut Vie2SinWorkspace,
) {
    check_vie2(g, f, fcc, q);
    assert!((1..=5).contains(&integ.k), "the order of the integrator must be in 1..=5");
    assert!(
        MATSUBARA <= tstp && tstp <= g.nt as i64,
        "time step {tstp} out of range"
    );
    let k = integ.k;
    let d = g.size1;
    let n2 = if tstp <= k { k + 1 } else { tstp };
    assert!((g.nt as i64) >= n2, "G must have at least {n2} real time steps");
    assert!(
        gsin.size1 == d && gsin.size2 == d,
        "Gsin must be {d} x {}, got {} x {}",
        d,
        gsin.size1,
        gsin.size2
    );
    assert!((gsin.nt as i64) >= n2, "Gsin must have at least {n2} real time steps");
    for (x, name) in [(fsin, "Fsin"), (qsin, "Qsin")] {
        assert!(
            x.size1 == d && x.size2 == d,
            "{name} must be {d} x {}, got {} x {}",
            d,
            x.size1,
            x.size2
        );
        assert!((x.nt as i64) >= n2, "{name} must have at least {n2} real time steps");
    }
    assert!(ws.f.nt == g.nt && ws.f.ntau == g.ntau && ws.f.size1 == g.size1 && ws.f.sign == g.sign,
        "the workspace is not compatible with G");

    // the transformed source term depends on the earlier values of Gsin,
    // so Gsin is a part of the key as well
    let src = vec![
        gsin as *const _ as usize,
        f as *const _ as usize,
        fcc as *const _ as usize,
        q as *const _ as usize,
        fsin as *const _ as usize,
        qsin as *const _ as usize,
    ];
    if ws.src != src {
        ws.ndone = -1;
        ws.src = src;
    }

    // the time steps which may have changed; after a reset (ndone < 0)
    // the Matsubara axis must be transformed as well
    let lo = if tstp <= k || ws.ndone < 0 {
        MATSUBARA
    } else {
        (ws.ndone + 1).min(tstp)
    };

    // steps to transform: the Matsubara axis plus the real steps
    let mut steps: Vec<i64> = Vec::new();
    if lo == MATSUBARA {
        steps.push(MATSUBARA);
    }
    for i in lo.max(0)..=n2 {
        steps.push(i);
    }

    // D^{-1} = [1 + Fsin]^{-1} and Gsin = D^{-1} Qsin
    let id = DenseMat::identity(d);
    for &i in &steps {
        let fi = fsin.block(i);
        for z in fi {
            assert!(z.re.is_finite() && z.im.is_finite(), "Fsin is not finite");
        }
        let fim = DenseMat::from_vec(d, d, fi.to_vec());
        // ||F - F^dag|| <= tol max(1, ||F||)
        let mut r = 0.0f64;
        let mut nrm = 0.0f64;
        for a in 0..d {
            for b in 0..d {
                let dz = fim.get(a, b) - fim.get(b, a).conj();
                r += dz.abs() * dz.abs();
                let e = fim.get(a, b);
                nrm += e.abs() * e.abs();
            }
        }
        assert!(
            r.sqrt() <= 1e-12 * 1.0f64.max(nrm.sqrt()),
            "Fsin is not hermitian"
        );
        let mut dd = id.clone();
        dd.axpy(1.0, &fim);
        let dinv = mat_inv(&dd);
        ws.finv.block_mut(i).copy_from_slice(dinv.raw());
        let qm = DenseMat::from_vec(d, d, qsin.block(i).to_vec());
        let gsin_i = dinv.mat_mul(&qm);
        gsin.block_mut(i).copy_from_slice(gsin_i.raw());
    }

    // F' = D^{-1} F, Fcc' = Fcc D^{-1}, Q' = D^{-1} [Q - F Gsin]
    for &i in &steps {
        vie2_sin_transform(ws, i, gsin, f, fcc, q);
    }
    ws.ndone = n2;

    super::vie2_timestep::vie2_timestep(tstp, g, &ws.f, &ws.fcc, &ws.q, integ, beta, h, method);
}
