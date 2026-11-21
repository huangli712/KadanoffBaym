// Project : KadanoffBaym
// File    : src/vie2/singular.rs
// Purpose : VIE2 with a singular kernel term
//
// With `F(t,t') + F_delta(t) delta_C(t,t')` and
// `Q(t,t') + Q_delta(t) delta_C(t,t')`, the singular part of the
// solution is `G_delta = D^{-1} Q_delta` with `D = 1 + F_delta`, and
// the regular part satisfies the ordinary VIE2 with the transformed
// `F' = D^{-1} F`, `F'^{cc} = F^{cc} D^{-1}`, and
// `Q' = D^{-1} [Q - F G_delta]`. Only the changed time steps are
// re-transformed in the reusable workspace.

//!
//! *Remarks* : *VIE2 With Singular Kernel*
//!
//! Let the kernel contain a singular term, ``F(t,t') + F_{\delta}(t)
//! \delta_{\mathcal{C}}(t,t')``, and the source term ``Q(t,t') +
//! Q_{\delta}(t)\delta_{\mathcal{C}}(t,t')``. Let ``D(t) = 1 + F_{\delta}(t)``.
//! The singular part of the solution is ``G_{\delta} = D^{-1} Q_{\delta}``,
//! and the regular part satisfies the ordinary VIE2 with
//!
//! ```math
//! \begin{equation}
//! F' = D^{-1} F, \quad F'^{cc} = F^{cc} D^{-1}, \quad
//! Q' = D^{-1} [Q - F G_{\delta}].
//! \end{equation}
//! ```
//!
//! Only the data at time step `i` of ``F'``, ``F'^{cc}``, and ``Q'`` are
//! needed to transform time step `i`, together with ``D^{-1}`` and
//! ``G_{\delta}`` at the time steps ``\le i`` and at the Matsubara axis.
//! Hence the transformed functions are kept in a reusable workspace
//! (`Vie2SingularWorkspace`), and each call transforms only the time steps which
//! may have changed:
//!
//! * `tstp = 0` -> The Matsubara axis.
//!
//! * `1 ≤ tstp ≤ k + 1` -> The Matsubara axis and the time steps
//!   `1, ⋯, k + 1` (all of them are involved in the start-up).
//!
//! * `tstp ≥ k + 2` -> The time step `tstp`, and the earlier time steps
//!   which have not been transformed yet.
//!
//! The time steps after the current one are invalidated. The arithmetic per
//! time step is the same as a full re-transformation of the time steps
//! `0, ⋯, n₂` of copies of `F`, `Fcc`, and `Q` in every call.

use crate::base::MatsubaraMethod;
use crate::complex::Complex;
use crate::linalg::{DenseMat, Mode};
use crate::types::{Contour, ContourGreen, TimeFunction};
use crate::weights::Integrator;

use crate::types::MATSUBARA;

use crate::linalg::mat_inv;
use crate::storage::timestep_at;
use crate::vie2::{check_vie2, vie2_timestep};

/// Reusable workspace of [`vie2_timestep_singular`]: it keeps the transformed
/// kernel and source term at the time steps up to `ndone` (`-1` for none)
/// together with `D^{-1}` and the identity of the input objects.
///
/// The workspace is bound to its input objects: the addresses of the
/// last call are recorded, and a call with different inputs rebuilds
/// the transformed kernel and source from scratch. Hence it needs no
/// `reset()`; drop it (or call with the new inputs) when the kernel
/// changes.
pub struct Vie2SingularWorkspace {
    /// Transformed kernel `F'`.
    f: ContourGreen,

    /// Transformed kernel `F'^{cc}`.
    fcc: ContourGreen,

    /// Transformed source term `Q'`.
    q: ContourGreen,

    /// Work space for `F' G_delta`.
    fg: ContourGreen,

    /// `D^{-1}`.
    finv: TimeFunction,

    /// Last time step kept (-1 for none, -2 the Matsubara axis only).
    ndone: i64,

    /// Object addresses of the last call (Gsing, F, Fcc, Q, Fsing, Qsing).
    src: Vec<usize>,
}

impl Vie2SingularWorkspace {
    /// Last time step kept (-1 for none, -2 the Matsubara axis only).
    pub fn ndone(&self) -> i64 {
        self.ndone
    }

    /// Create a workspace compatible with the contour-ordered Green's
    /// function `g`.
    pub fn new(g: &ContourGreen) -> Vie2SingularWorkspace {
        let c = Contour::new(
            g.nt as i64,
            g.ntau as i64,
            g.size1 as i64,
            g.size2 as i64,
            1.0,
            1.0,
        );
        let s = g.sign;
        Vie2SingularWorkspace {
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

/// Solve the VIE2 with singular kernel at the time step `tstp`.
/// The singular part of the solution is written into `gsing` at the time
/// steps which are transformed in this call. `fsing` must be hermitian.
/// For time stepping pass the same workspace `ws` to all the calls.
#[allow(clippy::too_many_arguments)]
pub fn vie2_timestep_singular(
    tstp: i64,
    g: &mut ContourGreen,
    gsing: &mut TimeFunction,
    f: &ContourGreen,
    fcc: &ContourGreen,
    fsing: &TimeFunction,
    q: &ContourGreen,
    qsing: &TimeFunction,
    integ: &Integrator,
    beta: f64,
    h: f64,
    method: MatsubaraMethod,
    ws: &mut Vie2SingularWorkspace,
) {
    check_vie2(g, f, fcc, q);
    assert!(
        (1..=5).contains(&integ.k),
        "the order of the integrator must be in 1..=5"
    );
    assert!(
        MATSUBARA <= tstp && tstp <= g.nt as i64,
        "time step {tstp} out of range"
    );
    let k = integ.k;
    let d = g.size1;
    let n2 = if tstp <= k { k + 1 } else { tstp };
    assert!(
        (g.nt as i64) >= n2,
        "G must have at least {n2} real time steps"
    );
    assert!(
        gsing.size1 == d && gsing.size2 == d,
        "Gsing must be {d} x {}, got {} x {}",
        d,
        gsing.size1,
        gsing.size2
    );
    assert!(
        (gsing.nt as i64) >= n2,
        "Gsing must have at least {n2} real time steps"
    );
    for (x, name) in [(fsing, "Fsing"), (qsing, "Qsing")] {
        assert!(
            x.size1 == d && x.size2 == d,
            "{name} must be {d} x {}, got {} x {}",
            d,
            x.size1,
            x.size2
        );
        assert!(
            (x.nt as i64) >= n2,
            "{name} must have at least {n2} real time steps"
        );
    }
    assert!(
        ws.f.nt == g.nt && ws.f.ntau == g.ntau && ws.f.size1 == g.size1 && ws.f.sign == g.sign,
        "the workspace is not compatible with G"
    );

    // the transformed source term depends on the earlier values of Gsing,
    // so Gsing is a part of the key as well
    let src = vec![
        gsing as *const _ as usize,
        f as *const _ as usize,
        fcc as *const _ as usize,
        q as *const _ as usize,
        fsing as *const _ as usize,
        qsing as *const _ as usize,
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

    // D^{-1} = [1 + Fsing]^{-1} and Gsing = D^{-1} Qsing
    let id = DenseMat::identity(d);
    for &i in &steps {
        let fi = fsing.block(i);
        for z in fi {
            assert!(z.re.is_finite() && z.im.is_finite(), "Fsing is not finite");
        }
        let fim = DenseMat::from_slice(d, d, fi);
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
            "Fsing is not hermitian"
        );
        let mut dd = id.clone();
        dd.axpy(1.0, &fim);
        let dinv = mat_inv(&dd);
        ws.finv.block_mut(i).copy_from_slice(dinv.raw());
        let qm = DenseMat::from_slice(d, d, qsing.block(i));
        let gsing_i = dinv.mat_mul(&qm);
        gsing.block_mut(i).copy_from_slice(gsing_i.raw());
    }

    // F' = D^{-1} F, Fcc' = Fcc D^{-1}, Q' = D^{-1} [Q - F Gsing]
    for &i in &steps {
        vie2_singular_transform(ws, i, gsing, f, fcc, q);
    }
    ws.ndone = n2;

    vie2_timestep(tstp, g, &ws.f, &ws.fcc, &ws.q, integ, beta, h, method);
}

/// Transform the time step `i` of the kernel and the source term;
/// `ws.finv` and `gsing` must be known at the time steps `<= i`.
fn vie2_singular_transform(
    ws: &mut Vie2SingularWorkspace,
    i: i64,
    gsing: &TimeFunction,
    f: &ContourGreen,
    fcc: &ContourGreen,
    q: &ContourGreen,
) {
    let d = ws.f.size1;
    // F' = D^{-1} F
    copy_step(&mut ws.f, f, i);
    let dinv = DenseMat::from_slice(d, d, ws.finv.block(i));
    ws.f.lmul_at(i, &dinv, Mode::Normal);
    // Fcc' = Fcc D^{-1}
    copy_step(&mut ws.fcc, fcc, i);
    ws.fcc.rmul_at(i, &dinv, Mode::Normal);
    // Q' = D^{-1} Q - F' Gsing
    copy_step(&mut ws.q, q, i);
    ws.q.lmul_at(i, &dinv, Mode::Normal);
    copy_step(&mut ws.fg, &ws.f, i);
    let gsing_m = DenseMat::from_slice(d, d, gsing.block(i));
    // Q' = D^{-1} Q - F' Gsing: F' is RIGHT-multiplied by Gsing
    ws.fg.rmul_at(i, &gsing_m, Mode::Normal);
    let fg_step = timestep_at(&ws.fg, i.max(0));
    ws.q.incr_by_step(&fg_step, i, -Complex::ONE);
}

/// Copy the time step `tstp` of `src` into `dst`: the Matsubara component
/// at `MATSUBARA`, otherwise the ret/lmix rows and the less column.
fn copy_step(dst: &mut ContourGreen, src: &ContourGreen, tstp: i64) {
    if tstp == MATSUBARA {
        dst.mat.raw_mut().copy_from_slice(src.mat.raw());
        return;
    }
    for j in 0..=tstp {
        dst.ret
            .raw_block_at_mut(tstp, j)
            .copy_from_slice(src.ret.raw_block_at(tstp, j));
        dst.less
            .raw_block_at_mut(j, tstp)
            .copy_from_slice(src.less.raw_block_at(j, tstp));
    }
    for m in 0..=(src.ntau as i64) {
        dst.lmix
            .raw_block_at_mut(tstp, m)
            .copy_from_slice(src.lmix.raw_block_at(tstp, m));
    }
}
