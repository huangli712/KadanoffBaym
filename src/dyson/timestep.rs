// Project : KadanoffBaym
// File    : src/dyson/timestep.rs
// Purpose : time stepping of the real time Dyson components

//!
//! *Remarks* : *Real Time Dyson Equation*
//!
//! In the following, the time indices `n`, `j`, `l`, `m` are 0-based,
//! i.e., ``t_n = nh`` with `n = 0, 1, ⋯`. Let ``D_{nl}`` be the polynomial
//! differentiation weights, ``I_{jnl}`` the polynomial integration weights,
//! ``\Omega_{nl}`` the Gregory weights, ``\omega_l`` the Gregory end
//! weights, and ``b_p`` the backward differentiation weights of order
//! `k + 1`.
//!
//! **Start-up**: For the first `k + 1` time steps, the time derivatives and
//! integrals are evaluated by polynomial differentiation and integration.
//! For each `j = 0, ⋯, k - 1`, the unknowns ``G^{R}(n,j)`` (`n = j + 1, ⋯,
//! k`) satisfy
//!
//! ```math
//! \begin{equation}
//! \frac{i}{h}\sum^{k}_{l=0} D_{nl} G^{R}(l,j) + (\mu - H_n) G^{R}(n,j)
//!     - h\sum^{k}_{l=0} I_{jnl} \Sigma^{R}(n,l) G^{R}(l,j) = 0.
//! \end{equation}
//! ```
//!
//! The left-mixing and lesser components at `n = 1, ⋯, k` satisfy the
//! `k × k` block system
//!
//! ```math
//! \begin{equation}
//! \sum^{k}_{q=1}\left[
//!     \frac{i}{h}D_{nq} + \delta_{nq}(\mu - H_n) - h\Omega_{nq}\Sigma^{R}(n,q)
//! \right] X(q) = Q_n,
//! \end{equation}
//! ```
//!
//! where ``X(q) = G^{\rceil}(q,\tau)`` or ``G^{<}(q,n')``, and ``Q_n``
//! contains the known source terms.
//!
//! **Time stepping**: For `n > k` the time derivatives are evaluated by the
//! backward differentiation formula, and the integrals by the Gregory
//! quadrature. For example, the left-mixing component satisfies
//!
//! ```math
//! \begin{equation}
//! \left[\frac{i}{h}b_0 + \mu - H_n - h\omega_0 \Sigma^{R}(n,n)\right]
//! G^{\rceil}(n,\tau)
//! =
//! [\Sigma * G]^{\rceil}_{known}(n,\tau)
//! - \frac{i}{h}\sum^{k+1}_{p=1} b_p G^{\rceil}(n-p,\tau).
//! \end{equation}
//! ```
//!
//! The retarded component is obtained from the right action Dyson equation
//! by marching backward in ``t'``, and the lesser component ``G^{<}(j,n)``
//! by marching forward in ``j``. The source term of the lesser component is
//! ``[\Sigma^{\rceil} * G^{\lceil} + \Sigma^{<} * G^{A}](j,n)``, while the
//! term ``\Sigma^{R} * G^{<}`` belongs to the linear system. The order of the
//! components is fixed: retarded, left-mixing, and then lesser.
//!

use crate::complex::Complex;
use crate::convolution::FOp;
use crate::linalg::DenseMat;
use crate::types::{ContourGreen, TimeFunction};
use crate::weights::Integrator;

use crate::convolution::incr_convolution_lmix_step;
use crate::convolution::{incr_less_adv_buf, incr_lmix_rmix_buf};
use crate::linalg::{axpy_c, axpy_c_flat};
use crate::linalg::{mul_acc_c2, mul_acc_c4, mul_acc_c8_u, mul_sq_into};
use crate::linalg::{solve_block_left, solve_block_left_single};
use crate::linalg::{solve_block_right, solve_block_right_single};
use crate::types::{adjoint_block_into, triangular_row_offset};

use crate::dyson::workspace::DysonWorkspace;

use crate::dyson::full::check_dyson_rt;

/// Solve the Dyson equation at the real time step `tstp > solve_order`.
/// The previous time steps of `g` must be known; the Matsubara component
/// is not recalculated.
///
/// Convenience wrapper: builds a fresh workspace on every call. In
/// time-step loops build a [`DysonWorkspace`] (and an `Integrator`)
/// once and call [`dyson_timestep_ws`] instead.
#[allow(clippy::too_many_arguments)]
pub fn dyson_timestep(
    tstp: i64,
    g: &mut ContourGreen,
    mu: f64,
    h: &TimeFunction,
    sigma: &ContourGreen,
    beta: f64,
    hstep: f64,
    solve_order: i64,
) {
    let k = solve_order;
    assert!(
        tstp > k,
        "dyson_timestep: time step {tstp} must be larger than {k}"
    );
    check_dyson_rt(g, sigma, h, k, tstp);
    assert!(beta > 0.0, "beta must be positive");
    assert!(hstep > 0.0, "h must be positive");

    let integ = Integrator::new(k);
    let mut ws = DysonWorkspace::new(k, g.size1, g.ntau);
    dyson_timestep_ws(tstp, g, mu, h, sigma, beta, hstep, &integ, &mut ws);
}

/// Workspace-reusing variant of [`dyson_timestep`]: pass one
/// `DysonWorkspace` (and one `Integrator`) built before the time-step
/// loop. Reuse is valid while `sigma` is unchanged; the lesser-path
/// convolution cache in the workspace assumes the same input sequence.
#[allow(clippy::too_many_arguments)]
pub fn dyson_timestep_ws(
    tstp: i64,
    g: &mut ContourGreen,
    mu: f64,
    h: &TimeFunction,
    sigma: &ContourGreen,
    beta: f64,
    hstep: f64,
    integ: &Integrator,
    ws: &mut DysonWorkspace,
) {
    let k = integ.k;
    assert!(
        tstp > k,
        "dyson_timestep: time step {tstp} must be larger than {k}"
    );
    check_dyson_rt(g, sigma, h, k, tstp);
    assert!(beta > 0.0, "beta must be positive");
    assert!(hstep > 0.0, "h must be positive");

    dyson_timestep_ret_ws(tstp, g, mu, h, sigma, hstep, integ, ws);
    dyson_timestep_lmix_ws(tstp, g, mu, h, sigma, beta, hstep, integ, ws);
    dyson_timestep_less_ws(tstp, g, mu, h, sigma, beta, hstep, integ, ws);
}

/// Time stepping of the retarded component `G^R(n, j)` for `0 <= j <= n`
/// with `n > k`.
#[allow(clippy::too_many_arguments)]
pub(super) fn dyson_timestep_ret_ws(
    tstp: i64,
    g: &mut ContourGreen,
    mu: f64,
    h: &TimeFunction,
    sigma: &ContourGreen,
    hstep: f64,
    integ: &Integrator,
    ws: &mut DysonWorkspace,
) {
    let k = integ.k;
    let n = tstp;
    assert!(n > k);
    let d = g.size1;
    let id = DenseMat::identity(d);

    // clear the time step, G^R(n,n) = -i
    for j in 0..=n {
        g.ret.raw_block_at_mut(n, j).fill(Complex::ZERO);
    }
    let gnn = &ws.gnn;
    let gnn_dm = &ws.gnn_dm;
    g.ret.raw_block_at_mut(n, n).copy_from_slice(gnn);

    // start values G^R(n, n-q), q = 1, .., k: solve sum_q X_q M_{qj} = Q_j
    for mrow in ws.ts.mblocks.iter_mut() {
        for m in mrow.iter_mut() {
            m.zero_mut();
        }
    }
    for q in ws.qblocks.iter_mut() {
        q.zero_mut();
    }
    for j in 1..=k {
        let qj = &mut ws.qblocks[(j - 1) as usize];
        // derivatives
        for l in 0..=k {
            let cw = Complex::I / Complex::new(hstep, 0.0)
                * Complex::new(integ.pdw.at(j as usize, l as usize), 0.0);
            if l == 0 {
                axpy_c(qj, -cw, gnn_dm);
            } else {
                axpy_c(
                    &mut ws.ts.mblocks[(j - 1) as usize][(l - 1) as usize],
                    cw,
                    &id,
                );
            }
        }
        // Hamiltonian
        axpy_c(
            &mut ws.ts.mblocks[(j - 1) as usize][(j - 1) as usize],
            Complex::new(mu, 0.0),
            &ws.ts.id,
        );
        axpy_c_flat(
            &mut ws.ts.mblocks[(j - 1) as usize][(j - 1) as usize],
            Complex::new(-1.0, 0.0),
            h.block(n - j),
        );
        // integral
        for l in 0..=k {
            let mut weight = hstep * integ.giw.at(j as usize, l as usize);
            let sref: &[Complex] = if l <= j {
                sigma.ret.raw_block_at(n - l, n - j)
            } else {
                weight = -weight;
                adjoint_block_into(
                    sigma.ret.raw_block_at(n - j, n - l),
                    &mut ws.adj,
                    sigma.ret.size2,
                    sigma.ret.size1,
                );
                &ws.adj
            };
            if l == 0 {
                mul_sq_into(&mut ws.prod, gnn, sref, d);
                axpy_c(qj, Complex::new(weight, 0.0), &ws.prod);
            } else {
                axpy_c_flat(
                    &mut ws.ts.mblocks[(j - 1) as usize][(l - 1) as usize],
                    Complex::new(-weight, 0.0),
                    sref,
                );
            }
        }
    }
    solve_block_left(
        &mut ws.ts.xblocks,
        &ws.ts.mblocks,
        &ws.qblocks,
        &mut ws.ts.wsk,
    );
    for q in 1..=k {
        g.ret
            .raw_block_at_mut(n, n - q)
            .copy_from_slice(ws.ts.xblocks[(q - 1) as usize].raw());
    }
    // contribution of G^R(n, m), m = n-k, .., n, to the convolution,
    // stored into qq(j) without the factor h
    let nel = d * d;
    ws.qq.clear();
    ws.qq.resize((n + 1) as usize * nel, Complex::ZERO);
    let qq = &mut ws.qq;
    if d == 1 {
        // tuned by instruction-level measurement: constant hoisted
        // Omega weights, contiguous sigma row walks, no block guards.
        // Same arithmetic shape as the generic path (p = g0 * s, then
        // zq.re/im += w * p.re/im with a real w) -- bit-identical.
        let om = &integ.giw.omega;
        let bs = &integ.giw.big_s;
        let sk = (k + 1) as usize;
        let srow = sigma.ret.raw();
        let jmax = n - k - 1;
        for m in (n - k)..=n {
            let c = (n - m) as usize; // <= k
            let g0 = g.ret.raw_block_at(n, m)[0];
            let mrow = triangular_row_offset(m as usize);
            // j <= n-2k-2 (or all j when n < 2k+2): n-j >= 2k+2,
            // Omega[n-j][c] = omega[c]
            let jmain = (n - 2 * k - 2).min(jmax);
            let wc = om[c];
            for j in 0..=jmain {
                let p = g0 * srow[mrow + j as usize];
                let zq = &mut qq[j as usize];
                zq.re += wc * p.re;
                zq.im += wc * p.im;
            }
            // strip n-j in [k+1, 2k+1]: Omega = big_s[(n-j-k-1)*s + c];
            // when n < 2k+2 every j sits in this regime (start at 0)
            for j in (jmain + 1).max(0)..=jmax {
                let w = bs[((n - j - k - 1) as usize) * sk + c];
                let p = g0 * srow[mrow + j as usize];
                let zq = &mut qq[j as usize];
                zq.re += w * p.re;
                zq.im += w * p.im;
            }
        }
    } else if d == 2 {
        // d = 2 specialization (see mul_acc_c2)
        for m in (n - k)..=n {
            let gnm = g.ret.raw_block_at(n, m);
            for j in 0..=(n - k - 1) {
                let w = integ.giw.at((n - j) as usize, (n - m) as usize);
                let s = sigma.ret.raw_block_at(m, j);
                let prod = &mut ws.prod;
                prod.raw_mut().fill(Complex::ZERO);
                mul_acc_c2(prod.raw_mut(), Complex::ONE, gnm, s);
                let dst = &mut qq[j as usize * nel..(j as usize + 1) * nel];
                for (zq, zp) in dst.iter_mut().zip(prod.raw().iter()) {
                    zq.re += w * zp.re;
                    zq.im += w * zp.im;
                }
            }
        }
    } else if d == 4 {
        // d = 4 specialization (see mul_acc_c4)
        for m in (n - k)..=n {
            let gnm = g.ret.raw_block_at(n, m);
            for j in 0..=(n - k - 1) {
                let w = integ.giw.at((n - j) as usize, (n - m) as usize);
                let s = sigma.ret.raw_block_at(m, j);
                let prod = &mut ws.prod;
                prod.raw_mut().fill(Complex::ZERO);
                mul_acc_c4(prod.raw_mut(), Complex::ONE, gnm, s);
                let dst = &mut qq[j as usize * nel..(j as usize + 1) * nel];
                for (zq, zp) in dst.iter_mut().zip(prod.raw().iter()) {
                    zq.re += w * zp.re;
                    zq.im += w * zp.im;
                }
            }
        }
    } else {
        for m in (n - k)..=n {
            let gnm = g.ret.raw_block_at(n, m);
            for j in 0..=(n - k - 1) {
                let w = integ.giw.at((n - j) as usize, (n - m) as usize);
                let s = sigma.ret.raw_block_at(m, j);
                mul_sq_into(&mut ws.prod, gnm, s, d);
                let prod = &ws.prod;
                let dst = &mut qq[j as usize * nel..(j as usize + 1) * nel];
                for (zq, zp) in dst.iter_mut().zip(prod.raw().iter()) {
                    zq.re += w * zp.re;
                    zq.im += w * zp.im;
                }
            }
        }
    }

    // the rest: t' = n - l, l = k+1, .., n
    ws.diffw.clear();
    ws.diffw.extend(
        (0..=(k + 1) as usize)
            .map(|p| Complex::I / Complex::new(hstep, 0.0) * Complex::new(integ.bdw.at(p), 0.0)),
    );
    let diffw = &ws.diffw;
    let w0 = hstep * integ.giw.omega[0];
    let mm = &mut ws.ts.mm;
    for l in (k + 1)..=n {
        let j = n - l;
        let qqj = &mut ws.ts.qqj;
        qqj.copy_flat(&qq[j as usize * nel..(j as usize + 1) * nel]);
        qqj.scale_mut(Complex::new(hstep, 0.0));
        for p in 1..=(k + 1) {
            axpy_c_flat(qqj, -diffw[p as usize], g.ret.raw_block_at(n, j + p));
        }
        // M = i b_0 / h + mu - H(j)^dagger - h omega_0 Sigma^R(j,j)
        mm.zero_mut();
        axpy_c(mm, diffw[0] + Complex::new(mu, 0.0), &id);
        let hj = h.block(j);
        adjoint_block_into(hj, &mut ws.adj, d, d);
        axpy_c_flat(mm, Complex::new(-1.0, 0.0), &ws.adj);
        let sj = sigma.ret.raw_block_at(j, j);
        axpy_c_flat(mm, Complex::new(-w0, 0.0), sj);
        let gnj = &mut ws.ts.gnj;
        gnj.copy_flat(g.ret.raw_block_at(n, j));
        solve_block_left_single(gnj, mm, qqj, &mut ws.ts.ws1);
        g.ret.raw_block_at_mut(n, j).copy_from_slice(gnj.raw());

        // contribution of G^R(n, j) to the remaining columns
        if d == 1 {
            // Omega[n-j2][l] with l = n-j >= k+1: exactly 1 for
            // j2 <= j-k-1, omega[j-j2] for j2 in [j-k, j)
            let om = &integ.giw.omega;
            let srow = sigma.ret.raw();
            let jrow = triangular_row_offset(j as usize);
            let g0 = gnj.raw()[0];
            let jhi1 = j - k - 1;
            for j2 in 0..=jhi1 {
                let p = g0 * srow[jrow + j2 as usize];
                let zq = &mut qq[j2 as usize];
                zq.re += p.re;
                zq.im += p.im;
            }
            for j2 in (jhi1 + 1).max(0)..j {
                let w = om[(j - j2) as usize];
                let p = g0 * srow[jrow + j2 as usize];
                let zq = &mut qq[j2 as usize];
                zq.re += w * p.re;
                zq.im += w * p.im;
            }
        } else if d == 2 {
            let prod = &mut ws.prod;
            for j2 in 0..j {
                let w = integ.giw.at((n - j2) as usize, (n - j) as usize);
                let s = sigma.ret.raw_block_at(j, j2);
                prod.raw_mut().fill(Complex::ZERO);
                mul_acc_c2(prod.raw_mut(), Complex::ONE, gnj.raw(), s);
                let dst = &mut qq[j2 as usize * nel..(j2 as usize + 1) * nel];
                for (zq, zp) in dst.iter_mut().zip(prod.raw().iter()) {
                    zq.re += w * zp.re;
                    zq.im += w * zp.im;
                }
            }
        } else if d == 4 {
            let prod = &mut ws.prod;
            for j2 in 0..j {
                let w = integ.giw.at((n - j2) as usize, (n - j) as usize);
                let s = sigma.ret.raw_block_at(j, j2);
                prod.raw_mut().fill(Complex::ZERO);
                mul_acc_c4(prod.raw_mut(), Complex::ONE, gnj.raw(), s);
                let dst = &mut qq[j2 as usize * nel..(j2 as usize + 1) * nel];
                for (zq, zp) in dst.iter_mut().zip(prod.raw().iter()) {
                    zq.re += w * zp.re;
                    zq.im += w * zp.im;
                }
            }
        } else {
            for j2 in 0..j {
                let w = integ.giw.at((n - j2) as usize, (n - j) as usize);
                let s = sigma.ret.raw_block_at(j, j2);
                mul_sq_into(&mut ws.prod, gnj.raw(), s, d);
                let prod = &ws.prod;
                let dst = &mut qq[j2 as usize * nel..(j2 as usize + 1) * nel];
                for (zq, zp) in dst.iter_mut().zip(prod.raw().iter()) {
                    zq.re += w * zp.re;
                    zq.im += w * zp.im;
                }
            }
        }
    }
}

/// Time stepping of the left-mixing component `G^rangle(n, tau)` with
/// `n > k`; the retarded component at time step `n` must be known.
#[allow(clippy::too_many_arguments)]
pub(crate) fn dyson_timestep_lmix_ws(
    tstp: i64,
    g: &mut ContourGreen,
    mu: f64,
    h: &TimeFunction,
    sigma: &ContourGreen,
    beta: f64,
    hstep: f64,
    integ: &Integrator,
    ws: &mut DysonWorkspace,
) {
    let k = integ.k;
    let n = tstp;
    assert!(n > k);
    let d = g.size1;
    let ntau = g.ntau;

    // clear the current row (the implicit term does not enter the source)
    let rowlen = (ntau + 1) * d * d;
    let off = n as usize * rowlen;
    g.lmix.raw_mut()[off..off + rowlen].fill(Complex::ZERO);

    // known part of the convolution [Sigma * G]^rangle(n, tau); the
    // target is a single lmix row
    ws.ts.lrow.raw_mut().fill(Complex::ZERO);
    incr_convolution_lmix_step(
        n,
        &mut ws.ts.lrow,
        sigma,
        sigma,
        None,
        g,
        g,
        integ,
        beta,
        hstep,
        Complex::ONE,
        Some(&mut ws.ts.cws),
    );

    // backward differentiation, G^rangle(m, tau), m = n-k-1, .., n-1
    let mcols = d * (ntau + 1);
    let qraw = ws.ts.lqblocks[0].raw_mut();
    let mcols_c = mcols;
    for m in 0..=ntau as i64 {
        for i in 0..d {
            for jj in 0..d {
                let mut v = ws.ts.lrow.raw_block_at(0, m)[i * d + jj];
                for l in (n - k - 1)..=n - 1 {
                    let cw = Complex::I / Complex::new(hstep, 0.0)
                        * Complex::new(integ.bdw.at((n - l) as usize), 0.0);
                    v -= cw * g.lmix.raw_block_at(l, m)[i * d + jj];
                }
                qraw[i * mcols_c + m as usize * d + jj] = v;
            }
        }
    }

    // [i b_0 / h - h Omega_{n0} Sigma^R(n,n) + mu - H(n)] G^rangle = Q
    let m1 = &mut ws.ts.lmblocks[0][0];
    m1.raw_mut().fill(Complex::ZERO);
    axpy_c(
        m1,
        Complex::I / Complex::new(hstep, 0.0) * Complex::new(integ.bdw.at(0), 0.0)
            + Complex::new(mu, 0.0),
        &ws.ts.id,
    );
    let wn = hstep * integ.giw.at(n as usize, 0);
    let sn = sigma.ret.raw_block_at(n, n);
    axpy_c_flat(m1, Complex::new(-wn, 0.0), sn);
    axpy_c_flat(m1, Complex::new(-1.0, 0.0), h.block(n));
    solve_block_right(
        &mut ws.ts.lxblocks,
        &ws.ts.lmblocks,
        &ws.ts.lqblocks,
        &mut ws.ts.lsolver,
    );
    let xraw = ws.ts.lxblocks[0].raw();
    for m in 0..=ntau as i64 {
        let blk = g.lmix.raw_block_at_mut(n, m);
        for i in 0..d {
            blk[i * d..(i + 1) * d].copy_from_slice(
                &xraw[i * mcols_c + m as usize * d..i * mcols_c + (m as usize + 1) * d],
            );
        }
    }
}

/// Time stepping of the lesser component `G^<(j, n)` for `0 <= j <= n`
/// with `n > k`; the retarded and left-mixing components at time step
/// `n` must be known.
#[allow(clippy::too_many_arguments)]
pub(super) fn dyson_timestep_less_ws(
    tstp: i64,
    g: &mut ContourGreen,
    mu: f64,
    h: &TimeFunction,
    sigma: &ContourGreen,
    beta: f64,
    hstep: f64,
    integ: &Integrator,
    ws: &mut DysonWorkspace,
) {
    let n = tstp;
    assert!(n > integ.k);
    let d = g.size1;

    // source term [Sigma^rangle * G^lceil + Sigma^< * G^A](j, n)
    // accumulated directly into the flat gles buffer (no timestep object)
    let nel = d * d;
    ws.ts
        .gles
        .resize((n + 1) as usize, vec![Complex::ZERO; nel]);
    for row in ws.ts.gles[..=(n as usize)].iter_mut() {
        row.fill(Complex::ZERO);
    }
    incr_lmix_rmix_buf(
        &mut ws.ts.gles,
        n,
        sigma,
        None,
        g,
        integ,
        sigma.sign,
        beta,
        Complex::ONE,
    );
    incr_less_adv_buf(
        &mut ws.ts.gles,
        n,
        sigma,
        sigma,
        &FOp::None,
        g,
        g,
        integ,
        beta,
        hstep,
        Complex::ONE,
        &mut ws.ts.cws,
        false,
    );
    let mut gles = std::mem::take(&mut ws.ts.gles);
    dyson_less_core(n, &mut gles, g, mu, h, sigma, hstep, integ, Some(ws));
    ws.ts.gles = gles;
}

/// Return `(Sigma^R(n, l), sign)` for 0-based indices: the block itself
/// with the sign `1` for `n >= l`, or its adjoint with the sign `-1`
/// otherwise.
pub(crate) fn sigma_ret(sigma: &ContourGreen, n: i64, l: i64) -> (Vec<Complex>, f64) {
    if n >= l {
        (sigma.ret.raw_block_at(n, l).to_vec(), 1.0)
    } else {
        let mut out = vec![Complex::ZERO; sigma.ret.size1 * sigma.ret.size2];
        adjoint_block_into(
            sigma.ret.raw_block_at(l, n),
            &mut out,
            sigma.ret.size2,
            sigma.ret.size1,
        );
        (out, -1.0)
    }
}

/// Solve the lesser component `G^<(j, n)` for `j = 0, .., n` (0-based
/// indices). On entry `gles[j]` contains the source term
/// `[Sigma^rangle * G^lceil + Sigma^< * G^A](j, n)`; on exit the
/// solution (also written into `g`).
///
/// `ws = None` builds a fresh workspace on every call; pass
/// `Some(&mut ws)` to reuse one across the time-step loop (the driver
/// `dyson_timestep_ws` does this).
#[allow(clippy::too_many_arguments)]
#[doc(hidden)] // test-support export: exercised directly by the integration tests
pub fn dyson_less_core(
    n: i64,
    gles: &mut [Vec<Complex>],
    g: &mut ContourGreen,
    mu: f64,
    h: &TimeFunction,
    sigma: &ContourGreen,
    hstep: f64,
    integ: &Integrator,
    ws: Option<&mut DysonWorkspace>,
) {
    let mut local;
    let ws = match ws {
        Some(ws) => ws,
        None => {
            local = DysonWorkspace::new(integ.k, g.size1, g.ntau);
            &mut local
        }
    };
    let k = integ.k;
    let d = g.size1;
    assert!(gles.len() >= (n.max(k) + 1) as usize);

    // start: j = 1, .., k, solve a k x k problem M X = Q
    // initial value G^<(0, n) = -G^rangle(n, 0)^dagger
    adjoint_block_into(g.lmix.raw_block_at(n, 0), &mut ws.adj, g.size2, g.size1);
    let g0: Vec<Complex> = ws.adj.iter().map(|z| -*z).collect();
    gles[0].copy_from_slice(&g0);
    for j in 1..=k {
        let qj = &mut ws.qblocks[(j - 1) as usize];
        qj.copy_flat(&gles[j as usize]);
        // derivative
        let cw =
            Complex::I / Complex::new(hstep, 0.0) * Complex::new(integ.pdw.at(j as usize, 0), 0.0);
        axpy_c_flat(qj, -cw, &g0);
        // integral Sigma^R * G^< (j = 0 term)
        let w0 = hstep * integ.giw.at(j as usize, 0);
        let s0 = sigma.ret.raw_block_at(j, 0);
        mul_sq_into(&mut ws.prod, s0, &g0, d);
        axpy_c(qj, Complex::new(w0, 0.0), &ws.prod);
        for m in 1..=k {
            let cw = Complex::I / Complex::new(hstep, 0.0)
                * Complex::new(integ.pdw.at(j as usize, m as usize), 0.0);
            let mj = &mut ws.ts.mblocks[(j - 1) as usize][(m - 1) as usize];
            mj.zero_mut();
            axpy_c(mj, cw, &ws.ts.id);
            if m == j {
                // Hamiltonian
                axpy_c(mj, Complex::new(mu, 0.0), &ws.ts.id);
                let hj = h.block(j);
                axpy_c_flat(mj, Complex::new(-1.0, 0.0), hj);
            }
            // integral Sigma^R * G^<
            let (s, sref): (f64, &[Complex]) = if j >= m {
                (1.0, sigma.ret.raw_block_at(j, m))
            } else {
                adjoint_block_into(
                    sigma.ret.raw_block_at(m, j),
                    &mut ws.adj,
                    sigma.ret.size2,
                    sigma.ret.size1,
                );
                (-1.0, &ws.adj)
            };
            let wm = s * hstep * integ.giw.at(j as usize, m as usize);
            axpy_c_flat(mj, Complex::new(-wm, 0.0), sref);
        }
    }
    let mblocks = &ws.ts.mblocks;
    let qblocks = &ws.qblocks;
    solve_block_right(&mut ws.ts.xblocks, mblocks, qblocks, &mut ws.ts.wsk);
    for j in 1..=k {
        gles[j as usize].copy_from_slice(ws.ts.xblocks[(j - 1) as usize].raw());
    }

    // time stepping: j = k+1, .., n
    for j in (k + 1)..=n {
        let qq = &mut ws.ts.qqj;
        qq.copy_flat(&gles[j as usize]);
        for p in 1..=(k + 1) {
            let w = -Complex::I / Complex::new(hstep, 0.0)
                * Complex::new(integ.bdw.at(p as usize), 0.0);
            axpy_c_flat(qq, w, &gles[(j - p) as usize]);
        }
        // M = i b_0 / h + mu - H(j) - h Omega_{jj} Sigma^R(j,j)
        let mm = &mut ws.ts.mm;
        mm.zero_mut();
        axpy_c(
            mm,
            Complex::I / Complex::new(hstep, 0.0) * Complex::new(integ.bdw.at(0), 0.0)
                + Complex::new(mu, 0.0),
            &ws.ts.id,
        );
        axpy_c_flat(mm, Complex::new(-1.0, 0.0), h.block(j));
        let wj = hstep * integ.giw.at(j as usize, j as usize);
        let sj = sigma.ret.raw_block_at(j, j);
        axpy_c_flat(mm, Complex::new(-wj, 0.0), sj);
        if d == 1 {
            // tuned by instruction-level measurement: five-case Omega
            // hoisted per stripe (omega walk / unit interior / omega
            // walk), contiguous sigma row and flat gles -- bit-identical
            // arithmetic (p = sm * gm, then w real on re/im)
            let om = &integ.giw.omega;
            let bs = &integ.giw.big_s;
            let sk = (k + 1) as usize;
            let srow = sigma.ret.raw();
            let jrow = triangular_row_offset(j as usize);
            let qrow = qq.raw_mut();
            // Omega[j][m], j fixed: m <= k reads omega[m] (j >= 2k+2)
            // or big_s (j in [k+1, 2k+1]); k < m < j-k is 1; m >= j-k
            // reads omega[j-m]
            if j >= 2 * k + 2 {
                let mmid_lo = k + 1;
                let mmid_hi = j - k - 1;
                for m in 0..=k {
                    let w = hstep * om[m as usize];
                    let p = srow[jrow + m as usize] * gles[m as usize][0];
                    qrow[0].re += w * p.re;
                    qrow[0].im += w * p.im;
                }
                for m in mmid_lo..=mmid_hi {
                    // unit quadrature weight still carries the h prefactor
                    let p = srow[jrow + m as usize] * gles[m as usize][0];
                    qrow[0].re += hstep * p.re;
                    qrow[0].im += hstep * p.im;
                }
                for m in (mmid_hi + 1)..j {
                    let w = hstep * om[(j - m) as usize];
                    let p = srow[jrow + m as usize] * gles[m as usize][0];
                    qrow[0].re += w * p.re;
                    qrow[0].im += w * p.im;
                }
            } else {
                // j in [k+1, 2k+1]
                for m in 0..=j - 1 {
                    let w = hstep
                        * if m <= k {
                            bs[((j - k - 1) as usize) * sk + m as usize]
                        } else {
                            om[(j - m) as usize]
                        };
                    let p = srow[jrow + m as usize] * gles[m as usize][0];
                    qrow[0].re += w * p.re;
                    qrow[0].im += w * p.im;
                }
            }
        } else if d == 2 {
            let prod = &mut ws.prod;
            let qrow = qq.raw_mut();
            for m in 0..j {
                let w = hstep * integ.giw.at(j as usize, m as usize);
                let sm = sigma.ret.raw_block_at(j, m);
                prod.raw_mut().fill(Complex::ZERO);
                mul_acc_c2(prod.raw_mut(), Complex::ONE, sm, &gles[m as usize]);
                let c = Complex::new(w, 0.0);
                for (zq, zp) in qrow.iter_mut().zip(prod.raw().iter()) {
                    *zq += c * *zp;
                }
            }
        } else if d == 4 {
            let prod = &mut ws.prod;
            let qrow = qq.raw_mut();
            for m in 0..j {
                let w = hstep * integ.giw.at(j as usize, m as usize);
                let sm = sigma.ret.raw_block_at(j, m);
                prod.raw_mut().fill(Complex::ZERO);
                mul_acc_c4(prod.raw_mut(), Complex::ONE, sm, &gles[m as usize]);
                let c = Complex::new(w, 0.0);
                for (zq, zp) in qrow.iter_mut().zip(prod.raw().iter()) {
                    *zq += c * *zp;
                }
            }
        } else if d == 8 {
            // d = 8 specialization: fold the row weight into the
            // source block once, then accumulate unweighted
            let qrow = qq.raw_mut();
            let mut sms = [Complex::ZERO; 64];
            for m in 0..j {
                let w = hstep * integ.giw.at(j as usize, m as usize);
                let sm = sigma.ret.raw_block_at(j, m);
                for (o, v) in sms.iter_mut().zip(sm) {
                    *o = *v * Complex::new(w, 0.0);
                }
                mul_acc_c8_u(qrow, &sms, &gles[m as usize]);
            }
        } else {
            for m in 0..j {
                let w = hstep * integ.giw.at(j as usize, m as usize);
                let sm = sigma.ret.raw_block_at(j, m);
                mul_sq_into(&mut ws.prod, sm, &gles[m as usize], d);
                axpy_c(qq, Complex::new(w, 0.0), &ws.prod);
            }
        }
        let xj = &mut ws.xj;
        xj.zero_mut();
        solve_block_right_single(xj, mm, qq, &mut ws.ts.ws1);
        gles[j as usize].copy_from_slice(xj.raw());
    }

    // write the stored triangle
    for j in 0..=n {
        g.less
            .raw_block_at_mut(j, n)
            .copy_from_slice(&gles[j as usize]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::types::{Contour, Sign};

    use crate::equilibrium::from_ham;

    /// One workspace across the loop equals a fresh workspace per step.
    #[test]
    fn dyson_timestep_ws_matches_fresh_workspace() {
        let (nt, ntau, beta, h, k) = (12i64, 16i64, 4.0f64, 0.05, 3i64);
        let c = Contour::new(nt, ntau, 2, 2, nt as f64 * h, beta);
        let h0 = DenseMat::from_rows(&[
            vec![Complex::new(-0.4, 0.0), Complex::new(0.0, 0.1)],
            vec![Complex::new(0.0, -0.1), Complex::new(0.6, 0.0)],
        ]);
        let mut hh = TimeFunction::new(nt as usize, 2, 2, Complex::ZERO);
        for t in -1..=nt {
            for i in 0..2 {
                for j in 0..2 {
                    hh.set(t, i, j, h0.get(i, j));
                }
            }
        }
        let mut sigma = ContourGreen::new(&c, Sign::Fermi, Complex::ZERO);
        from_ham(&mut sigma, 0.0, &h0, beta, h);
        let integ = Integrator::new(k);
        let mut g1 = ContourGreen::new(&c, Sign::Fermi, Complex::ZERO);
        let mut g2 = ContourGreen::new(&c, Sign::Fermi, Complex::ZERO);
        for g in [&mut g1, &mut g2] {
            crate::dyson::start::dyson_start(g, 0.0, &hh, &sigma, beta, h, k);
        }
        let mut dws = DysonWorkspace::new(k, 2, ntau as usize);
        for tstp in (k + 1)..=nt {
            dyson_timestep(tstp, &mut g1, 0.0, &hh, &sigma, beta, h, k);
            dyson_timestep_ws(tstp, &mut g2, 0.0, &hh, &sigma, beta, h, &integ, &mut dws);
        }
        assert!(g1.equal(&g2));
    }
}
