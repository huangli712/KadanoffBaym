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

use crate::complex::Complex;
use crate::convolution::{ConvWorkspace, FOp};
use crate::linalg::{BlockSolver, DenseMat};
use crate::types::{ContourGreen, LmixComp, TimeFunction};
use crate::weights::Integrator;

use crate::convolution::{incr_convolution_lmix_step, mul_acc};
use crate::convolution::{incr_less_adv_buf, incr_lmix_rmix_buf};
use crate::dyson::full::check_dyson_rt;
use crate::linalg::{axpy_c, axpy_c_flat};
use crate::linalg::{solve_block_left, solve_block_left_single};
use crate::linalg::{solve_block_right, solve_block_right_single};
use crate::types::{adjoint_block_into, tri_off};

/// Reusable scratch of the Dyson time stepping: one instance per
/// driver call, all inner loops are allocation-free.
pub(crate) struct DysonWs {
    wsk: BlockSolver,
    ws1: BlockSolver,
    mblocks: Vec<Vec<DenseMat>>,
    qblocks: Vec<DenseMat>,
    xblocks: Vec<DenseMat>,
    prod: DenseMat,
    mm: DenseMat,
    qqj: DenseMat,
    gnj: DenseMat,
    xj: DenseMat,
    gnn: Vec<Complex>,
    gnn_dm: DenseMat,
    adj: Vec<Complex>,
    diffw: Vec<Complex>,
    qq: Vec<Complex>,
}

impl DysonWs {
    fn new(k: i64, d: usize) -> DysonWs {
        let ku = k as usize;
        let mut gnn = vec![Complex::ZERO; d * d];
        for i in 0..d {
            gnn[i * (d + 1)] = Complex::new(0.0, -1.0);
        }
        let mut ws = DysonWs {
            wsk: BlockSolver::new(ku, d),
            ws1: BlockSolver::new(1, d),
            mblocks: vec![vec![DenseMat::zeros(d); ku]; ku],
            qblocks: vec![DenseMat::zeros(d); ku],
            xblocks: vec![DenseMat::zeros(d); ku],
            prod: DenseMat::zeros(d),
            mm: DenseMat::zeros(d),
            qqj: DenseMat::zeros(d),
            gnj: DenseMat::zeros(d),
            xj: DenseMat::zeros(d),
            gnn,
            gnn_dm: DenseMat::zeros_shape(1, 1),
            adj: vec![Complex::ZERO; d * d],
            diffw: Vec::new(),
            qq: Vec::new(),
        };
        ws.gnn_dm = DenseMat::from_vec(d, d, ws.gnn.clone());
        ws
    }
}

/// `out = a * b` for square `d x d` flat blocks, zero-allocation path
/// (the convolution workhorse `mul_acc` on a preallocated target).
fn mul_sq_into(out: &mut DenseMat, a: &[Complex], b: &[Complex], d: usize) {
    let dst = out.raw_mut();
    dst.fill(Complex::ZERO);
    mul_acc(dst, 1.0, a, b, d, d, d);
}

/// Solve the Dyson equation at the real time step `tstp > solve_order`.
/// The previous time steps of `g` must be known; the Matsubara component
/// is not recalculated.
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
    let mut ws = DysonWs::new(k, g.size1);
    dyson_timestep_ret_ws(tstp, g, mu, h, sigma, hstep, &integ, &mut ws);
    dyson_timestep_lmix(tstp, g, mu, h, sigma, beta, hstep, &integ);
    dyson_timestep_less_ws(tstp, g, mu, h, sigma, beta, hstep, &integ, &mut ws);
}

/// Solve the lesser component `G^<(j, n)` for `j = 0, .., n` (0-based
/// indices). On entry `gles[j]` contains the source term
/// `[Sigma^rangle * G^lceil + Sigma^< * G^A](j, n)`; on exit the
/// solution (also written into `g`).
#[allow(clippy::too_many_arguments)]
pub fn dyson_less_core(
    n: i64,
    gles: &mut [Vec<Complex>],
    g: &mut ContourGreen,
    mu: f64,
    h: &TimeFunction,
    sigma: &ContourGreen,
    hstep: f64,
    integ: &Integrator,
) {
    let mut ws = DysonWs::new(integ.k, g.size1);
    dyson_less_core_ws(n, gles, g, mu, h, sigma, hstep, integ, &mut ws);
}

/// Workspace-sharing variant of [`dyson_less_core`]: the caller owns
/// the [`DysonWs`] buffers (one construction per driver instead of one
/// per time step).
#[allow(clippy::too_many_arguments)]
pub(crate) fn dyson_less_core_ws(
    n: i64,
    gles: &mut [Vec<Complex>],
    g: &mut ContourGreen,
    mu: f64,
    h: &TimeFunction,
    sigma: &ContourGreen,
    hstep: f64,
    integ: &Integrator,
    ws: &mut DysonWs,
) {
    let k = integ.k;
    let d = g.size1;
    assert!(gles.len() >= (n.max(k) + 1) as usize);

    // start: j = 1, .., k, solve a k x k problem M X = Q
    // initial value G^<(0, n) = -G^rangle(n, 0)^dagger
    adjoint_block_into(g.lmix.raw_block_at(n, 0), &mut ws.adj, g.size2, g.size1);
    let g0: Vec<Complex> = ws.adj.iter().map(|z| -*z).collect();
    gles[0].copy_from_slice(&g0);
    let id = DenseMat::identity(d);
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
            let mj = &mut ws.mblocks[(j - 1) as usize][(m - 1) as usize];
            mj.zero_mut();
            axpy_c(mj, cw, &id);
            if m == j {
                // Hamiltonian
                axpy_c(mj, Complex::new(mu, 0.0), &id);
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
    let mblocks = &ws.mblocks;
    let qblocks = &ws.qblocks;
    solve_block_right(&mut ws.xblocks, mblocks, qblocks, &mut ws.wsk);
    for j in 1..=k {
        gles[j as usize].copy_from_slice(ws.xblocks[(j - 1) as usize].raw());
    }

    // time stepping: j = k+1, .., n
    for j in (k + 1)..=n {
        let qq = &mut ws.qqj;
        qq.copy_flat(&gles[j as usize]);
        for p in 1..=(k + 1) {
            let w = -Complex::I / Complex::new(hstep, 0.0)
                * Complex::new(integ.bdw.at(p as usize), 0.0);
            axpy_c_flat(qq, w, &gles[(j - p) as usize]);
        }
        // M = i b_0 / h + mu - H(j) - h Omega_{jj} Sigma^R(j,j)
        let mm = &mut ws.mm;
        mm.zero_mut();
        axpy_c(
            mm,
            Complex::I / Complex::new(hstep, 0.0) * Complex::new(integ.bdw.at(0), 0.0)
                + Complex::new(mu, 0.0),
            &id,
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
            let jrow = tri_off(j as usize);
            let qrow = qq.raw_mut();
            // Omega[j][m], j fixed: m <= k reads omega[m] (j >= 2k+2)
            // or big_s (j in [k+1, 2k+1]); k < m < j-k is 1; m >= j-k
            // reads omega[j-m]
            if j >= 2 * k + 2 {
                let mmid_lo = k + 1;
                let mmid_hi = (j - k - 1).min(j - 1);
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
                for m in (mmid_hi + 1).max(j - k)..j {
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
        solve_block_right_single(xj, mm, qq, &mut ws.ws1);
        gles[j as usize].copy_from_slice(xj.raw());
    }

    // write the stored triangle
    for j in 0..=n {
        g.less
            .raw_block_at_mut(j, n)
            .copy_from_slice(&gles[j as usize]);
    }
}

/// Time stepping of the retarded component `G^R(n, j)` for `0 <= j <= n`
/// with `n > k`.
#[allow(clippy::too_many_arguments)]
pub fn dyson_timestep_ret(
    tstp: i64,
    g: &mut ContourGreen,
    mu: f64,
    h: &TimeFunction,
    sigma: &ContourGreen,
    hstep: f64,
    integ: &Integrator,
) {
    let mut ws = DysonWs::new(integ.k, g.size1);
    dyson_timestep_ret_ws(tstp, g, mu, h, sigma, hstep, integ, &mut ws);
}

/// The workspace-sharing variant of [`dyson_timestep_ret`].
#[allow(clippy::too_many_arguments)]
pub(crate) fn dyson_timestep_ret_ws(
    tstp: i64,
    g: &mut ContourGreen,
    mu: f64,
    h: &TimeFunction,
    sigma: &ContourGreen,
    hstep: f64,
    integ: &Integrator,
    ws: &mut DysonWs,
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
    for mrow in ws.mblocks.iter_mut() {
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
                axpy_c(&mut ws.mblocks[(j - 1) as usize][(l - 1) as usize], cw, &id);
            }
        }
        // Hamiltonian
        axpy_c(
            &mut ws.mblocks[(j - 1) as usize][(j - 1) as usize],
            Complex::new(mu, 0.0),
            &id,
        );
        axpy_c_flat(
            &mut ws.mblocks[(j - 1) as usize][(j - 1) as usize],
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
                    &mut ws.mblocks[(j - 1) as usize][(l - 1) as usize],
                    Complex::new(-weight, 0.0),
                    sref,
                );
            }
        }
    }
    solve_block_left(&mut ws.xblocks, &ws.mblocks, &ws.qblocks, &mut ws.wsk);
    for q in 1..=k {
        g.ret
            .raw_block_at_mut(n, n - q)
            .copy_from_slice(ws.xblocks[(q - 1) as usize].raw());
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
            let mrow = tri_off(m as usize);
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
    let mm = &mut ws.mm;
    for l in (k + 1)..=n {
        let j = n - l;
        let qqj = &mut ws.qqj;
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
        let gnj = &mut ws.gnj;
        gnj.copy_flat(g.ret.raw_block_at(n, j));
        solve_block_left_single(gnj, mm, qqj, &mut ws.ws1);
        g.ret.raw_block_at_mut(n, j).copy_from_slice(gnj.raw());

        // contribution of G^R(n, j) to the remaining columns
        if d == 1 {
            // Omega[n-j2][l] with l = n-j >= k+1: exactly 1 for
            // j2 <= j-k-1, omega[j-j2] for j2 in [j-k, j)
            let om = &integ.giw.omega;
            let srow = sigma.ret.raw();
            let jrow = tri_off(j as usize);
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
pub fn dyson_timestep_lmix(
    tstp: i64,
    g: &mut ContourGreen,
    mu: f64,
    h: &TimeFunction,
    sigma: &ContourGreen,
    beta: f64,
    hstep: f64,
    integ: &Integrator,
) {
    let k = integ.k;
    let n = tstp;
    assert!(n > k);
    let d = g.size1;
    let ntau = g.ntau;
    let id = DenseMat::identity(d);

    // clear the current row (the implicit term does not enter the source)
    let rowlen = (ntau + 1) * d * d;
    let off = n as usize * rowlen;
    g.lmix.raw_mut()[off..off + rowlen].fill(Complex::ZERO);

    // known part of the convolution [Sigma * G]^rangle(n, tau); the
    // target is a single lmix row
    let mut w = LmixComp::new(0, ntau, d, d, Complex::ZERO);
    incr_convolution_lmix_step(
        n,
        &mut w,
        sigma,
        sigma,
        None,
        g,
        g,
        integ,
        beta,
        hstep,
        Complex::ONE,
    );

    // backward differentiation, G^rangle(m, tau), m = n-k-1, .., n-1
    let mcols = d * (ntau + 1);
    let mut ws = BlockSolver::new_rect(1, d, mcols);
    let mut q = DenseMat::zeros_shape(d, mcols);
    let qraw = q.raw_mut();
    let mcols_c = mcols;
    for m in 0..=ntau as i64 {
        for i in 0..d {
            for jj in 0..d {
                let mut v = w.raw_block_at(0, m)[i * d + jj];
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
    let mut m1 = DenseMat::zeros(d);
    axpy_c(
        &mut m1,
        Complex::I / Complex::new(hstep, 0.0) * Complex::new(integ.bdw.at(0), 0.0)
            + Complex::new(mu, 0.0),
        &id,
    );
    let wn = hstep * integ.giw.at(n as usize, 0);
    let sn = sigma.ret.raw_block_at(n, n);
    axpy_c_flat(&mut m1, Complex::new(-wn, 0.0), sn);
    axpy_c_flat(&mut m1, Complex::new(-1.0, 0.0), h.block(n));
    let mblocks = vec![vec![m1]];
    let qblocks = vec![q];
    let mut xblocks = vec![DenseMat::zeros_shape(d, mcols)];
    solve_block_right(&mut xblocks, &mblocks, &qblocks, &mut ws);
    let xraw = xblocks[0].raw();
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
pub fn dyson_timestep_less(
    tstp: i64,
    g: &mut ContourGreen,
    mu: f64,
    h: &TimeFunction,
    sigma: &ContourGreen,
    beta: f64,
    hstep: f64,
    integ: &Integrator,
) {
    let mut lws = DysonWs::new(integ.k, g.size1);
    dyson_timestep_less_ws(tstp, g, mu, h, sigma, beta, hstep, integ, &mut lws);
}

/// Workspace-sharing variant of [`dyson_timestep_less`].
#[allow(clippy::too_many_arguments)]
pub(crate) fn dyson_timestep_less_ws(
    tstp: i64,
    g: &mut ContourGreen,
    mu: f64,
    h: &TimeFunction,
    sigma: &ContourGreen,
    beta: f64,
    hstep: f64,
    integ: &Integrator,
    ws: &mut DysonWs,
) {
    let n = tstp;
    assert!(n > integ.k);
    let d = g.size1;

    // source term [Sigma^rangle * G^lceil + Sigma^< * G^A](j, n)
    // accumulated directly into the flat gles buffer (no timestep object)
    let nel = d * d;
    let mut gles: Vec<Vec<Complex>> = (0..=n).map(|_| vec![Complex::ZERO; nel]).collect();
    let mut cws = ConvWorkspace::new(sigma.size1, sigma.size2, g.size1, g.size2);
    incr_lmix_rmix_buf(
        &mut gles,
        n,
        sigma,
        None,
        g,
        integ,
        sigma.sign,
        beta,
        hstep,
        Complex::ONE,
        &mut cws,
    );
    incr_less_adv_buf(
        &mut gles,
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
        &mut cws,
        false,
    );
    dyson_less_core_ws(n, &mut gles, g, mu, h, sigma, hstep, integ, ws);
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
