// Project : KadanoffBaym
// File    : src/vie2/timestep.rs
// Purpose : time stepping of the real time VIE2 components
//
// All time indices are 0-based real times.

use crate::base::MatsubaraMethod;
use crate::complex::Complex;
use crate::convolution::FOp;
use crate::linalg::{BlockSolver, DenseMat};
use crate::types::ContourGreen;
use crate::weights::Integrator;

use crate::types::MATSUBARA;

use crate::convolution::incr_convolution_lmix_step;
use crate::convolution::{incr_less_adv_buf, incr_lmix_rmix_buf};
use crate::linalg::{axpy_c, axpy_c_flat};
use crate::linalg::{mul_acc_c, mul_acc_c2, mul_acc_c4};
use crate::linalg::{solve_block_left, solve_block_left_single};
use crate::linalg::{solve_block_right, solve_block_right_single};
use crate::types::{adjoint_block_into, triangular_row_offset};

use crate::vie2::workspace::Vie2Workspace;

use crate::vie2::full::{check_vie2, check_vie2_order};
use crate::vie2::start::vie2_bootstrap;
use crate::vie2::{vie2_mat, vie2_start};

/// Solve the VIE2 at the time step `tstp`:
///
/// * `MATSUBARA` -> the Matsubara component (`vie2_mat`)
/// * `0` -> the zeroth order start-up at `t = 0`
/// * `1 <= tstp <= k` -> the start-up of order `tstp`, i.e. the time
///   steps `0, .., tstp` are (re)calculated
/// * `tstp >= k + 1` -> the time stepping; the previous time steps must
///   be known
#[allow(clippy::too_many_arguments)]
pub fn vie2_timestep(
    tstp: i64,
    g: &mut ContourGreen,
    f: &ContourGreen,
    fcc: &ContourGreen,
    q: &ContourGreen,
    integ: &Integrator,
    beta: f64,
    h: f64,
    method: MatsubaraMethod,
) {
    check_vie2(g, f, fcc, q);
    check_vie2_order(integ);
    assert!(
        MATSUBARA <= tstp && tstp <= g.nt as i64,
        "time step {tstp} out of range"
    );
    assert!(beta > 0.0, "beta must be positive");
    assert!(h > 0.0, "h must be positive");
    let k = integ.k;

    if tstp == MATSUBARA {
        vie2_mat(g, f, fcc, q, beta, integ, method, 6, 40, 1e-12);
    } else if tstp == 0 {
        vie2_bootstrap(g, f, q, beta);
    } else if tstp <= k {
        vie2_start(g, f, fcc, q, tstp, beta, h);
    } else {
        let mut ws = Vie2Workspace::new(integ.k, g.size1, g.ntau);
        vie2_timestep_ws(tstp, g, f, fcc, q, integ, beta, h, &mut ws);
    }
}

/// Workspace-reusing VIE2 time stepping (the `tstp > k` branch of
/// [`vie2_timestep`]): pass one [`Vie2Workspace`] built before the
/// time-step loop. Reuse is valid while the kernel `f`/`fcc` pair is
/// unchanged (the lesser-path convolution cache assumes it).
#[allow(clippy::too_many_arguments)]
pub fn vie2_timestep_ws(
    tstp: i64,
    g: &mut ContourGreen,
    f: &ContourGreen,
    fcc: &ContourGreen,
    q: &ContourGreen,
    integ: &Integrator,
    beta: f64,
    h: f64,
    ws: &mut Vie2Workspace,
) {
    // the retarded component is solved from the right action equation,
    // so F and Fcc are exchanged
    vie2_timestep_ret_ws(tstp, g, fcc, f, q, h, integ, ws);
    vie2_timestep_lmix_ws(tstp, g, f, fcc, q, beta, h, integ, ws);
    vie2_timestep_less_ws(tstp, g, f, fcc, q, beta, h, integ, ws);
}

/// Time stepping of the retarded component: the right action equation
/// `G^R + G^R * K^R = Q^R` with the kernel `K = Fcc` and its partner
/// `Kcc = F`.
#[allow(clippy::too_many_arguments)]
pub fn vie2_timestep_ret(
    tstp: i64,
    g: &mut ContourGreen,
    kk: &ContourGreen,
    kcc: &ContourGreen,
    q: &ContourGreen,
    hstep: f64,
    integ: &Integrator,
) {
    let mut ws = Vie2Workspace::new(integ.k, g.size1, g.ntau);
    vie2_timestep_ret_ws(tstp, g, kk, kcc, q, hstep, integ, &mut ws);
}

/// Workspace-reusing core of [`vie2_timestep_ret`].
#[allow(clippy::too_many_arguments)]
pub(crate) fn vie2_timestep_ret_ws(
    tstp: i64,
    g: &mut ContourGreen,
    kk: &ContourGreen,
    kcc: &ContourGreen,
    q: &ContourGreen,
    hstep: f64,
    integ: &Integrator,
    ws: &mut Vie2Workspace,
) {
    let k = integ.k;
    let n = tstp;
    assert!(n > k);
    let d = g.size1;

    // clear the time step, G^R(n,n) = Q^R(n,n)
    for j in 0..=n {
        g.ret.raw_block_at_mut(n, j).fill(Complex::ZERO);
    }
    g.ret
        .raw_block_at_mut(n, n)
        .copy_from_slice(q.ret.raw_block_at(n, n));
    let gnn = g.ret.raw_block_at(n, n);

    // start values G^R(n, n-q), q = 1, .., k: solve sum_q X_q M_{qj} = R_j
    // reset the accumulated blocks of the start-up system
    for row in ws.ts.mblocks.iter_mut().flatten() {
        row.raw_mut().fill(Complex::ZERO);
    }
    for j in 1..=k {
        let mut rj = DenseMat::from_slice(d, d, q.ret.raw_block_at(n, n - j));
        // M_jj starts from the identity; the l = j integral term is
        // accumulated into the SAME block by the loop below
        ws.ts.mblocks[(j - 1) as usize][(j - 1) as usize].axpy(1.0, &ws.ts.id);
        for l in 0..=k {
            let mut weight = -hstep * integ.giw.at(j as usize, l as usize);
            let sref: &[Complex] = if l <= j {
                kk.ret.raw_block_at(n - l, n - j)
            } else {
                weight = -weight;
                adjoint_block_into(
                    kcc.ret.raw_block_at(n - j, n - l),
                    &mut ws.adjbuf,
                    kcc.ret.size2,
                    kcc.ret.size1,
                );
                &ws.adjbuf
            };
            if l == 0 {
                mul_acc_c(rj.raw_mut(), Complex::new(weight, 0.0), gnn, sref, d, d, d);
            } else {
                axpy_c_flat(
                    &mut ws.ts.mblocks[(j - 1) as usize][(l - 1) as usize],
                    Complex::new(-weight, 0.0),
                    sref,
                );
            }
        }
        ws.rblocks[(j - 1) as usize]
            .raw_mut()
            .copy_from_slice(rj.raw());
    }
    solve_block_left(
        &mut ws.ts.xblocks,
        &ws.ts.mblocks,
        &ws.rblocks,
        &mut ws.ts.wsk,
    );
    for qq in 1..=k {
        g.ret
            .raw_block_at_mut(n, n - qq)
            .copy_from_slice(ws.ts.xblocks[(qq - 1) as usize].raw());
    }

    // contribution of G^R(n, m), m = n-k, .., n, to the convolution
    // -int dm G^R(n,m) K^R(m,j), stored into qq(j) without the factor h
    let nel = d * d;
    ws.qqbuf.resize((n + 1) as usize * nel, Complex::ZERO);
    ws.qqbuf[..(n + 1) as usize * nel].fill(Complex::ZERO);
    let qqbuf = &mut ws.qqbuf;
    if d == 1 {
        // tuned by instruction-level measurement (mirrors the dyson
        // side): hoisted constant Omega weights and contiguous kernel
        // row walks; arithmetic shape (p = g0 * s, real w on re/im) is
        // bit-identical to the generic path
        let om = &integ.giw.omega;
        let bs = &integ.giw.big_s;
        let sk = (k + 1) as usize;
        let srow = kk.ret.raw();
        let jmax = n - k - 1;
        for m in (n - k)..=n {
            let c = (n - m) as usize; // <= k
            let g0 = g.ret.raw_block_at(n, m)[0];
            let mrow = triangular_row_offset(m as usize);
            // j <= n-2k-2: Omega[n-j][c] = omega[c]; beyond it the
            // big_s strip n-j in [k+1, 2k+1]
            let jmain = (n - 2 * k - 2).min(jmax);
            let wc = -om[c];
            for j in 0..=jmain {
                let p = g0 * srow[mrow + j as usize];
                let zq = &mut qqbuf[j as usize];
                zq.re += wc * p.re;
                zq.im += wc * p.im;
            }
            for j in (jmain + 1).max(0)..=jmax {
                let w = -bs[((n - j - k - 1) as usize) * sk + c];
                let p = g0 * srow[mrow + j as usize];
                let zq = &mut qqbuf[j as usize];
                zq.re += w * p.re;
                zq.im += w * p.im;
            }
        }
    } else if d == 2 || d == 4 {
        // d = 2/4 specializations: the K^R row (m, j) and the qqbuf
        // segment are contiguous walks; the Omega[n-j][n-m] weight is
        // the constant omega[n-m] for j <= n-2k-2 and reads big_s in
        // the narrow strip beyond (n-j in [k+1, 2k+1]; n-j <= k is
        // impossible since j <= n-k-1)
        let nel2 = nel;
        let om = &integ.giw.omega;
        let bs = &integ.giw.big_s;
        let sk = (k + 1) as usize;
        let srow = kk.ret.raw();
        let jmax = n - k - 1;
        for m in (n - k)..=n {
            let gnm = g.ret.raw_block_at(n, m);
            let c = (n - m) as usize; // <= k
            let mrow = triangular_row_offset(m as usize) * nel2;
            let jmain = (n - 2 * k - 2).min(jmax);
            if jmain >= 0 {
                let cnt = (jmain + 1) as usize;
                let w = Complex::new(-om[c], 0.0);
                let sseg = &srow[mrow..mrow + cnt * nel2];
                for (i2, sblk) in sseg.chunks_exact(nel2).enumerate() {
                    let q = &mut qqbuf[i2 * nel2..(i2 + 1) * nel2];
                    if nel2 == 4 {
                        mul_acc_c2(q, w, gnm, sblk)
                    } else {
                        mul_acc_c4(q, w, gnm, sblk)
                    };
                }
            }
            for j in (jmain + 1).max(0)..=jmax {
                let w = -bs[((n - j - k - 1) as usize) * sk + c];
                let sblk = &srow[mrow + j as usize * nel2..mrow + (j as usize + 1) * nel2];
                let q = &mut qqbuf[j as usize * nel2..(j as usize + 1) * nel2];
                if nel2 == 4 {
                    mul_acc_c2(q, Complex::new(w, 0.0), gnm, sblk)
                } else {
                    mul_acc_c4(q, Complex::new(w, 0.0), gnm, sblk)
                };
            }
        }
    } else {
        for m in (n - k)..=n {
            let gnm = g.ret.raw_block_at(n, m);
            for j in 0..=(n - k - 1) {
                let w = -integ.giw.at((n - j) as usize, (n - m) as usize);
                let s = kk.ret.raw_block_at(m, j);
                mul_acc_c(
                    &mut qqbuf[j as usize * nel..(j as usize + 1) * nel],
                    Complex::new(w, 0.0),
                    gnm,
                    s,
                    d,
                    d,
                    d,
                );
            }
        }
    }

    // the rest: t' = n-l, l = k+1, .., n
    let w0 = hstep * integ.giw.omega[0];
    let hc = Complex::new(hstep, 0.0);
    let mm = &mut ws.ts.mm;
    let qqj = &mut ws.ts.qqj;
    let gnj = &mut ws.ts.gnj;
    for l in (k + 1)..=n {
        let j = n - l;
        qqj.raw_mut()
            .copy_from_slice(&qqbuf[j as usize * nel..(j as usize + 1) * nel]);
        qqj.scale_mut(hc);
        axpy_c_flat(qqj, Complex::ONE, q.ret.raw_block_at(n, j));
        // G^R(n,j) [1 + h omega_0 K^R(j,j)] = qq(j)
        mm.set_from(&ws.ts.id);
        axpy_c_flat(mm, Complex::new(w0, 0.0), kk.ret.raw_block_at(j, j));
        gnj.raw_mut().copy_from_slice(g.ret.raw_block_at(n, j));
        solve_block_left_single(gnj, mm, qqj, &mut ws.ts.ws1);
        g.ret.raw_block_at_mut(n, j).copy_from_slice(gnj.raw());

        // contribution of G^R(n, j) to the remaining columns
        if d == 1 {
            // Omega[n-j2][l] with l = n-j >= k+1: exactly 1 for
            // j2 <= j-k-1, omega[j-j2] for j2 in [j-k, j); both with
            // the minus sign of the kernel convolution
            let om = &integ.giw.omega;
            let srow = kk.ret.raw();
            let jrow = triangular_row_offset(j as usize);
            let g0 = gnj.raw()[0];
            let jhi1 = j - k - 1;
            for j2 in 0..=jhi1 {
                let p = g0 * srow[jrow + j2 as usize];
                let zq = &mut qqbuf[j2 as usize];
                zq.re -= p.re;
                zq.im -= p.im;
            }
            for j2 in (jhi1 + 1).max(0)..j {
                let w = om[(j - j2) as usize];
                let p = g0 * srow[jrow + j2 as usize];
                let zq = &mut qqbuf[j2 as usize];
                zq.re -= w * p.re;
                zq.im -= w * p.im;
            }
        } else if d == 2 || d == 4 {
            // d = 2/4 specializations: the K^R row (j, j2) and the
            // qqbuf segment are contiguous; Omega[n-j2][n-j] is 1 for
            // j2 <= j-k-1 and omega[j-j2] in the narrow tail stripe
            // (both with the kernel's minus sign)
            let om = &integ.giw.omega;
            let srow = kk.ret.raw();
            let gj = g.ret.raw_block_at(n, j);
            let jrow = triangular_row_offset(j as usize) * nel;
            let jhi1 = j - k - 1;
            if jhi1 >= 0 {
                let cnt = (jhi1 + 1) as usize;
                let sseg = &srow[jrow..jrow + cnt * nel];
                for (i2, sblk) in sseg.chunks_exact(nel).enumerate() {
                    let q = &mut qqbuf[i2 * nel..(i2 + 1) * nel];
                    if nel == 4 {
                        mul_acc_c2(q, -Complex::ONE, gj, sblk)
                    } else {
                        mul_acc_c4(q, -Complex::ONE, gj, sblk)
                    };
                }
            }
            for j2 in (jhi1 + 1).max(0)..j {
                let w = Complex::new(-om[(j - j2) as usize], 0.0);
                let sblk = &srow[jrow + j2 as usize * nel..jrow + (j2 as usize + 1) * nel];
                let q = &mut qqbuf[j2 as usize * nel..(j2 as usize + 1) * nel];
                if nel == 4 {
                    mul_acc_c2(q, w, gj, sblk)
                } else {
                    mul_acc_c4(q, w, gj, sblk)
                };
            }
        } else {
            let gj = g.ret.raw_block_at(n, j);
            for j2 in 0..j {
                let w = -integ.giw.at((n - j2) as usize, (n - j) as usize);
                let s = kk.ret.raw_block_at(j, j2);
                mul_acc_c(
                    &mut qqbuf[j2 as usize * nel..(j2 as usize + 1) * nel],
                    Complex::new(w, 0.0),
                    gj,
                    s,
                    d,
                    d,
                    d,
                );
            }
        }
    }
}

/// Time stepping of the left-mixing component: the row `n` is cleared
/// first so that its old values do not enter the source term, then
/// `[1 + h Omega_{n0} F^R(n,n)] G^rangle(n, tau) = Q^rangle - [F*G]^rangle`
/// is solved for the whole imaginary-time row.
#[allow(clippy::too_many_arguments)]
pub fn vie2_timestep_lmix(
    tstp: i64,
    g: &mut ContourGreen,
    f: &ContourGreen,
    fcc: &ContourGreen,
    q: &ContourGreen,
    beta: f64,
    hstep: f64,
    integ: &Integrator,
) {
    let mut ws = Vie2Workspace::new(integ.k, g.size1, g.ntau);
    vie2_timestep_lmix_ws(tstp, g, f, fcc, q, beta, hstep, integ, &mut ws);
}

/// Workspace-reusing core of [`vie2_timestep_lmix`].
#[allow(clippy::too_many_arguments)]
pub(crate) fn vie2_timestep_lmix_ws(
    tstp: i64,
    g: &mut ContourGreen,
    f: &ContourGreen,
    fcc: &ContourGreen,
    q: &ContourGreen,
    beta: f64,
    hstep: f64,
    integ: &Integrator,
    ws: &mut Vie2Workspace,
) {
    let k = integ.k;
    let n = tstp;
    assert!(n > k);
    let d = g.size1;
    let ntau = g.ntau;

    // clear the current row first, so that its old values do not enter
    // the source term
    let rowlen = (ntau + 1) * d * d;
    let off = n as usize * rowlen;
    g.lmix.raw_mut()[off..off + rowlen].fill(Complex::ZERO);

    // convolution [F * G]^rangle(n, tau) without the implicit term;
    // the target is a single lmix row
    ws.ts.lrow.raw_mut().fill(Complex::ZERO);
    incr_convolution_lmix_step(
        n,
        &mut ws.ts.lrow,
        f,
        fcc,
        None,
        g,
        g,
        integ,
        beta,
        hstep,
        Complex::ONE,
        Some(&mut ws.ts.cws),
    );

    // [1 + h Omega_{n0} F^R(n,n)] G^rangle(n,tau) = Q^rangle - [F*G]^rangle
    let qb = &mut ws.ts.lqblocks[0];
    for m in 0..=(ntau as i64) {
        for a in 0..d {
            for b in 0..d {
                let v =
                    q.lmix.raw_block_at(n, m)[a * d + b] - ws.ts.lrow.raw_block_at(0, m)[a * d + b];
                qb.set(a, m as usize * d + b, v);
            }
        }
    }
    let m1 = &mut ws.ts.lmblocks[0][0];
    m1.set_from(&ws.ts.id);
    let wn = hstep * integ.giw.at(n as usize, 0);
    axpy_c(
        m1,
        Complex::new(wn, 0.0),
        &DenseMat::from_slice(d, d, f.ret.raw_block_at(n, n)),
    );
    solve_block_right(
        &mut ws.ts.lxblocks,
        &ws.ts.lmblocks,
        &ws.ts.lqblocks,
        &mut ws.ts.lsolver,
    );
    for m in 0..=(ntau as i64) {
        for a in 0..d {
            for b in 0..d {
                g.lmix.raw_block_at_mut(n, m)[a * d + b] =
                    ws.ts.lxblocks[0].get(a, m as usize * d + b);
            }
        }
    }
}

/// Time stepping of the lesser component.
///
/// Write-back convention: the leading row is
/// `G^{<}(n, j) = -G^{<}(j, n)^dagger` for `j < n`; the diagonal block
/// `G^{<}(n, n)` is stored as it is solved (it is anti-hermitian only up
/// to the discretization error).
#[allow(clippy::too_many_arguments)]
pub fn vie2_timestep_less(
    tstp: i64,
    g: &mut ContourGreen,
    f: &ContourGreen,
    fcc: &ContourGreen,
    q: &ContourGreen,
    beta: f64,
    hstep: f64,
    integ: &Integrator,
) {
    let mut ws = Vie2Workspace::new(integ.k, g.size1, g.ntau);
    vie2_timestep_less_ws(tstp, g, f, fcc, q, beta, hstep, integ, &mut ws);
}

/// Workspace-reusing core of [`vie2_timestep_less`].
#[allow(clippy::too_many_arguments)]
pub(crate) fn vie2_timestep_less_ws(
    tstp: i64,
    g: &mut ContourGreen,
    f: &ContourGreen,
    fcc: &ContourGreen,
    q: &ContourGreen,
    beta: f64,
    hstep: f64,
    integ: &Integrator,
    ws: &mut Vie2Workspace,
) {
    let n = tstp;
    assert!(n > integ.k);
    let d = g.size1;

    // source term: [F^rangle * G^lceil + F^< * G^A](j, n) accumulated
    // directly into the flat gles buffer (no timestep object)
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
        f,
        None,
        g,
        integ,
        f.sign,
        beta,
        -Complex::ONE,
    );
    incr_less_adv_buf(
        &mut ws.ts.gles,
        n,
        f,
        fcc,
        &FOp::None,
        g,
        g,
        integ,
        beta,
        hstep,
        -Complex::ONE,
        &mut ws.ts.cws,
        false,
    );
    let mut gles = std::mem::take(&mut ws.ts.gles);
    vie2_less_core(n, &mut gles, g, f, fcc, q, hstep, integ);
    ws.ts.gles = gles;
}

/// Return `(F^R(n, l), sign)` for 0-based indices: the block itself with
/// the sign `1` for `n >= l`, or its adjoint partner `F^{cc,R}(l, n)` with
/// the sign `-1` otherwise.
pub(super) fn vie2_kernel_ret(
    f: &ContourGreen,
    fcc: &ContourGreen,
    n: i64,
    l: i64,
) -> (Vec<Complex>, f64) {
    if n >= l {
        (f.ret.raw_block_at(n, l).to_vec(), 1.0)
    } else {
        let mut out = vec![Complex::ZERO; fcc.ret.size1 * fcc.ret.size2];
        adjoint_block_into(
            fcc.ret.raw_block_at(l, n),
            &mut out,
            fcc.ret.size2,
            fcc.ret.size1,
        );
        (out, -1.0)
    }
}

/// Solve the lesser component `G^<(j, n)` for `j = 0, .., n` (0-based
/// indices). On entry `gles[j]` contains the source term
/// `-[F^rangle * G^lceil + F^< * G^A](j, n)`; on exit the solution
/// (also written into `g`).
#[allow(clippy::too_many_arguments)]
pub(super) fn vie2_less_core(
    n: i64,
    gles: &mut [Vec<Complex>],
    g: &mut ContourGreen,
    f: &ContourGreen,
    fcc: &ContourGreen,
    q: &ContourGreen,
    hstep: f64,
    integ: &Integrator,
) {
    let k = integ.k;
    let d = g.size1;
    let id = DenseMat::identity(d);
    assert!(gles.len() >= (n.max(k) + 1) as usize);

    // start: j = 0, .., k, a (k+1) x (k+1) problem M X = R
    let dim = (k + 1) as usize;
    let mut ws = BlockSolver::new(dim, d);
    let mut mblocks: Vec<Vec<DenseMat>> = vec![vec![DenseMat::zeros(d); dim]; dim];
    let mut rblocks: Vec<DenseMat> = vec![DenseMat::zeros(d); dim];
    let mut adjbuf = vec![Complex::ZERO; fcc.ret.size1 * fcc.ret.size2];
    for j in 0..=k {
        let mut rj = DenseMat::from_slice(d, d, &gles[j as usize]);
        if j <= n {
            axpy_c(
                &mut rj,
                Complex::ONE,
                &DenseMat::from_slice(d, d, q.less.raw_block_at(j, n)),
            );
        } else {
            adjoint_block_into(
                q.less.raw_block_at(n, j),
                &mut adjbuf,
                q.less.size2,
                q.less.size1,
            );
            axpy_c(
                &mut rj,
                Complex::new(-1.0, 0.0),
                &DenseMat::from_slice(d, d, &adjbuf),
            );
        }
        rblocks[j as usize] = rj;
        for m in 0..=k {
            let (s, sref): (f64, &[Complex]) = if j >= m {
                (1.0, f.ret.raw_block_at(j, m))
            } else {
                adjoint_block_into(
                    fcc.ret.raw_block_at(m, j),
                    &mut adjbuf,
                    fcc.ret.size2,
                    fcc.ret.size1,
                );
                (-1.0, &adjbuf)
            };
            let w = s * hstep * integ.giw.at(j as usize, m as usize);
            let mut mjm = DenseMat::zeros(d);
            if m == j {
                mjm.axpy(1.0, &id);
            }
            axpy_c(
                &mut mjm,
                Complex::new(w, 0.0),
                &DenseMat::from_slice(d, d, sref),
            );
            mblocks[j as usize][m as usize] = mjm;
        }
    }
    let mut xblocks: Vec<DenseMat> = vec![DenseMat::zeros(d); dim];
    solve_block_right(&mut xblocks, &mblocks, &rblocks, &mut ws);
    for j in 0..=k {
        gles[j as usize] = xblocks[j as usize].raw().to_vec();
    }

    // time stepping: j = k+1, .., n
    let mut ws1 = BlockSolver::new(1, d);
    let mut qq = DenseMat::zeros(d);
    let mut mm = DenseMat::zeros(d);
    let mut xj = DenseMat::zeros(d);
    for j in (k + 1)..=n {
        qq.raw_mut().copy_from_slice(&gles[j as usize]);
        axpy_c_flat(&mut qq, Complex::ONE, q.less.raw_block_at(j, n));
        mm.raw_mut().fill(Complex::ZERO);
        mm.axpy(1.0, &id);
        let wj = hstep * integ.giw.at(j as usize, j as usize);
        axpy_c_flat(&mut mm, Complex::new(wj, 0.0), f.ret.raw_block_at(j, j));
        if d == 1 {
            // tuned by instruction-level measurement (mirrors the dyson
            // side): five-case Omega hoisted per stripe with the -h
            // prefactor folded in, contiguous kernel row and flat gles
            let om = &integ.giw.omega;
            let bs = &integ.giw.big_s;
            let sk = (k + 1) as usize;
            let srow = f.ret.raw();
            let jrow = triangular_row_offset(j as usize);
            let qrow = qq.raw_mut();
            if j >= 2 * k + 2 {
                let mmid_hi = j - k - 1;
                for m in 0..=k {
                    let w = hstep * om[m as usize];
                    let p = srow[jrow + m as usize] * gles[m as usize][0];
                    qrow[0].re -= w * p.re;
                    qrow[0].im -= w * p.im;
                }
                for m in (k + 1)..=mmid_hi {
                    // unit quadrature weight still carries hstep
                    let w = hstep;
                    let p = srow[jrow + m as usize] * gles[m as usize][0];
                    qrow[0].re -= w * p.re;
                    qrow[0].im -= w * p.im;
                }
                for m in (mmid_hi + 1)..j {
                    let w = hstep * om[(j - m) as usize];
                    let p = srow[jrow + m as usize] * gles[m as usize][0];
                    qrow[0].re -= w * p.re;
                    qrow[0].im -= w * p.im;
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
                    qrow[0].re -= w * p.re;
                    qrow[0].im -= w * p.im;
                }
            }
        } else if d == 2 || d == 4 {
            // d = 2/4 specializations: the F^R row (j, m) is contiguous
            // in m; the five-case Omega[j][m] splits into the omega
            // head, the unit interior (both scaled by hstep with the
            // equation's minus sign) and the omega tail
            let nel = d * d;
            let om = &integ.giw.omega;
            let bs = &integ.giw.big_s;
            let sk = (k + 1) as usize;
            let srow = f.ret.raw();
            let jrow = triangular_row_offset(j as usize) * nel;
            let qrow = qq.raw_mut();
            if j >= 2 * k + 2 {
                let head = &srow[jrow..jrow + (k as usize + 1) * nel];
                for (m, (sblk, g)) in head.chunks_exact(nel).zip(gles.iter()).enumerate() {
                    let c = Complex::new(-hstep * om[m], 0.0);
                    if d == 2 {
                        mul_acc_c2(qrow, c, sblk, g);
                    } else {
                        mul_acc_c4(qrow, c, sblk, g);
                    }
                }
                let m0 = k as usize + 1;
                let m1 = (j - k) as usize;
                let mid = &srow[jrow + m0 * nel..jrow + m1 * nel];
                let c = Complex::new(-hstep, 0.0);
                for (sblk, g) in mid.chunks_exact(nel).zip(&gles[m0..m1]) {
                    if d == 2 {
                        mul_acc_c2(qrow, c, sblk, g);
                    } else {
                        mul_acc_c4(qrow, c, sblk, g);
                    }
                }
                let m0 = (j - k) as usize;
                let tail = &srow[jrow + m0 * nel..jrow + j as usize * nel];
                for (m, (sblk, g)) in tail.chunks_exact(nel).zip(&gles[m0..]).enumerate() {
                    let c = Complex::new(-hstep * om[k as usize - m], 0.0);
                    if d == 2 {
                        mul_acc_c2(qrow, c, sblk, g);
                    } else {
                        mul_acc_c4(qrow, c, sblk, g);
                    }
                }
            } else {
                let seg = &srow[jrow..jrow + j as usize * nel];
                for (m, (sblk, g)) in seg.chunks_exact(nel).zip(gles.iter()).enumerate() {
                    let w0 = if m <= k as usize {
                        bs[((j - k - 1) as usize) * sk + m]
                    } else {
                        om[(j - m as i64) as usize]
                    };
                    let c = Complex::new(-hstep * w0, 0.0);
                    if d == 2 {
                        mul_acc_c2(qrow, c, sblk, g);
                    } else {
                        mul_acc_c4(qrow, c, sblk, g);
                    }
                }
            }
        } else {
            for m in 0..j {
                let w = hstep * integ.giw.at(j as usize, m as usize);
                mul_acc_c(
                    qq.raw_mut(),
                    Complex::new(-w, 0.0),
                    f.ret.raw_block_at(j, m),
                    &gles[m as usize],
                    d,
                    d,
                    d,
                );
            }
        }
        solve_block_right_single(&mut xj, &mm, &qq, &mut ws1);
        gles[j as usize].copy_from_slice(xj.raw());
    }

    // write the stored triangle
    for j in 0..=n {
        g.less
            .raw_block_at_mut(j, n)
            .copy_from_slice(&gles[j as usize]);
    }
}
