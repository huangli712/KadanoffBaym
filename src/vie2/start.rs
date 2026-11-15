// Project : KadanoffBaym
// File    : src/vie2/start.rs
// Purpose : start-up of the real time VIE2 components at the first
//           k + 1 time steps and the zeroth order bootstrap (blueprint:
//           solvers/vie2_start.jl)
//
// All time indices follow the NESSi convention (0-based real times).

use crate::complex::Complex;
use crate::convolution::{ConvWorkspace, FOp};
use crate::linalg::{BlockSolver, DenseMat};
use crate::types::{Contour, ContourGreen, GreenTimeStep};
use crate::weights::Integrator;

use crate::vie2::timestep::{vie2_kernel_ret, vie2_less_core};
use crate::convolution::{
    incr_convolution_less_adv_step, incr_convolution_lmix_rmix_step, incr_less_adv_buf,
    incr_lmix_rmix_buf,
};
use crate::convolution::{conv_lmix_mat_loop};
use crate::linalg::{axpy_c, solve_block_right};
use crate::vie2::full::{check_vie2, check_vie2_order};

const I: Complex = Complex::new(0.0, 1.0);

/// Solve the VIE2 for the first `k + 1` real time steps; the Matsubara
/// component of `g` must be known (blueprint `vie2_start!`).
#[allow(clippy::too_many_arguments)]
pub fn vie2_start(
    g: &mut ContourGreen,
    f: &ContourGreen,
    fcc: &ContourGreen,
    q: &ContourGreen,
    solve_order: i64,
    beta: f64,
    h: f64,
) {
    check_vie2(g, f, fcc, q);
    let integ = Integrator::new(solve_order);
    check_vie2_order(&integ);
    assert!((g.nt as i64) >= solve_order, "G must have at least {} real time steps", solve_order);
    assert!(beta > 0.0, "beta must be positive");
    assert!(h > 0.0, "h must be positive");
    vie2_start_impl(g, f, fcc, q, &integ, beta, h);
}

/// The three start-up components with an explicit integrator (used by
/// the `vie2_timestep!` dispatch for the orders `1, .., k`).
#[allow(clippy::too_many_arguments)]
pub fn vie2_start_impl(
    g: &mut ContourGreen,
    f: &ContourGreen,
    fcc: &ContourGreen,
    q: &ContourGreen,
    integ: &Integrator,
    beta: f64,
    h: f64,
) {
    vie2_start_ret(g, f, fcc, q, h, integ);
    vie2_start_lmix(g, f, fcc, q, beta, h, integ);
    vie2_start_less(g, f, fcc, q, beta, h, integ);
}

/// Zeroth order bootstrap at the first real time step (blueprint
/// `_vie2_bootstrap!`): the trapezoidal rule along the imaginary time
/// axis; only the data at `t = 0` are used.
pub(crate) fn vie2_bootstrap(g: &mut ContourGreen, f: &ContourGreen, q: &ContourGreen, beta: f64) {
    let d = g.size1;
    let np = g.ntau as i64; // max tau index
    let sign = g.sign;
    let dtau = beta / np as f64;
    let mut c1 = vec![Complex::ZERO; d * d];
    let mut c2 = vec![Complex::ZERO; d * d];

    // retarded component, G^R(0,0) = Q^R(0,0)
    g.ret
        .raw_block_at_mut(0, 0)
        .copy_from_slice(q.ret.raw_block_at(0, 0));

    // left-mixing component
    let tw = |l: i64, a: i64, b: i64| -> f64 {
        if l == a || l == b {
            0.5
        } else {
            1.0
        }
    };
    for m in 0..=np {
        c1.iter_mut().for_each(|z| *z = Complex::ZERO);
        c2.iter_mut().for_each(|z| *z = Complex::ZERO);
        // 0 .. tau
        if m > 0 {
            for l in 0..=m {
                let fl = f.lmix.raw_block_at(0, m - l);
                let gm = g.mat.block((np - l) as usize);
                let w = tw(l, 0, m);
                for kk in 0..d * d {
                    let mut acc = Complex::ZERO;
                    for r in 0..d {
                        acc += fl[kk / d * d + r] * gm[r * d + kk % d];
                    }
                    c1[kk] += Complex::new(w, 0.0) * acc;
                }
            }
        }
        // tau .. beta
        if m < np {
            for l in m..=np {
                let fl = f.lmix.raw_block_at(0, l);
                let gm = g.mat.block((l - m) as usize);
                let w = tw(l, m, np);
                for kk in 0..d * d {
                    let mut acc = Complex::ZERO;
                    for r in 0..d {
                        acc += fl[kk / d * d + r] * gm[r * d + kk % d];
                    }
                    c2[kk] += Complex::new(w, 0.0) * acc;
                }
            }
        }
        let blk = g.lmix.raw_block_at_mut(0, m);
        let qr = q.lmix.raw_block_at(0, m);
        for kk in 0..d * d {
            blk[kk] = qr[kk] - Complex::new(dtau, 0.0) * (c2[kk] + Complex::new(sign.as_i64() as f64, 0.0) * c1[kk]);
        }
    }

    // lesser component, G^<(0,0) = Q^<(0,0) - [F^rangle star G^lceil](0,0)
    // where G^lceil(tau,0) = -xi G^rangle(0,beta-tau)^dagger
    c1.iter_mut().for_each(|z| *z = Complex::ZERO);
    for m in 0..=np {
        let fl = f.lmix.raw_block_at(0, m);
        let gadj = crate::convolution::adjoint_block(
            g.lmix.raw_block_at(0, np - m),
            g.size1,
            g.size2,
        );
        let w = tw(m, 0, np);
        for kk in 0..d * d {
            let mut acc = Complex::ZERO;
            for r in 0..d {
                acc += fl[kk / d * d + r] * gadj[r * d + kk % d];
            }
            c1[kk] += Complex::new(w, 0.0) * acc;
        }
    }
    let blk = g.less.raw_block_at_mut(0, 0);
    let qr = q.less.raw_block_at(0, 0);
    let fac = I * Complex::new(dtau * sign.as_i64() as f64, 0.0);
    for kk in 0..d * d {
        blk[kk] = qr[kk] - fac * c1[kk];
    }
}

/// Start-up of the retarded component `G^R(n, j)` for `0 <= j <= n <= k`
/// (blueprint `_vie2_start_ret!`).
#[allow(clippy::too_many_arguments)]
pub(crate) fn vie2_start_ret(
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

    // G^R(n, n) = Q^R(n, n)
    for n in 0..=k {
        g.ret
            .raw_block_at_mut(n, n)
            .copy_from_slice(q.ret.raw_block_at(n, n));
    }

    // determine G^R(n, j), n = j+1, .., k
    for j in 0..k {
        let dim = (k - j) as usize;
        let mut ws = BlockSolver::new(dim, d);
        let mut mblocks: Vec<Vec<DenseMat>> = vec![vec![DenseMat::zeros(d); dim]; dim];
        let mut qblocks: Vec<DenseMat> = vec![DenseMat::zeros(d); dim];

        for n in (j + 1)..=k {
            let p = (n - j - 1) as usize;
            let mut qp = dm(q.ret.raw_block_at(n, j), d);
            for l in 0..=k {
                if l <= j {
                    // known G^R(l, j) = -G^R(j, l)^dagger, it goes into Q
                    let gtemp: Vec<Complex> = crate::convolution::adjoint_block(
                        g.ret.raw_block_at(j, l),
                        g.ret.size1,
                        g.ret.size2,
                    )
                    .iter()
                    .map(|z| -*z)
                    .collect();
                    let w = hstep * integ.xiw.at(j as usize, n as usize, l as usize);
                    let prod = dm(f.ret.raw_block_at(n, l), d).mat_mul(&dm(&gtemp, d));
                    axpy_c(&mut qp, Complex::new(-w, 0.0), &prod);
                } else {
                    // unknown G^R(l, j), it goes into M
                    let o = (l - j - 1) as usize;
                    let mut mpo = DenseMat::zeros(d);
                    if l == n {
                        mpo.axpy(1.0, &id);
                    }
                    let (sblk, s) = vie2_kernel_ret(f, fcc, n, l);
                    let w = s * hstep * integ.xiw.at(j as usize, n as usize, l as usize);
                    axpy_c(&mut mpo, Complex::new(w, 0.0), &dm(&sblk, d));
                    mblocks[p][o] = mpo;
                }
            }
            qblocks[p] = qp;
        }

        let mut xblocks: Vec<DenseMat> = vec![DenseMat::zeros(d); dim];
        solve_block_right(&mut xblocks, &mblocks, &qblocks, &mut ws);
        for n in (j + 1)..=k {
            g.ret
                .raw_block_at_mut(n, j)
                .copy_from_slice(xblocks[(n - j - 1) as usize].raw());
        }
    }
}

/// Start-up of the left-mixing component `G^rangle(n, tau)` for
/// `0 <= n <= k` (blueprint `_vie2_start_lmix!`).
#[allow(clippy::too_many_arguments)]
pub(crate) fn vie2_start_lmix(
    g: &mut ContourGreen,
    f: &ContourGreen,
    fcc: &ContourGreen,
    q: &ContourGreen,
    beta: f64,
    hstep: f64,
    integ: &Integrator,
) {
    let k = integ.k;
    let d = g.size1;
    let ntau = g.ntau; // intervals
    let sign = g.sign;
    let np = ntau as i64;
    let dtau = beta / np as f64;

    // source term: -delta_tau int dtau' F^rangle(n,tau') G^M(tau'-tau)
    let bt: Vec<Vec<Complex>> = (0..=np).map(|m| g.mat.block(m as usize).to_vec()).collect();
    let rowlen = (ntau + 1) * d * d;
    for n in 0..=k {
        let off = n as usize * rowlen;
        let row = &mut g.lmix.raw_mut()[off..off + rowlen];
        row.fill(Complex::ZERO);
        conv_lmix_mat_loop(
            row,
            n,
            &f.lmix,
            &bt,
            integ,
            sign,
            Complex::new(-dtau, 0.0),
            (0, np),
        );
    }

    // block system sum_q M_{pq} X_q = Q_p with X_q = G^rangle(q, tau);
    // the matrix M does not depend on tau
    let mcols = d * (ntau + 1);
    let mut ws = BlockSolver::new_rect((k + 1) as usize, d, mcols);
    let id = DenseMat::identity(d);
    let mut mblocks: Vec<Vec<DenseMat>> =
        vec![vec![DenseMat::zeros(d); (k + 1) as usize]; (k + 1) as usize];
    let mut qblocks: Vec<DenseMat> = vec![DenseMat::zeros_shape(d, mcols); (k + 1) as usize];
    for n in 0..=k as i64 {
        let p = n as usize;
        let mut qn = DenseMat::zeros_shape(d, mcols);
        for m in 0..=np {
            let src = g.lmix.raw_block_at(n, m);
            let qr = q.lmix.raw_block_at(n, m);
            for a in 0..d {
                for b in 0..d {
                    qn.set(a, m as usize * d + b, src[a * d + b] + qr[a * d + b]);
                }
            }
        }
        qblocks[p] = qn;
        for j in 0..=k as i64 {
            let (sblk, s) = vie2_kernel_ret(f, fcc, n, j);
            let w = s * hstep * integ.giw.at(n as usize, j as usize);
            let mut mnj = DenseMat::zeros(d);
            if n == j {
                mnj.axpy(1.0, &id);
            }
            axpy_c(&mut mnj, Complex::new(w, 0.0), &dm(&sblk, d));
            mblocks[p][j as usize] = mnj;
        }
    }
    let mut xblocks: Vec<DenseMat> = vec![DenseMat::zeros_shape(d, mcols); (k + 1) as usize];
    solve_block_right(&mut xblocks, &mblocks, &qblocks, &mut ws);
    for n in 0..=k as i64 {
        for m in 0..=np {
            for a in 0..d {
                for b in 0..d {
                    g.lmix.raw_block_at_mut(n, m)[a * d + b] =
                        xblocks[n as usize].get(a, m as usize * d + b);
                }
            }
        }
    }
}

/// Start-up of the lesser component `G^<(j, n)` for `0 <= j <= n <= k`
/// (blueprint `_vie2_start_less!`).
#[allow(clippy::too_many_arguments)]
pub(crate) fn vie2_start_less(
    g: &mut ContourGreen,
    f: &ContourGreen,
    fcc: &ContourGreen,
    q: &ContourGreen,
    beta: f64,
    hstep: f64,
    integ: &Integrator,
) {
    let k = integ.k;
    let d = g.size1;
    let ntau = g.ntau;

    // NESSi n = 0, .., k-1: the scratch-target convolutions
    let mut ws = ConvWorkspace::new(f.size1, f.size2, g.size1, g.size2);
    for n in 0..k {
        let mut buf: Vec<Vec<Complex>> = vec![vec![Complex::ZERO; d * d]; (k + 1) as usize];
        incr_lmix_rmix_buf(&mut buf, n, f, None, g, integ, g.sign, beta, hstep, -Complex::ONE, &mut ws);
        incr_less_adv_buf(
            &mut buf,
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
            &mut ws,
            false,
        );
        vie2_less_core(n, &mut buf, g, f, fcc, q, hstep, integ);
    }

    // NESSi n = k: the public slice-target convolutions
    let cshape = Contour::new(k, ntau as i64, d as i64, d as i64, 1.0, beta);
    let mut w = GreenTimeStep::new(&cshape, k, g.sign, Complex::ZERO);
    incr_convolution_lmix_rmix_step(
        k,
        &mut w,
        f,
        fcc,
        None,
        g,
        g,
        integ,
        beta,
        hstep,
        -Complex::ONE,
    );
    incr_convolution_less_adv_step(
        k,
        &mut w,
        f,
        fcc,
        None,
        g,
        g,
        integ,
        beta,
        hstep,
        -Complex::ONE,
    );
    let mut gles: Vec<Vec<Complex>> = (0..=k).map(|j| w.less.raw_block_at(j, k).to_vec()).collect();
    vie2_less_core(k, &mut gles, g, f, fcc, q, hstep, integ);
}

fn dm(v: &[Complex], d: usize) -> DenseMat {
    DenseMat::from_vec(d, d, v.to_vec())
}

