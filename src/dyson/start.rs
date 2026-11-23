// Project : KadanoffBaym
// File    : src/dyson/start.rs
// Purpose : start-up of the real time Dyson components at the first
//           k + 1 time steps
//
// All time indices are 0-based real times.

use crate::complex::Complex;
use crate::convolution::{ConvWorkspace, FOp};
use crate::linalg::{BlockSolver, DenseMat};
use crate::types::{Contour, ContourGreen, GreenTimeStep, TimeFunction};
use crate::weights::Integrator;

use crate::convolution::conv_lmix_mat;
use crate::convolution::{incr_convolution_less_adv_step, incr_convolution_lmix_rmix_step};
use crate::convolution::{incr_less_adv_buf, incr_lmix_rmix_buf};
use crate::linalg::{axpy_c, scale_c, solve_block_right};
use crate::types::adjoint_block_into;

use crate::dyson::full::check_dyson_rt;
use crate::dyson::timestep::{dyson_less_core, sigma_ret};

/// Solve the Dyson equation for the first `solve_order + 1` real time
/// steps; the Matsubara component of `g` must be known. The components
/// are solved in the order: retarded, left-mixing, and lesser.
#[allow(clippy::too_many_arguments)]
pub fn dyson_start(
    g: &mut ContourGreen,
    mu: f64,
    h: &TimeFunction,
    sigma: &ContourGreen,
    beta: f64,
    hstep: f64,
    solve_order: i64,
) {
    let k = solve_order;
    check_dyson_rt(g, sigma, h, k, k);
    assert!(beta > 0.0, "beta must be positive");
    assert!(hstep > 0.0, "h must be positive");

    let integ = Integrator::new(k);
    dyson_start_ret(g, mu, h, sigma, hstep, &integ);
    dyson_start_lmix(g, mu, h, sigma, beta, hstep, &integ);
    dyson_start_less(g, mu, h, sigma, beta, hstep, &integ);
}

/// Start-up of the retarded component `G^R(n, j)` for `0 <= j <= n <= k`.
#[allow(clippy::too_many_arguments)]
pub fn dyson_start_ret(
    g: &mut ContourGreen,
    mu: f64,
    h: &TimeFunction,
    sigma: &ContourGreen,
    hstep: f64,
    integ: &Integrator,
) {
    let k = integ.k;
    let d = g.size1;
    let id = DenseMat::identity(d);
    let mut adjbuf = vec![Complex::ZERO; g.size1 * g.size2];

    // G^R(n, n) = -i * Id
    for n in 0..=k {
        for a in 0..d {
            g.ret.raw_block_at_mut(n, n)[a * d + a] = Complex::new(0.0, -1.0);
        }
    }

    // determine G^R(n, j), n = j+1, .., k
    for j in 0..k {
        let dim = (k - j) as usize;
        let mut ws = BlockSolver::new(dim, d);
        let mut mblocks: Vec<Vec<DenseMat>> = vec![vec![DenseMat::zeros(d); dim]; dim];
        let mut qblocks: Vec<DenseMat> = vec![DenseMat::zeros(d); dim];

        for n in (j + 1)..=k {
            let p = (n - j - 1) as usize; // 0-based block row
            let mut qp = DenseMat::zeros(d);
            for l in 0..=k {
                if l <= j {
                    // known G^R(l, j) = -G^R(j, l)^dagger, it goes into Q
                    adjoint_block_into(g.ret.raw_block_at(j, l), &mut adjbuf, g.size2, g.size1);
                    let gtemp: Vec<Complex> = adjbuf.iter().map(|z| -*z).collect();
                    let cw = -Complex::I / Complex::new(hstep, 0.0)
                        * Complex::new(integ.pdw.at(n as usize, l as usize), 0.0);
                    axpy_c(&mut qp, cw, &DenseMat::from_slice(d, d, &gtemp));
                    // h * XIW[j,n,l] * (Sigma^R(n,l) * gtemp)
                    let w = hstep * integ.xiw.at(j as usize, n as usize, l as usize);
                    let prod = DenseMat::from_slice(d, d, sigma.ret.raw_block_at(n, l))
                        .mat_mul(&DenseMat::from_slice(d, d, &gtemp));
                    axpy_c(&mut qp, Complex::new(w, 0.0), &prod);
                } else {
                    // unknown G^R(l, j), it goes into M
                    let o = (l - j - 1) as usize;
                    let cw = Complex::I / Complex::new(hstep, 0.0)
                        * Complex::new(integ.pdw.at(n as usize, l as usize), 0.0);
                    let mut mpo = scale_c(&id, cw);
                    if l == n {
                        axpy_c(&mut mpo, Complex::new(mu, 0.0), &id);
                        axpy_c(
                            &mut mpo,
                            Complex::new(-1.0, 0.0),
                            &DenseMat::from_slice(d, d, h.block(l)),
                        );
                    }
                    let (sblk, s) = sigma_ret(sigma, n, l);
                    let w = s * hstep * integ.xiw.at(j as usize, n as usize, l as usize);
                    axpy_c(
                        &mut mpo,
                        Complex::new(-w, 0.0),
                        &DenseMat::from_slice(d, d, &sblk),
                    );
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
/// `0 <= n <= k`; the retarded component at the first k + 1 time steps
/// must be known.
#[allow(clippy::too_many_arguments)]
pub fn dyson_start_lmix(
    g: &mut ContourGreen,
    mu: f64,
    h: &TimeFunction,
    sigma: &ContourGreen,
    beta: f64,
    hstep: f64,
    integ: &Integrator,
) {
    let k = integ.k;
    let d = g.size1;
    let ntau = g.ntau; // intervals; points = ntau + 1
    let sign = g.sign;
    let np = ntau as i64; // max tau index

    // initial value: G^rangle(0, tau) = i xi G^M(beta - tau)
    let isign = Complex::new(0.0, sign.as_i64() as f64);
    for m in 0..=np {
        for q in 0..d * d {
            let v = g.mat.block((np - m) as usize)[q];
            g.lmix.raw_block_at_mut(0, m)[q] = isign * v;
        }
    }

    // source term: delta_tau int dtau' Sigma^rangle(n,tau') G^M(tau'-tau)
    for n in 1..=k {
        conv_lmix_mat(n, &mut g.lmix, &sigma.lmix, &g.mat, integ, beta, sign);
    }

    // block system sum_q M_{pq} X_q = Q_p with X_q = G^rangle(q, tau);
    // the matrix M does not depend on tau
    let mcols = d * (ntau + 1);
    let mut ws = BlockSolver::new_rect(k as usize, d, mcols);
    let id = DenseMat::identity(d);
    let mut mblocks: Vec<Vec<DenseMat>> = vec![vec![DenseMat::zeros(d); k as usize]; k as usize];
    let mut qblocks: Vec<DenseMat> = vec![DenseMat::zeros_shape(d, mcols); k as usize];
    for n in 1..=k {
        let p = (n - 1) as usize;
        let mut qn = DenseMat::zeros_shape(d, mcols);
        for m in 0..=np {
            let g0m = g.lmix.raw_block_at(0, m);
            // integral Sigma^rangle * G^M (the accumulated source)
            let src = g.lmix.raw_block_at(n, m);
            // derivative, j = 0
            let cw = -Complex::I / Complex::new(hstep, 0.0)
                * Complex::new(integ.pdw.at(n as usize, 0), 0.0);
            // integral Sigma^R * G^rangle, j = 0
            let w0 = hstep * integ.giw.at(n as usize, 0);
            let prod = DenseMat::from_slice(d, d, sigma.ret.raw_block_at(n, 0))
                .mat_mul(&DenseMat::from_slice(d, d, g0m));
            for a in 0..d {
                for b in 0..d {
                    let mut v = src[a * d + b] + cw * g0m[a * d + b];
                    v += Complex::new(w0, 0.0) * prod.raw()[a * d + b];
                    qn.set(a, m as usize * d + b, v);
                }
            }
        }
        qblocks[p] = qn;
        for j in 1..=k {
            let cw = Complex::I / Complex::new(hstep, 0.0)
                * Complex::new(integ.pdw.at(n as usize, j as usize), 0.0);
            let mut mnj = scale_c(&id, cw);
            let (sblk, s) = sigma_ret(sigma, n, j);
            let w = s * hstep * integ.giw.at(n as usize, j as usize);
            axpy_c(
                &mut mnj,
                Complex::new(-w, 0.0),
                &DenseMat::from_slice(d, d, &sblk),
            );
            if n == j {
                axpy_c(&mut mnj, Complex::new(mu, 0.0), &id);
                axpy_c(
                    &mut mnj,
                    Complex::new(-1.0, 0.0),
                    &DenseMat::from_slice(d, d, h.block(n)),
                );
            }
            mblocks[p][(j - 1) as usize] = mnj;
        }
    }
    let mut xblocks: Vec<DenseMat> = vec![DenseMat::zeros_shape(d, mcols); k as usize];
    solve_block_right(&mut xblocks, &mblocks, &qblocks, &mut ws);
    for n in 1..=k {
        for m in 0..=np {
            for a in 0..d {
                for b in 0..d {
                    g.lmix.raw_block_at_mut(n, m)[a * d + b] =
                        xblocks[(n - 1) as usize].get(a, m as usize * d + b);
                }
            }
        }
    }
}

/// Start-up of the lesser component `G^<(j, n)` for `0 <= j <= n <= k`;
/// the retarded and left-mixing components at the first k + 1 time
/// steps must be known.
#[allow(clippy::too_many_arguments)]
pub(crate) fn dyson_start_less(
    g: &mut ContourGreen,
    mu: f64,
    h: &TimeFunction,
    sigma: &ContourGreen,
    beta: f64,
    hstep: f64,
    integ: &Integrator,
) {
    let k = integ.k;
    let d = g.size1;
    let ntau = g.ntau;

    // n = 0, .., k-1: the private scratch-target convolutions provide
    // the source blocks j = 0, .., k, including j > n
    let mut ws = ConvWorkspace::new(sigma.size1, sigma.size2, g.size1, g.size2);
    for n in 0..k {
        let mut buf: Vec<Vec<Complex>> = vec![vec![Complex::ZERO; d * d]; (k + 1) as usize];
        incr_lmix_rmix_buf(
            &mut buf,
            n,
            sigma,
            None,
            g,
            integ,
            g.sign,
            beta,
            hstep,
            Complex::ONE,
            &mut ws,
        );
        incr_less_adv_buf(
            &mut buf,
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
            &mut ws,
            false,
        );
        dyson_less_core(n, &mut buf, g, mu, h, sigma, hstep, integ, None);
    }

    // n = k: the public slice-target convolutions
    let cshape = Contour::new(k, ntau as i64, d as i64, d as i64, 1.0, beta);
    let mut w = GreenTimeStep::new(&cshape, k, g.sign, Complex::ZERO);
    incr_convolution_lmix_rmix_step(
        k,
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
    incr_convolution_less_adv_step(
        k,
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
    let mut gles: Vec<Vec<Complex>> = (0..=k)
        .map(|j| w.less.raw_block_at(j, k).to_vec())
        .collect();
    dyson_less_core(k, &mut gles, g, mu, h, sigma, hstep, integ, None);
}
