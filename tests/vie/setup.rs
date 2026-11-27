// Project : KadanoffBaym
// File    : tests/vie/setup.rs
// Purpose : exact VIE2 fixtures and reference helpers for the VIE2 tests
//

pub(crate) use kadanoffbaym::base::MatsubaraMethod;
pub(crate) use kadanoffbaym::linalg::Adjoint;
pub(crate) use kadanoffbaym::vie2::Vie2SingularWorkspace;
pub(crate) use kadanoffbaym::weights::Integrator;

pub(crate) use kadanoffbaym::convolution::{conv_mat, convolution_ab};
pub(crate) use kadanoffbaym::equilibrium::from_ham;
pub(crate) use kadanoffbaym::linalg::mat_inv;
pub(crate) use kadanoffbaym::storage::timestep_at;
pub(crate) use kadanoffbaym::vie2::{
    vie2, vie2_mat, vie2_start, vie2_timestep, vie2_timestep_less, vie2_timestep_lmix,
    vie2_timestep_ret, vie2_timestep_singular,
};

pub(crate) use super::util::{
    c, check_order_gate, constant_time_function, is_release, test_hamiltonian_a,
    test_hamiltonian_b, test_interaction,
};

pub(crate) use super::util::prelude::*;

// ------------------------------------------------------------------
// helper types
// ------------------------------------------------------------------

pub(crate) struct VtSetup {
    pub(crate) g: ContourGreen,
    pub(crate) f: ContourGreen,
    pub(crate) fcc: ContourGreen,
    pub(crate) q: ContourGreen,
    pub(crate) contour: Contour,
    pub(crate) beta: f64,
    pub(crate) h: f64,
    pub(crate) nt: i64,
    pub(crate) sign: Sign,
}

// ------------------------------------------------------------------
// helpers
// ------------------------------------------------------------------

/// Errors of the ret, tv, and les components per element; the
/// Matsubara component is taken from the exact solution (reference
/// `errors`).
pub(crate) fn errors(ks: i64, nt: i64, sign: Sign) -> (f64, f64, f64) {
    let s = setup(nt, ks, sign);
    let mut g = ContourGreen::new(&s.contour, s.sign, Complex::ZERO);
    g.mat.raw_mut().copy_from_slice(s.g.mat.raw());
    let ik = Integrator::new(ks);
    vie2_start(&mut g, &s.f, &s.fcc, &s.q, ks, s.beta, s.h);
    for tstp in (ks + 1)..=s.nt {
        vie2_timestep(
            tstp,
            &mut g,
            &s.f,
            &s.fcc,
            &s.q,
            &ik,
            s.beta,
            s.h,
            MatsubaraMethod::Fixpoint,
        );
    }
    let ntau = g.ntau as i64;
    let mut er = 0.0f64;
    let mut et = 0.0f64;
    let mut el = 0.0f64;
    for n in 0..=s.nt {
        er = er.max(g.distance_ret(&s.g, n) / (n + 1) as f64);
        et = et.max(g.distance_lmix(&s.g, n) / (ntau + 1) as f64);
        el = el.max(g.distance_less(&s.g, n) / (n + 1) as f64);
    }
    (er, et, el)
}

pub(crate) fn same_green(a: &ContourGreen, b: &ContourGreen) -> bool {
    a.mat.raw() == b.mat.raw()
        && a.ret.raw() == b.ret.raw()
        && a.lmix.raw() == b.lmix.raw()
        && a.less.raw() == b.less.raw()
}

/// Exact solution G, kernel F, partner Fcc, and source Q; the kernels
/// are evaluated by the convolution of order `ks` (reference
/// `setup`). `nt` is the number of real time intervals.
pub(crate) fn setup(nt: i64, ks: i64, sign: Sign) -> VtSetup {
    let (ntau, beta, tmax) = (200i64, 1.0, 2.0);
    let mu = if sign == Sign::Fermi { 0.1 } else { -2.5 };
    let h = tmax / nt as f64;
    let c4 = Contour::new(nt, ntau, 4, 4, tmax, beta);
    let c2 = Contour::new(nt, ntau, 2, 2, tmax, beta);
    let (ha, hb, v) = (
        test_hamiltonian_a(),
        test_hamiltonian_b(),
        test_interaction(),
    );
    let mut hfull = DenseMat::zeros(4);
    for i in 0..4 {
        for j in 0..4 {
            let x = if i < 2 && j < 2 {
                ha.get(i, j)
            } else if i < 2 {
                v.get(i, j - 2)
            } else if j < 2 {
                v.get(j, i - 2).conj()
            } else {
                hb.get(i - 2, j - 2)
            };
            hfull.set(i, j, x);
        }
    }
    let mut gf = ContourGreen::new(&c4, sign, Complex::ZERO);
    from_ham(&mut gf, mu, &hfull, beta, h);
    let mut gx = ContourGreen::new(&c2, sign, Complex::ZERO);
    // upper-left 2 x 2 block of each 4 x 4 block (strided extraction)
    for m in 0..=(ntau as usize) {
        for a in 0..2usize {
            for b in 0..2usize {
                gx.mat.block_mut(m)[a * 2 + b] = gf.mat.block(m)[a * 4 + b];
            }
        }
    }
    for i in 0..=nt {
        for m in 0..=(ntau as usize) {
            for a in 0..2usize {
                for b in 0..2usize {
                    gx.lmix.raw_block_at_mut(i, m as i64)[a * 2 + b] =
                        gf.lmix.raw_block_at(i, m as i64)[a * 4 + b];
                }
            }
        }
        for j in 0..=i {
            for a in 0..2usize {
                for b in 0..2usize {
                    gx.ret.raw_block_at_mut(i, j)[a * 2 + b] = gf.ret.raw_block_at(i, j)[a * 4 + b];
                    gx.less.raw_block_at_mut(j, i)[a * 2 + b] =
                        gf.less.raw_block_at(j, i)[a * 4 + b];
                }
            }
        }
    }
    let mut gb = ContourGreen::new(&c2, sign, Complex::ZERO);
    from_ham(&mut gb, mu, &hb, beta, h);
    let vadj = v.adjoint();
    let mut sigma = ContourGreen::new(&c2, sign, Complex::ZERO);
    let put = |blk: &[Complex], out: &mut [Complex]| {
        let src = DenseMat::from_vec(2, 2, blk.to_vec());
        let s2 = v.mat_mul(&src).mat_mul(&vadj);
        out.copy_from_slice(s2.raw());
    };
    for m in 0..=(ntau as usize) {
        put(gb.mat.block(m), sigma.mat.block_mut(m));
    }
    for i in 0..=nt {
        for m in 0..=(ntau as usize) {
            put(
                gb.lmix.raw_block_at(i, m as i64),
                sigma.lmix.raw_block_at_mut(i, m as i64),
            );
        }
        for j in 0..=i {
            put(gb.ret.raw_block_at(i, j), sigma.ret.raw_block_at_mut(i, j));
            put(
                gb.less.raw_block_at(j, i),
                sigma.less.raw_block_at_mut(j, i),
            );
        }
    }
    let mut qg = ContourGreen::new(&c2, sign, Complex::ZERO);
    from_ham(&mut qg, mu, &ha, beta, h);
    let ik = Integrator::new(ks);
    let mut f = ContourGreen::new(&c2, sign, Complex::ZERO);
    let mut fcc = ContourGreen::new(&c2, sign, Complex::ZERO);
    convolution_ab(&mut f, &qg, &sigma, &ik, beta, h);
    convolution_ab(&mut fcc, &sigma, &qg, &ik, beta, h);
    f.smul(c(-1.0, 0.0));
    fcc.smul(c(-1.0, 0.0));
    VtSetup {
        g: gx,
        f,
        fcc,
        q: qg,
        contour: c2,
        beta,
        h,
        nt,
        sign,
    }
}

/// GW-like calling pattern ; `nt = 12`.
pub(crate) fn sin_pattern<F>(k: i64, sign: Sign, mut call: F) -> (ContourGreen, TimeFunction)
where
    F: FnMut(
        i64,
        &mut ContourGreen,
        &mut TimeFunction,
        &ContourGreen,
        &ContourGreen,
        &TimeFunction,
        &ContourGreen,
        &TimeFunction,
        &Integrator,
        f64,
        f64,
    ),
{
    let s = setup(12, k, sign);
    let nt = s.nt;
    let (mut f, mut fcc, mut q) = (s.f.clone(), s.fcc.clone(), s.q.clone());
    let mut fs = TimeFunction::new(nt as usize, 2, 2, Complex::ZERO);
    let mut qs = TimeFunction::new(nt as usize, 2, 2, Complex::ZERO);
    for t in MATSUBARA..=nt {
        let ijl = (t + 1) as f64;
        let fs_blk = [
            c(0.3 + 0.01 * ijl, 0.0),
            c(0.0, 0.1),
            c(0.0, -0.1),
            c(-0.2, 0.0),
        ];
        let qs_blk = [
            c(0.5, 0.0),
            c(0.2 + 0.01 * ijl, 0.0),
            c(-0.3, 0.0),
            c(-0.4, 0.0),
        ];
        for a in 0..2usize {
            for b in 0..2usize {
                fs.set(t, a, b, fs_blk[a * 2 + b]);
                qs.set(t, a, b, qs_blk[a * 2 + b]);
            }
        }
    }
    let mut g = ContourGreen::new(&s.contour, sign, Complex::ZERO);
    let mut gs = TimeFunction::new(nt as usize, 2, 2, Complex::ZERO);
    let ik = Integrator::new(k);
    let id = DenseMat::identity(2);

    // repeated calls at tstp = 0 (the Matsubara axis)
    for it in 1..=2i64 {
        for x in [&mut f, &mut fcc, &mut q] {
            x.smul_at(MATSUBARA, c(1.0 + 0.01 * it as f64, 0.0));
        }
        call(
            MATSUBARA, &mut g, &mut gs, &f, &fcc, &fs, &q, &qs, &ik, s.beta, s.h,
        );
    }
    // repeated start-up with modified inputs at tstp = 1, .., k + 1
    for it in 1..=2i64 {
        for t in 0..=k {
            for x in [&mut f, &mut fcc, &mut q] {
                x.smul_at(t, c(1.0 + 0.01 * it as f64, 0.0));
            }
        }
        for a in 0..2usize {
            for b in 0..2usize {
                let v = fs.block(k)[a * 2 + b] + c(0.001 * it as f64, 0.0) * id.get(a, b);
                fs.set(k, a, b, v);
            }
        }
        for tstp in 0..=k {
            call(
                tstp, &mut g, &mut gs, &f, &fcc, &fs, &q, &qs, &ik, s.beta, s.h,
            );
        }
    }
    // repeated calls at each later time step with modified inputs
    for tstp in (k + 1)..=nt {
        for it in 1..=2i64 {
            for x in [&mut f, &mut fcc, &mut q] {
                x.smul_at(tstp, c(1.0 + 0.01 * it as f64, 0.0));
            }
            for a in 0..2usize {
                for b in 0..2usize {
                    let v = fs.block(tstp)[a * 2 + b] + c(0.001 * it as f64, 0.0) * id.get(a, b);
                    fs.set(tstp, a, b, v);
                }
            }
            call(
                tstp, &mut g, &mut gs, &f, &fcc, &fs, &q, &qs, &ik, s.beta, s.h,
            );
        }
    }
    (g, gs)
}

/// Full-copy implementation of the singular VIE2 step: F, Fcc, and Q
/// are transformed at the time steps MATSUBARA, .., n2 in every call.
#[allow(clippy::too_many_arguments)]
pub(crate) fn sin_reference(
    tstp: i64,
    g: &mut ContourGreen,
    gsing: &mut TimeFunction,
    f: &ContourGreen,
    fcc: &ContourGreen,
    fsing: &TimeFunction,
    q: &ContourGreen,
    qsing: &TimeFunction,
    ik: &Integrator,
    beta: f64,
    h: f64,
) {
    let k = ik.k;
    let n2 = if tstp == MATSUBARA {
        MATSUBARA
    } else if tstp <= k {
        k + 1
    } else {
        tstp
    };
    let id = DenseMat::identity(2);
    let steps: Vec<i64> = {
        let mut v = Vec::new();
        if n2 >= MATSUBARA {
            v.push(MATSUBARA);
        }
        for i in 0..=n2.max(-1) {
            v.push(i);
        }
        v
    };
    let mut finv_blocks: Vec<(i64, DenseMat)> = Vec::new();
    for &i in &steps {
        let mut d = id.clone();
        for q in 0..4 {
            d.raw_mut()[q] += fsing.block(i)[q];
        }
        let dinv = mat_inv(&d);
        let qs = DenseMat::from_vec(2, 2, qsing.block(i).to_vec());
        let gs = dinv.mat_mul(&qs);
        for a in 0..2usize {
            for b in 0..2usize {
                gsing.set(i, a, b, gs.get(a, b));
            }
        }
        finv_blocks.push((i, dinv));
    }
    let mut tf = f.clone();
    let mut tfcc = fcc.clone();
    let mut tq = q.clone();
    for (i, dinv) in &finv_blocks {
        tf.lmul_at(*i, dinv, Adjoint::No);
        tfcc.rmul_at(*i, dinv, Adjoint::No);
        tq.lmul_at(*i, dinv, Adjoint::No);
    }
    let mut tf1 = tf.clone();
    for &(i, _) in &finv_blocks {
        let gsm = DenseMat::from_vec(2, 2, gsing.block(i).to_vec());
        tf1.rmul_at(i, &gsm, Adjoint::No);
        let step = timestep_at(&tf1, i.max(0));
        tq.incr_by_step(i, &step, c(-1.0, 0.0));
    }
    vie2_timestep(
        tstp,
        g,
        &tf,
        &tfcc,
        &tq,
        ik,
        beta,
        h,
        MatsubaraMethod::Fixpoint,
    );
}
