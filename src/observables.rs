// Project : KadanoffBaym
// File    : src/observables.rs
// Purpose : observables layer (blueprint: julia/src/observables/)

use crate::complex::c64::C64;
use crate::convolution::density_matrix::convolution_density_matrix_herm;
use crate::mat::DenseMat;
use crate::types::green::{ContourGreen, GreenTimeStep, TimeFunction};
use crate::types::MATSUBARA;
use crate::weights::Integrator;

/// Density matrix of a contour-ordered Green's function (blueprint
/// `density_matrix`): `-G^M(beta)` at equilibrium (`tstp == MATSUBARA`),
/// `i * sign * G^<(t,t)` at real time `tstp >= 0`.
pub fn density_matrix_green(g: &ContourGreen, tstp: i64) -> DenseMat {
    if tstp == MATSUBARA {
        let blk = g.mat.block(g.ntau);
        let mut out = DenseMat::zeros_shape(g.size1, g.size2);
        for q in 0..blk.len() {
            out.raw_mut()[q] = -blk[q];
        }
        out
    } else {
        assert!((0..=g.nt as i64).contains(&tstp));
        let blk = g.less.raw_block_at(tstp, tstp);
        let s = C64::new(0.0, g.sign.as_i64() as f64);
        let mut out = DenseMat::zeros_shape(g.size1, g.size2);
        for q in 0..blk.len() {
            out.raw_mut()[q] = s * blk[q];
        }
        out
    }
}

/// `density_matrix` for a timestep (𝒻) object.
pub fn density_matrix_step(g: &GreenTimeStep, tstp: i64) -> DenseMat {
    if tstp == MATSUBARA {
        let blk = g.mat.block(g.ntau);
        let mut out = DenseMat::zeros_shape(g.size1, g.size2);
        for q in 0..blk.len() {
            out.raw_mut()[q] = -blk[q];
        }
        out
    } else {
        assert!(tstp >= 0 && tstp <= g.tstp as i64);
        let blk = g.less.raw_block_at(tstp, tstp);
        let s = C64::new(0.0, g.sign.as_i64() as f64);
        let mut out = DenseMat::zeros_shape(g.size1, g.size2);
        for q in 0..blk.len() {
            out.raw_mut()[q] = s * blk[q];
        }
        out
    }
}

/// Alias of `density_matrix` (blueprint `density`).
pub fn density(g: &ContourGreen, tstp: i64) -> DenseMat {
    density_matrix_green(g, tstp)
}

/// Kinetic energy `Re Tr[H rho]` for constant `H`.
pub fn kinetic_energy(h: &DenseMat, rho: &DenseMat) -> f64 {
    assert_eq!((h.nrows(), h.ncols()), (rho.nrows(), rho.ncols()));
    let p = h.mat_mul(rho);
    let mut tr = C64::ZERO;
    for i in 0..h.nrows() {
        tr += p.get(i, i);
    }
    tr.re
}

/// Kinetic energy with a time-dependent Hamiltonian: uses `H[tstp]`.
pub fn kinetic_energy_cf(h: &TimeFunction, g: &ContourGreen, tstp: i64) -> f64 {
    let idx = if tstp == MATSUBARA { MATSUBARA } else { tstp };
    let hm = DenseMat::from_vec(h.size1, h.size2, h.block(idx).to_vec());
    kinetic_energy(&hm, &density_matrix_green(g, tstp))
}

/// Correlation energy `E_corr = 1/2 Re Tr[rho_{Sigma G}]` with
/// `rho_{Sigma G} = convolution_density_matrix(Sigma, G)`
/// (blueprint `correlation_energy`).
pub fn correlation_energy(
    tstp: i64,
    g: &ContourGreen,
    sigma: &ContourGreen,
    integ: &Integrator,
    beta: f64,
    h: f64,
) -> f64 {
    assert!(beta > 0.0);
    assert!(h > 0.0);
    let rho = convolution_density_matrix_herm(tstp, sigma, None, g, integ, beta, h);
    let mut tr = C64::ZERO;
    for i in 0..rho.nrows() {
        tr += rho.get(i, i);
    }
    0.5 * tr.re
}

/// Discrete k sum of the current `J = sum_k w_k Re Tr[v_k rho_k]`
/// (blueprint `current`).
pub fn current(v: &[DenseMat], rho: &[DenseMat], w: &[f64]) -> f64 {
    assert_eq!(v.len(), rho.len());
    assert_eq!(v.len(), w.len());
    let mut j = 0.0;
    for k in 0..v.len() {
        let p = v[k].mat_mul(&rho[k]);
        let mut tr = C64::ZERO;
        for i in 0..v[k].nrows() {
            tr += p.get(i, i);
        }
        j += w[k] * tr.re;
    }
    j
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::equilibrium::from_ham;
    use crate::types::contour::{Contour, Sign};

    #[test]
    fn free_density_equals_fermi() {
        // H = diag(-0.4, 0.6), mu = 0: rho = diag(f(-0.4), f(0.6)) with
        // f the Fermi function at beta
        let beta = 2.0;
        let c = Contour::new(4, 20, 2, 2, 1.0, beta);
        let mut g = ContourGreen::new(&c, Sign::Fermi, C64::ZERO);
        let mut h0 = DenseMat::zeros(2);
        h0.set(0, 0, C64::new(-0.4, 0.0));
        h0.set(1, 1, C64::new(0.6, 0.0));
        from_ham(&mut g, 0.0, &h0, beta, 0.25);
        let rho = density_matrix_green(&g, MATSUBARA);
        let f1 = 1.0 / (1.0 + (-0.4 * beta).exp());
        let f2 = 1.0 / (1.0 + (0.6 * beta).exp());
        assert!((rho.get(0, 0).re - f1).abs() < 1e-12);
        assert!((rho.get(1, 1).re - f2).abs() < 1e-12);
        assert!(rho.get(0, 1).abs() < 1e-14);
        // real time branch equals the Matsubara one (equilibrium)
        let rho_t = density_matrix_green(&g, 3);
        assert!((rho_t.get(0, 0) - rho.get(0, 0)).abs() < 1e-12);
        assert!((rho_t.get(1, 1) - rho.get(1, 1)).abs() < 1e-12);
    }

    #[test]
    fn kinetic_energy_traces() {
        let mut h = DenseMat::zeros(2);
        h.set(0, 0, C64::new(-0.4, 0.0));
        h.set(1, 1, C64::new(0.6, 0.0));
        let rho = DenseMat::from_vec(
            2,
            2,
            vec![C64::new(0.4, 0.0), C64::ZERO, C64::ZERO, C64::new(0.6, 0.0)],
        );
        let e = kinetic_energy(&h, &rho);
        assert!((e - (-0.4 * 0.4 + 0.6 * 0.6)).abs() < 1e-14);
        let mut v = DenseMat::zeros(2);
        v.set(0, 1, C64::ONE);
        let j = current(&[v.clone()], &[rho], &[2.0]);
        assert!(j.abs() < 1e-14); // tr(v rho) = 0 for the diagonal rho
    }
}

/// Truncated kinetic (or correlation) energy integral at the leading
/// row `tstp` (blueprint `kinetic_energy_trunc`),
/// `E = Im Tr h sum_l w_l [G^{R}(0,l)^dagger D^{<}(0,l)
/// - G^{<}(0,l)^dagger D^{R}(0,l)]` with the Gregory end weights at
/// both ends and the weight 1 in between.
pub fn kinetic_energy_trunc(
    tstp: i64,
    g: &crate::types::moving::MovingGreen,
    dmat: &crate::types::moving::MovingGreen,
    integ: &crate::weights::Integrator,
    h: f64,
) -> f64 {
    let k = integ.k;
    crate::moving::layout::check_moving_solver(tstp, k, &[g, dmat]);
    let tc = g.tc;
    let dd = g.ret.size1;
    let mut s = crate::mat::DenseMat::zeros(dd);
    for l in 0..=tc {
        let w = if l <= k {
            integ.giw.omega[l as usize]
        } else if l >= tc - k {
            integ.giw.omega[(tc - l) as usize]
        } else {
            1.0
        };
        let gr = DenseMat::from_vec(dd, dd, g.ret.mblock(0, l).to_vec()).adjoint();
        let dl = DenseMat::from_vec(dd, dd, dmat.less.mblock(0, l).to_vec());
        let p1 = gr.mat_mul(&dl);
        let gl = DenseMat::from_vec(dd, dd, g.less.mblock(0, l).to_vec()).adjoint();
        let dr = DenseMat::from_vec(dd, dd, dmat.ret.mblock(0, l).to_vec());
        let p2 = gl.mat_mul(&dr);
        let cw = crate::complex::c64::C64::new(h * w, 0.0);
        for q in 0..(dd * dd) {
            let v = cw * (p1.raw()[q] - p2.raw()[q]);
            s.raw_mut()[q] += v;
        }
    }
    let mut tr = crate::complex::c64::C64::ZERO;
    for a in 0..dd {
        tr += s.get(a, a);
    }
    tr.im
}

// ------------------------------------------------------------------
// Wigner slices and windowed spectral functions
// ------------------------------------------------------------------
//
// Blueprint julia/src/observables/spectrum.jl. The time-dependent
// spectral function is evaluated in the Wigner representation.

/// Extract the retarded and lesser components at the fixed average
/// time step `tav` (0-based) as functions of the relative time
/// (blueprint `wigner_slices`): for `j = 0, ..., min(tav, nt - tav)`
/// the times are `t1 = tav + j` and `t2 = tav - j`.
pub fn wigner_slices(g: &ContourGreen, tav: i64, h: f64) -> (Vec<f64>, Vec<Vec<C64>>, Vec<Vec<C64>>) {
    let d1 = g.ret.size1;
    let d2 = g.ret.size2;
    assert!(d1 == d2, "the matrices of G must be square, got {d1} x {d2}");
    let nt = g.nt as i64;
    assert!((0..=nt).contains(&tav), "tav must be in 0..={nt}, got {tav}");
    assert!(h > 0.0, "h must be positive, got {h}");

    let nj = (tav.min(nt - tav) + 1) as usize;
    let t_rel: Vec<f64> = (0..nj).map(|j| 2.0 * j as f64 * h).collect();
    let mut ret = Vec::with_capacity(nj);
    let mut les = Vec::with_capacity(nj);
    for j in 0..nj as i64 {
        let t1 = tav + j;
        let t2 = tav - j;
        ret.push(g.ret_block(t1, t2).raw().to_vec());
        les.push(g.less_block(t1, t2).raw().to_vec());
    }
    (t_rel, ret, les)
}

/// Fourier integral `integral dt e^{i w t} v(t)` of the piecewise
/// linear interpolation of `v` on the equidistant grid `t` (blueprint
/// `_linear_fourier`). At `w = 0` the trapezoidal rule is used.
pub fn linear_fourier(v: &[C64], t: &[f64], w: f64) -> C64 {
    let nn = v.len() - 1;
    let dt = t[1] - t[0];
    if w == 0.0 {
        let mut res = 0.5 * (v[0] + v[nn]);
        for it in 1..nn {
            res += v[it];
        }
        return res * dt;
    }
    let mut res = C64::ZERO;
    for it in 0..nn {
        res += v[it] * C64::from_polar(1.0, w * t[it]);
    }
    let s = (w * dt / 2.0).sin();
    res *= C64::new(4.0 * s * s / w / w / dt, 0.0);
    let cis_t = |x: f64| C64::from_polar(1.0, x);
    res += (v[nn] * cis_t(w * t[nn]) - v[0] * cis_t(w * t[0]))
        * (C64::new(0.0, -1.0 / w) - (cis_t(-w * dt) - C64::ONE) / (w * w * dt));
    res
}

/// Method of the windowed spectral function (blueprint `:cubic` /
/// `:linear`).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SpectrumMethod {
    Cubic,
    Linear,
}

/// Windowed spectral function of the diagonal elements
/// (blueprint `evaluate_spectrum_windowed`),
/// `A_ii(w) = -1/pi Im integral dt e^{i w t} W(t) G^R_ii(t)`, on the
/// arbitrary frequencies `omega` over the equidistant grid `t_rel`.
/// `gret` holds one `d x d` block per time point (see
/// `wigner_slices`).
pub fn evaluate_spectrum_windowed(
    gret: &[Vec<C64>],
    t_rel: &[f64],
    omega: &[f64],
    method: SpectrumMethod,
    window: &dyn Fn(&[f64]) -> Vec<C64>,
) -> Vec<Vec<f64>> {
    let nt = gret.len();
    let b = gret[0].len();
    let d = (b as f64).sqrt() as usize;
    assert!(d * d == b, "gret must contain square matrices");
    assert_eq!(t_rel.len(), nt, "t_rel must have {nt} points");
    assert!(nt >= 2, "at least two time points are needed");
    let nn = nt - 1;
    if method == SpectrumMethod::Cubic {
        assert!(
            nn > 16 && nn % 2 == 0,
            ":cubic needs an even number (> 16) of intervals, got {nn}"
        );
    }
    let dt = t_rel[1] - t_rel[0];
    assert!(dt > 0.0, "t_rel must be increasing");
    for i in 0..nn {
        assert!(
            (t_rel[i + 1] - t_rel[i] - dt).abs() <= 1e-10 * dt.abs(),
            "t_rel must be equidistant"
        );
    }
    assert!(
        omega.iter().all(|w| w.is_finite()),
        "omega must be finite"
    );

    let w = window(t_rel);
    assert_eq!(w.len(), nt, "window(t_rel) must return a vector of the length {nt}");

    let mut a_out = vec![vec![0.0f64; d]; omega.len()];
    let mut v = vec![C64::ZERO; nt];
    for i in 0..d {
        for (q, item) in v.iter_mut().enumerate() {
            *item = gret[q][i * d + i] * w[q];
        }
        for (k, om) in omega.iter().enumerate() {
            let res = if method == SpectrumMethod::Cubic {
                crate::fourier::adft::dft_cplx(*om, t_rel[0], t_rel[nt - 1], &v).0
            } else {
                linear_fourier(&v, t_rel, *om)
            };
            a_out[k][i] = -res.im / std::f64::consts::PI;
        }
    }
    a_out
}

// ------------------------------------------------------------------
// steady-state density matrices
// ------------------------------------------------------------------

use crate::types::contour::Sign;

fn c(re: f64, im: f64) -> C64 {
    C64::new(re, im)
}

/// Density matrix `rho = xi i G^<(0)` of a steady-state Green's
/// function (blueprint `density_matrix(G::NessHermMat, sign)`).
pub fn density_matrix_ness(g: &crate::types::ness::NessHermMat, sign: Sign) -> Vec<C64> {
    let blk = g.les.block(crate::types::ness::FftDomain::Time, 0).to_vec();
    let xi = c(0.0, sign.as_i64() as f64);
    blk.into_iter().map(|z| xi * z).collect()
}

/// Density matrix `xi i C^<(0)` of the steady-state convolution
/// `C = A * B` with `C^<(w) = A^<(w) B^A(w) + A^R(w) B^<(w)`
/// (blueprint `convolution_density_matrix(A::NessHermMat, ...)`). The
/// inputs are transformed on local copies and not changed.
pub fn convolution_density_matrix_ness(
    a: &crate::types::ness::NessHermMat,
    b: &crate::types::ness::NessHermMat,
    sign: Sign,
) -> Vec<C64> {
    assert!(
        a.is_compatible(b),
        "convolution_density_matrix: input mismatch"
    );
    let mut ac = a.clone();
    let mut bc = b.clone();
    crate::fourier::transforms::integral_transform_to_freq(
        &mut ac,
        crate::fourier::transforms::FftIntegralMethod::Trapez,
    );
    crate::fourier::transforms::integral_transform_to_freq(
        &mut bc,
        crate::fourier::transforms::FftIntegralMethod::Trapez,
    );
    let d = a.ret.size1;
    let bsz = d * d;
    let nft = a.grid.nft;
    let mut cc = vec![C64::ZERO; bsz];
    for k in 0..nft as usize {
        let al = &ac.les.freq[k * bsz..(k + 1) * bsz];
        let ar = &ac.ret.freq[k * bsz..(k + 1) * bsz];
        let bl = &bc.les.freq[k * bsz..(k + 1) * bsz];
        let br = &bc.ret.freq[k * bsz..(k + 1) * bsz];
        // C += A^< (B^R)^dagger + A^R B^<
        for i in 0..d {
            for j in 0..d {
                let mut acc = C64::ZERO;
                for m in 0..d {
                    acc += al[i * d + m] * br[j * d + m].conj();
                    acc += ar[i * d + m] * bl[m * d + j];
                }
                cc[i * d + j] += acc;
            }
        }
    }
    let fac = c(0.0, sign.as_i64() as f64 * a.grid.dw * 0.5 / std::f64::consts::PI);
    cc.into_iter().map(|z| fac * z).collect()
}
