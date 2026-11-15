//
// Project : KadanoffBaym
// File    : src/observables/density.rs
// Purpose : density matrices, energies, and currents of the standard
//           and truncated (moving) containers
//
// Blueprint: julia/src/observables/ (density_matrix / density /
// kinetic_energy / correlation_energy / current / kinetic_energy_trunc).

use crate::complex::Complex;
use crate::convolution::convolution_density_matrix_herm;
use crate::linalg::DenseMat;
use crate::moving::check_moving_solver;
use crate::types::MATSUBARA;
use crate::types::{ContourGreen, GreenTimeStep, MovingGreen, TimeFunction};
use crate::weights::Integrator;

/// Density matrix of a contour-ordered Green's function (blueprint
/// `density_matrix`): `-G^M(beta)` at equilibrium (`tstp == MATSUBARA`),
/// `i * sign * G^<(t,t)` at real time `tstp >= 0`.
pub fn density_matrix_green(g: &ContourGreen, tstp: i64) -> DenseMat {
    if tstp == MATSUBARA {
        let blk = g.mat.block(g.ntau);
        let mut out = DenseMat::zeros_shape(g.size1, g.size2);
        for (oq, bq) in out.raw_mut().iter_mut().zip(blk.iter()) {
            *oq = -*bq;
        }
        out
    } else {
        assert!((0..=g.nt as i64).contains(&tstp));
        let blk = g.less.raw_block_at(tstp, tstp);
        let s = Complex::new(0.0, g.sign.as_i64() as f64);
        let mut out = DenseMat::zeros_shape(g.size1, g.size2);
        for (oq, bq) in out.raw_mut().iter_mut().zip(blk.iter()) {
            *oq = s * *bq;
        }
        out
    }
}

#[allow(dead_code)]
/// `density_matrix` for a timestep (𝒻) object.
fn density_matrix_step(g: &GreenTimeStep, tstp: i64) -> DenseMat {
    if tstp == MATSUBARA {
        let blk = g.mat.block(g.ntau);
        let mut out = DenseMat::zeros_shape(g.size1, g.size2);
        for (oq, bq) in out.raw_mut().iter_mut().zip(blk.iter()) {
            *oq = -*bq;
        }
        out
    } else {
        assert!(tstp >= 0 && tstp <= g.tstp as i64);
        let blk = g.less.raw_block_at(tstp, tstp);
        let s = Complex::new(0.0, g.sign.as_i64() as f64);
        let mut out = DenseMat::zeros_shape(g.size1, g.size2);
        for (oq, bq) in out.raw_mut().iter_mut().zip(blk.iter()) {
            *oq = s * *bq;
        }
        out
    }
}

/// Alias of `density_matrix` (blueprint `density`).
pub fn density(g: &ContourGreen, tstp: i64) -> DenseMat {
    density_matrix_green(g, tstp)
}

#[allow(dead_code)]
/// Kinetic energy `Re Tr[H rho]` for constant `H`.
fn kinetic_energy(h: &DenseMat, rho: &DenseMat) -> f64 {
    assert_eq!((h.nrows(), h.ncols()), (rho.nrows(), rho.ncols()));
    let p = h.mat_mul(rho);
    let mut tr = Complex::ZERO;
    for i in 0..h.nrows() {
        tr += p.get(i, i);
    }
    tr.re
}

#[allow(dead_code)]
/// Kinetic energy with a time-dependent Hamiltonian: uses `H[tstp]`.
fn kinetic_energy_cf(h: &TimeFunction, g: &ContourGreen, tstp: i64) -> f64 {
    let idx = if tstp == MATSUBARA { MATSUBARA } else { tstp };
    let hm = DenseMat::from_slice(h.size1, h.size2, h.block(idx));
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
    let mut tr = Complex::ZERO;
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
        let mut tr = Complex::ZERO;
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
    use crate::types::Contour;
    use crate::types::Sign;

    #[test]
    fn free_density_equals_fermi() {
        // H = diag(-0.4, 0.6), mu = 0: rho = diag(f(-0.4), f(0.6)) with
        // f the Fermi function at beta
        let beta = 2.0;
        let c = Contour::new(4, 20, 2, 2, 1.0, beta);
        let mut g = ContourGreen::new(&c, Sign::Fermi, Complex::ZERO);
        let mut h0 = DenseMat::zeros(2);
        h0.set(0, 0, Complex::new(-0.4, 0.0));
        h0.set(1, 1, Complex::new(0.6, 0.0));
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
        h.set(0, 0, Complex::new(-0.4, 0.0));
        h.set(1, 1, Complex::new(0.6, 0.0));
        let rho = DenseMat::from_vec(
            2,
            2,
            vec![
                Complex::new(0.4, 0.0),
                Complex::ZERO,
                Complex::ZERO,
                Complex::new(0.6, 0.0),
            ],
        );
        let e = kinetic_energy(&h, &rho);
        assert!((e - (-0.4 * 0.4 + 0.6 * 0.6)).abs() < 1e-14);
        let mut v = DenseMat::zeros(2);
        v.set(0, 1, Complex::ONE);
        let j = current(&[v.clone()], &[rho], &[2.0]);
        assert!(j.abs() < 1e-14); // tr(v rho) = 0 for the diagonal rho
    }
}

/// Truncated kinetic (or correlation) energy integral at the leading
/// row `tstp` (blueprint `kinetic_energy_trunc`),
/// `E = Im Tr h sum_l w_l [G^{R}(0,l)^dagger D^{<}(0,l) -
/// G^{<}(0,l)^dagger D^{R}(0,l)]` with the Gregory end weights at both
/// ends and the weight 1 in between.
///
/// ```math
/// E = \\mathrm{Im}\\,\\mathrm{Tr}\\, h\\sum_{l=0}^{t_c} w_l
/// \\left[G^{R}(t,t-l)^{\\dagger} D^{<}(t,t-l)
/// - G^{<}(t,t-l)^{\\dagger} D^{R}(t,t-l)\\right],
/// ```
///
/// with the Gregory end weights ``\\omega`` at both ends and the weight 1
/// in between. It requires `tc ≥ 2k + 2`.
///
/// *References* :
///
/// Please see [`NESSi`] `cntr_dyson_moving_impl.hpp`,
/// `get_kinetic_trunc_timestep` (without its debug output).
pub fn kinetic_energy_trunc(
    tstp: i64,
    g: &MovingGreen,
    dmat: &MovingGreen,
    integ: &Integrator,
    h: f64,
) -> f64 {
    let k = integ.k;
    check_moving_solver(tstp, k, &[g, dmat]);
    let tc = g.tc;
    let dd = g.ret.size1;
    let mut s = DenseMat::zeros(dd);
    for l in 0..=tc {
        let w = if l <= k {
            integ.giw.omega[l as usize]
        } else if l >= tc - k {
            integ.giw.omega[(tc - l) as usize]
        } else {
            1.0
        };
        let gr = DenseMat::from_slice(dd, dd, g.ret.mblock(0, l)).adjoint();
        let dl = DenseMat::from_slice(dd, dd, dmat.less.mblock(0, l));
        let p1 = gr.mat_mul(&dl);
        let gl = DenseMat::from_slice(dd, dd, g.less.mblock(0, l)).adjoint();
        let dr = DenseMat::from_slice(dd, dd, dmat.ret.mblock(0, l));
        let p2 = gl.mat_mul(&dr);
        let cw = Complex::new(h * w, 0.0);
        for q in 0..(dd * dd) {
            let v = cw * (p1.raw()[q] - p2.raw()[q]);
            s.raw_mut()[q] += v;
        }
    }
    let mut tr = Complex::ZERO;
    for a in 0..dd {
        tr += s.get(a, a);
    }
    tr.im
}
