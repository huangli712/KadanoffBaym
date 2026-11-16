//
// Project : KadanoffBaym
// File    : src/observables/energy.rs
// Purpose : energy observables: kinetic and correlation energies of
//           the standard and truncated (moving) containers
//
// correlation_energy / kinetic_energy_trunc).

use crate::complex::Complex;
use crate::linalg::DenseMat;
use crate::types::{ContourGreen, MovingGreen, TimeFunction};
use crate::weights::Integrator;

use crate::types::MATSUBARA;

use crate::basics::check_moving_solver;
use crate::convolution::convolution_density_matrix_herm;
use crate::observables::density_matrix;

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
    kinetic_energy(&hm, &density_matrix(g, tstp))
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

#[cfg(test)]
mod tests {
    use super::*;

    use crate::observables::current;

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
