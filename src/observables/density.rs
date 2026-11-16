//
// Project : KadanoffBaym
// File    : src/observables/density.rs
// Purpose : density-matrix observables of all container flavors:
//           standard, steady-state, and pseudo-particle
//
// standard, NessHermMat, and pseudo-particle containers,
// convolution_density_matrix).

use crate::complex::Complex;
use crate::fourier::FftIntegralMethod;
use crate::linalg::DenseMat;
use crate::types::{ContourGreen, GreenTimeStep, PseudoGreen, PseudoGreenTimeStep, SteadyGreen};
use crate::types::{FftDomain, Sign};

use crate::types::MATSUBARA;

use crate::fourier::integral_transform_to_freq;

/// Alias of `density_matrix`.
pub fn density(g: &ContourGreen, tstp: i64) -> DenseMat {
    density_matrix(g, tstp)
}

/// Density matrix of a contour-ordered Green's function : `-G^M(beta)` at equilibrium (`tstp == MATSUBARA`),
/// `i * sign * G^<(t,t)` at real time `tstp >= 0`.
pub fn density_matrix(g: &ContourGreen, tstp: i64) -> DenseMat {
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

/// Density (the trace of the density matrix).
pub fn density_pseudo(g: &PseudoGreen, tstp: i64) -> Complex {
    let rho = density_matrix_pseudo(g, tstp);
    let d = g.size1;
    (0..d).map(|i| rho[i * d + i]).sum()
}

/// Density matrix of the pseudo-particle Green's function : `rho = -G^M(beta)` on the
/// Matsubara axis, `rho = i xi G^{<}(t, t)` at real time.
pub fn density_matrix_pseudo(g: &PseudoGreen, tstp: i64) -> Vec<Complex> {
    let xi = g.sign.as_i64() as f64;
    if tstp == MATSUBARA {
        g.mat.block(g.ntau).iter().map(|z| -*z).collect()
    } else {
        assert!(
            (0..=g.nt as i64).contains(&tstp),
            "density_matrix_pseudo: tstp {tstp} out of range"
        );
        let blk = g.less.raw_block_at(tstp, tstp);
        (0..blk.len())
            .map(|q| Complex::new(0.0, xi) * blk[q])
            .collect()
    }
}

/// Density matrix at a time step object.
pub fn density_matrix_pseudo_step(x: &PseudoGreenTimeStep, tstp: i64) -> Vec<Complex> {
    assert!(
        (x.tstp as i64) == tstp || (tstp == MATSUBARA && x.tstp == 0),
        "the time step object carries tstp {}, not {tstp}",
        x.tstp
    );
    let xi = x.sign.as_i64() as f64;
    if tstp == MATSUBARA {
        x.mat.block(x.ntau).iter().map(|z| -*z).collect()
    } else {
        let blk = x.less.raw_block_at(tstp, tstp);
        (0..blk.len())
            .map(|q| Complex::new(0.0, xi) * blk[q])
            .collect()
    }
}

/// Density matrix `rho = xi i G^<(0)` of a steady-state Green's function.
pub fn density_matrix_steady(g: &SteadyGreen, sign: Sign) -> Vec<Complex> {
    let blk = g.les.block(FftDomain::Time, 0).to_vec();
    let xi = Complex::new(0.0, sign.as_i64() as f64);
    blk.into_iter().map(|z| xi * z).collect()
}

/// Density matrix `xi i C^<(0)` of the steady-state convolution
/// `C = A * B` with `C^<(w) = A^<(w) B^A(w) + A^R(w) B^<(w)`
/// . The
/// inputs are transformed on local copies and not changed.
pub fn convolution_density_matrix_steady(
    a: &SteadyGreen,
    b: &SteadyGreen,
    sign: Sign,
) -> Vec<Complex> {
    assert!(
        a.is_compatible(b),
        "convolution_density_matrix: input mismatch"
    );
    let mut ac = a.clone();
    let mut bc = b.clone();
    integral_transform_to_freq(&mut ac, FftIntegralMethod::Trapez);
    integral_transform_to_freq(&mut bc, FftIntegralMethod::Trapez);
    let d = a.ret.size1;
    let bsz = d * d;
    let nft = a.grid.nft;
    let mut cc = vec![Complex::ZERO; bsz];
    for k in 0..nft as usize {
        let al = &ac.les.freq[k * bsz..(k + 1) * bsz];
        let ar = &ac.ret.freq[k * bsz..(k + 1) * bsz];
        let bl = &bc.les.freq[k * bsz..(k + 1) * bsz];
        let br = &bc.ret.freq[k * bsz..(k + 1) * bsz];
        // C += A^< (B^R)^dagger + A^R B^<
        for i in 0..d {
            for j in 0..d {
                let mut acc = Complex::ZERO;
                for m in 0..d {
                    acc += al[i * d + m] * br[j * d + m].conj();
                    acc += ar[i * d + m] * bl[m * d + j];
                }
                cc[i * d + j] += acc;
            }
        }
    }
    let fac = Complex::new(
        0.0,
        sign.as_i64() as f64 * a.grid.dw * 0.5 / std::f64::consts::PI,
    );
    cc.into_iter().map(|z| fac * z).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::types::{Contour, Sign};

    use crate::equilibrium::from_ham;

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
        let rho = density_matrix(&g, MATSUBARA);
        let f1 = 1.0 / (1.0 + (-0.4 * beta).exp());
        let f2 = 1.0 / (1.0 + (0.6 * beta).exp());
        assert!((rho.get(0, 0).re - f1).abs() < 1e-12);
        assert!((rho.get(1, 1).re - f2).abs() < 1e-12);
        assert!(rho.get(0, 1).abs() < 1e-14);
        // real time branch equals the Matsubara one (equilibrium)
        let rho_t = density_matrix(&g, 3);
        assert!((rho_t.get(0, 0) - rho.get(0, 0)).abs() < 1e-12);
        assert!((rho_t.get(1, 1) - rho.get(1, 1)).abs() < 1e-12);
    }
}
