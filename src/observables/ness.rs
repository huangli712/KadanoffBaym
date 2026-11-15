//
// Project : KadanoffBaym
// File    : src/observables/ness.rs
// Purpose : steady-state and pseudo-particle density matrices
//
// Blueprint: julia/src/observables/ (density_matrix on NessHermMat
// and pseudo-particle containers, convolution_density_matrix).

use crate::complex::Complex;
use crate::fourier::{FftIntegralMethod, integral_transform_to_freq};
use crate::types::{FftDomain, MATSUBARA, Sign};
use crate::types::{PseudoGreen, PseudoGreenTimeStep, SteadyGreen};

/// Density matrix `rho = xi i G^<(0)` of a steady-state Green's
/// function (blueprint `density_matrix(G::NessHermMat, sign)`).
pub fn density_matrix_ness(g: &SteadyGreen, sign: Sign) -> Vec<Complex> {
    let blk = g.les.block(FftDomain::Time, 0).to_vec();
    let xi = Complex::new(0.0, sign.as_i64() as f64);
    blk.into_iter().map(|z| xi * z).collect()
}

/// Density matrix of the pseudo-particle Green's function (blueprint
/// `density_matrix(G::F^{ps}, t)`): `rho = -G^M(beta)` on the
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

/// Density matrix at a time step object (blueprint
/// `density_matrix(g::f^{ps}, t)`).
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

/// Density (the trace of the density matrix, blueprint `density`).
pub fn density_pseudo(g: &PseudoGreen, tstp: i64) -> Complex {
    let rho = density_matrix_pseudo(g, tstp);
    let d = g.size1;
    (0..d).map(|i| rho[i * d + i]).sum()
}

/// Density matrix `xi i C^<(0)` of the steady-state convolution
/// `C = A * B` with `C^<(w) = A^<(w) B^A(w) + A^R(w) B^<(w)`
/// (blueprint `convolution_density_matrix(A::NessHermMat, ...)`). The
/// inputs are transformed on local copies and not changed.
pub fn convolution_density_matrix_ness(
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
