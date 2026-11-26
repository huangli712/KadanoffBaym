//
// Project : KadanoffBaym
// File    : src/observables/density.rs
// Purpose : density-matrix observables of all container flavors:
//           standard, steady-state, and pseudo-particle
//

use crate::complex::Complex;
use crate::linalg::DenseMat;
use crate::types::{ContourGreen, PseudoGreen, SteadyGreen};
use crate::types::{FftDomain, Sign};

use crate::types::MATSUBARA;

/// Density (the trace of the density matrix).
pub fn density(g: &ContourGreen, tstp: i64) -> Complex {
    let rho = density_matrix(g, tstp);
    let d = g.size1;
    (0..d).map(|i| rho.get(i, i)).sum()
}

/// Density matrix of a contour-ordered Green's function: `-G^M(beta)` at
/// equilibrium (`tstp == MATSUBARA`), `i * sign * G^<(t,t)` at real time
/// `tstp >= 0`.
pub fn density_matrix(g: &ContourGreen, tstp: i64) -> DenseMat {
    assert!(
        g.size1 == g.size2,
        "density_matrix: the Green's function must be square, got {}x{}",
        g.size1,
        g.size2
    );
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

/// Density (the trace of the density matrix).
pub fn pseudo_density(g: &PseudoGreen, tstp: i64) -> Complex {
    let rho = pseudo_density_matrix(g, tstp);
    let d = g.size1;
    (0..d).map(|i| rho[i * d + i]).sum()
}

/// Density matrix of the pseudo-particle Green's function: `rho = -G^M(beta)`
/// on the Matsubara axis, `rho = i xi G^{<}(t, t)` at real time.
pub fn pseudo_density_matrix(g: &PseudoGreen, tstp: i64) -> Vec<Complex> {
    let xi = g.sign.as_i64() as f64;
    if tstp == MATSUBARA {
        g.mat.block(g.ntau).iter().map(|z| -*z).collect()
    } else {
        assert!(
            (0..=g.nt as i64).contains(&tstp),
            "pseudo_density_matrix: tstp {tstp} out of range"
        );
        let blk = g.less.raw_block_at(tstp, tstp);
        (0..blk.len())
            .map(|q| Complex::new(0.0, xi) * blk[q])
            .collect()
    }
}

/// Density of a steady-state Green's function (the trace of the
/// density matrix).
pub fn steady_density(g: &SteadyGreen, sign: Sign) -> Complex {
    let rho = steady_density_matrix(g, sign);
    let d = g.les.size1;
    (0..d).map(|i| rho[i * d + i]).sum()
}

/// Density matrix `rho = xi i G^<(0)` of a steady-state Green's function.
pub fn steady_density_matrix(g: &SteadyGreen, sign: Sign) -> Vec<Complex> {
    let blk = g.les.block(FftDomain::Time, 0).to_vec();
    let xi = Complex::new(0.0, sign.as_i64() as f64);
    blk.into_iter().map(|z| xi * z).collect()
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
        // the trace sums the occupied diagonals and stays real
        let n_equil = density(&g, MATSUBARA);
        assert!((n_equil.re - (f1 + f2)).abs() < 1e-12);
        assert!(n_equil.im.abs() < 1e-12);
        // real-time trace equals the equilibrium one
        assert!((density(&g, 3) - n_equil).abs() < 1e-12);
    }
}
