// Project : KadanoffBaym
// File    : src/solvers/dyson.rs
// Purpose : Dyson equation: argument checks and the full driver
//           (blueprint: julia/src/solvers/dyson.jl)

use crate::core::contour::Sign;
use super::dyson_mat::MatsubaraMethod;
use crate::core::green::{ContourGreen, TimeFunction};

/// Check that `h` is a finite and hermitian `d x d` matrix at all the
/// time steps in `tstps` (`MATSUBARA` for the Matsubara axis)
/// (blueprint `_check_hamiltonian`).
pub fn check_hamiltonian(h: &TimeFunction, tstps: impl Iterator<Item = i64>, d: usize) {
    assert!(
        h.size1 == d && h.size2 == d,
        "H must be {d} x {}, got {} x {}",
        d,
        h.size1,
        h.size2
    );
    for n in tstps {
        assert!(
            (-1..=(h.nt as i64)).contains(&n),
            "H is not defined at time step {n}"
        );
        let hn = h.block(n);
        for z in hn {
            assert!(
                z.re.is_finite() && z.im.is_finite(),
                "H is not finite at time step {n}"
            );
        }
        // ||H - H^dag|| <= tol max(1, ||H||)
        let mut r = 0.0f64;
        let mut norm = 0.0f64;
        for b in 0..d {
            for a in 0..d {
                let dz = hn[a * d + b] - hn[b * d + a].conj();
                r += dz.abs() * dz.abs();
                let e = hn[a * d + b];
                norm += e.abs() * e.abs();
            }
        }
        assert!(
            r.sqrt() <= 1e-12 * 1.0f64.max(norm.sqrt()),
            "H is not hermitian at time step {n}"
        );
    }
}

/// Check the compatibility of the arguments of the Dyson solvers
/// (blueprint `_check_dyson`).
pub fn check_dyson(g: &ContourGreen, sigma: &ContourGreen, h: &TimeFunction, solve_order: i64) {
    assert!(
        (1..=5).contains(&solve_order),
        "solve_order must be in 1..=5, got {solve_order}"
    );
    assert!(
        g.sign == Sign::Fermi || g.sign == Sign::Bose,
        "unknown statistics of G"
    );
    assert!(
        g.sign == sigma.sign,
        "G and Sigma must have the same statistics"
    );
    assert!(g.size1 == g.size2, "G must be square");
    assert!(
        sigma.size1 == g.size1 && sigma.size2 == g.size2,
        "G and Sigma have different sizes"
    );
    assert!(
        g.ntau == sigma.ntau,
        "G and Sigma have different imaginary time grids"
    );
    assert!(
        g.nt == sigma.nt,
        "G and Sigma have different real time grids"
    );
    assert!(
        h.size1 == g.size1 && h.size2 == g.size2,
        "H must be {} x {}, got {} x {}",
        g.size1,
        g.size2,
        h.size1,
        h.size2
    );
}

/// Check the arguments of the real time Dyson solvers; the time steps
/// `0, .., nmax` of `G`, `Sigma`, and `H` are needed (blueprint
/// `_check_dyson_rt`).
pub fn check_dyson_rt(g: &ContourGreen, sigma: &ContourGreen, h: &TimeFunction, k: i64, nmax: i64) {
    check_dyson(g, sigma, h, k);
    assert!(
        (g.nt as i64) >= nmax,
        "G must have at least {nmax} real time steps"
    );
    assert!(
        (h.nt as i64) >= nmax,
        "H must have at least {nmax} real time steps"
    );
    check_hamiltonian(h, 0..=nmax, g.size1);
}

/// Solve the Dyson equation for all the components of `g`: Matsubara,
/// start-up, and then all the remaining real time steps (blueprint
/// `dyson!`). `h` is the Hamiltonian, `sigma` the self-energy.
#[allow(clippy::too_many_arguments)]
pub fn dyson(
    g: &mut ContourGreen,
    mu: f64,
    h: &TimeFunction,
    sigma: &ContourGreen,
    beta: f64,
    hstep: f64,
    solve_order: i64,
    matsubara_method: MatsubaraMethod,
    force_hermitian: bool,
) {
    let k = solve_order;
    check_dyson_rt(g, sigma, h, k, g.nt as i64);

    super::dyson_mat::dyson_mat(
        g,
        sigma,
        mu,
        h,
        beta,
        solve_order,
        matsubara_method,
        force_hermitian,
        None,
        1e-12,
    );
    super::dyson_start::dyson_start(g, mu, h, sigma, beta, hstep, solve_order);
    for tstp in (k + 1)..=(g.nt as i64) {
        super::dyson_timestep::dyson_timestep(tstp, g, mu, h, sigma, beta, hstep, solve_order);
    }
}

/// Convenience wrapper matching the blueprint defaults (`:fixpoint`,
/// hermitian symmetrization on).
#[allow(clippy::too_many_arguments)]
pub fn dyson_default(
    g: &mut ContourGreen,
    mu: f64,
    h: &TimeFunction,
    sigma: &ContourGreen,
    beta: f64,
    hstep: f64,
    solve_order: i64,
) {
    dyson(
        g,
        mu,
        h,
        sigma,
        beta,
        hstep,
        solve_order,
        MatsubaraMethod::Fixpoint,
        true,
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::complex::c64::C64;
    use crate::equilibrium::from_ham;
    use crate::linalg::mat::DenseMat;
    use crate::core::contour::{Contour, Sign};

    fn const_hamiltonian(nt: usize) -> (TimeFunction, DenseMat) {
        let mut h = TimeFunction::new(nt, 2, 2, C64::ZERO);
        let m = DenseMat::from_rows(&[
            vec![C64::new(-0.4, 0.0), C64::new(0.0, 0.1)],
            vec![C64::new(0.0, -0.1), C64::new(0.6, 0.0)],
        ]);
        for t in -1..=(nt as i64) {
            for i in 0..2 {
                for j in 0..2 {
                    h.set(t, i, j, m.get(i, j));
                }
            }
        }
        (h, m)
    }

    /// Sigma = 0: the full dyson! driver must reproduce the free
    /// propagator of from_ham in all four components.
    #[test]
    fn sigma_zero_full_dyson_matches_from_ham() {
        let (nt, ntau) = (20usize, 100);
        let (beta, hstep, mu, k) = (4.0f64, 0.05, 0.0, 5i64);
        let c = Contour::new(nt as i64, ntau as i64, 2, 2, hstep * nt as f64, beta);
        let (h, h0) = const_hamiltonian(nt);
        let sigma = ContourGreen::new(&c, Sign::Fermi, C64::ZERO);
        let mut g = ContourGreen::new(&c, Sign::Fermi, C64::ZERO);
        dyson_default(&mut g, mu, &h, &sigma, beta, hstep, k);

        let mut want = ContourGreen::new(&c, Sign::Fermi, C64::ZERO);
        from_ham(&mut want, mu, &h0, beta, hstep);

        let mut e_mat = 0.0f64;
        for q in 0..want.mat.raw().len() {
            e_mat = e_mat.max((g.mat.raw()[q] - want.mat.raw()[q]).abs());
        }
        let mut e_ret = 0.0f64;
        let mut e_lmix = 0.0f64;
        let mut e_less = 0.0f64;
        for i in 0..=(nt as i64) {
            for j in 0..=i {
                for q in 0..4 {
                    e_ret = e_ret
                        .max((g.ret.raw_block_at(i, j)[q] - want.ret.raw_block_at(i, j)[q]).abs());
                    e_less = e_less
                        .max((g.less.raw_block_at(j, i)[q] - want.less.raw_block_at(j, i)[q]).abs());
                }
            }
            for m in 0..=(ntau as i64) {
                for q in 0..4 {
                    e_lmix = e_lmix
                        .max((g.lmix.raw_block_at(i, m)[q] - want.lmix.raw_block_at(i, m)[q]).abs());
                }
            }
        }
        assert!(e_mat < 1e-10, "mat {e_mat}");
        assert!(e_ret < 1e-6, "ret {e_ret}");
        assert!(e_lmix < 1e-6, "lmix {e_lmix}");
        assert!(e_less < 1e-6, "less {e_less}");
    }
}
