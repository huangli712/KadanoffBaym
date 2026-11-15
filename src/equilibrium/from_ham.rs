// Project : KadanoffBaym
// File    : src/equilibrium/from_ham.rs
// Purpose : equilibrium propagators from a Hamiltonian (blueprint
//           julia/src/equilibrium/from_ham.jl)
//
// Split from the former standard.rs (one file per blueprint source).

use crate::complex::Complex;
use crate::linalg::DenseMat;
use crate::linalg::eigh;
use crate::linalg::expm;
use crate::types::ContourGreen;
use crate::types::Sign;

/// `fermi_exp(beta, tau, omega)` (blueprint alias of `fermi`).
pub(crate) fn fermi_exp(beta: f64, tau: f64, omega: f64) -> f64 {
    crate::math::fermi_tau(beta, tau, omega)
}

/// `bose_exp(beta, tau, omega)`; the zero mode is a pole (panics like
/// the blueprint's `DomainError`).
pub(crate) fn bose_exp(beta: f64, tau: f64, omega: f64) -> f64 {
    assert!(omega != 0.0, "bose_exp: zero mode of the Bose function");
    crate::math::bose_tau(beta, tau, omega)
}

/// `sign * R diag(d) R^dagger` for real eigenvalue scales `d`.
fn r_diag_rdag(r: &DenseMat, d: &[f64], sign: f64) -> DenseMat {
    let n = r.nrows();
    let mut rd = DenseMat::zeros(n);
    for (j, dj) in d.iter().enumerate().take(n) {
        for i in 0..n {
            rd.set(i, j, r.get(i, j).scale(*dj));
        }
    }
    let mut out = rd.mat_mul(&r.adjoint());
    if sign != 1.0 {
        out.scale(sign);
    }
    out
}

/// Blueprint `_check_free_hamiltonian`: shapes, finiteness, hermiticity.
fn check_free_hamiltonian(g: &ContourGreen, h0: &DenseMat) {
    assert!(
        g.size1 == g.size2,
        "from_ham: G must be square, got {}x{}",
        g.size1,
        g.size2
    );
    assert_eq!(
        (h0.nrows(), h0.ncols()),
        (g.size1, g.size2),
        "from_ham: H must be {}x{}",
        g.size1,
        g.size2
    );
    for i in 0..g.size1 {
        for j in 0..g.size2 {
            let v = h0.get(i, j);
            assert!(
                v.re.is_finite() && v.im.is_finite(),
                "from_ham: H is not finite"
            );
        }
    }
    // ||H - H^dagger|| <= 1e-12 * max(1, ||H||)
    let mut diff = 0.0;
    let mut norm = 0.0;
    for i in 0..g.size1 {
        for j in 0..g.size2 {
            let v = h0.get(i, j) - h0.get(j, i).conj();
            diff += v.abs_sqr();
            norm += h0.get(i, j).abs_sqr();
        }
    }
    assert!(
        diff.sqrt() <= 1e-12 * 1.0f64.max(norm.sqrt()),
        "from_ham: H is not hermitian"
    );
}

/// `(-i * hstep) * m`.
fn scale_ih(m: &DenseMat, hstep: f64) -> DenseMat {
    let w = Complex::new(0.0, -hstep);
    DenseMat::from_vec(
        m.nrows(),
        m.ncols(),
        m.raw().iter().map(|&z| z * w).collect(),
    )
}

/// `mu * I - m`.
fn mu_minus(m: &DenseMat, mu: f64) -> DenseMat {
    let d = m.nrows();
    let mut r = m.clone();
    for i in 0..d {
        let v = r.get(i, i);
        r.set(i, i, v - Complex::new(mu, 0.0));
    }
    r
}

/// Fill all the components of the free Green's function `g` from the
/// Hamiltonian `h0` at the Matsubara axis and the propagators `u[n]`
/// including the phase `e^{i mu t}` (blueprint
/// `_fill_green_from_propagator!`).
fn fill_from_propagator(g: &mut ContourGreen, mu: f64, h0: &DenseMat, beta: f64, u: &[DenseMat]) {
    let d = g.size1;
    let nt = g.nt;
    let fermion = g.sign == Sign::Fermi;

    // K = mu*I - H0 and its eigendecomposition
    let k = {
        let mut kk = DenseMat::zeros(d);
        for i in 0..d {
            for j in 0..d {
                let diag = if i == j {
                    Complex::new(mu, 0.0)
                } else {
                    Complex::ZERO
                };
                kk.set(i, j, diag - h0.get(i, j));
            }
        }
        kk
    };
    let (lambda, r) = eigh(&k);

    // matsubara and lmix components
    let dtau = beta / g.ntau as f64;
    for m in 0..=(g.ntau as i64) {
        let tau = m as f64 * dtau;
        let (matval, mut x) = if fermion {
            let fl: Vec<f64> = lambda.iter().map(|&w| fermi_exp(beta, tau, w)).collect();
            let fneg: Vec<f64> = lambda.iter().map(|&w| fermi_exp(beta, tau, -w)).collect();
            (r_diag_rdag(&r, &fl, -1.0), r_diag_rdag(&r, &fneg, 1.0))
        } else {
            let bl: Vec<f64> = lambda.iter().map(|&w| bose_exp(beta, tau, w)).collect();
            let bneg: Vec<f64> = lambda.iter().map(|&w| bose_exp(beta, tau, -w)).collect();
            (r_diag_rdag(&r, &bl, 1.0), r_diag_rdag(&r, &bneg, -1.0))
        };
        // X carries a leading +/- i
        for i in 0..d {
            for j in 0..d {
                let v = x.get(i, j);
                x.set(i, j, Complex::new(-v.im, v.re));
            }
        }
        for i in 0..d {
            for j in 0..d {
                g.set_mat(m, i, j, matval.get(i, j));
            }
        }
        for n in 0..=(nt as i64) {
            let ux = u[n as usize].mat_mul(&x);
            for i in 0..d {
                for j in 0..d {
                    g.set_lmix(n, m, i, j, ux.get(i, j));
                }
            }
        }
    }

    // occupation matrix N
    let nocc: Vec<f64> = lambda
        .iter()
        .map(|&w| {
            if fermion {
                crate::math::fermi(beta, -w)
            } else {
                -crate::math::bose(beta, -w)
            }
        })
        .collect();
    let nmat = r_diag_rdag(&r, &nocc, 1.0);

    // ret and less components
    for row in 0..=(nt as i64) {
        for col in 0..=row {
            // G^R(t_row, t_col) = -i U_row U_col^dagger
            let uu = u[row as usize].mat_mul(&u[col as usize].adjoint());
            for i in 0..d {
                for j in 0..d {
                    let v = uu.get(i, j);
                    g.set_ret(row, col, i, j, Complex::new(v.im, -v.re));
                }
            }
            // G^<(t_col, t_row) = i U_col N U_row^dagger
            let un = u[col as usize]
                .mat_mul(&nmat)
                .mat_mul(&u[row as usize].adjoint());
            for i in 0..d {
                for j in 0..d {
                    let v = un.get(i, j);
                    g.set_less(col, row, i, j, Complex::new(-v.im, v.re));
                }
            }
        }
    }
}

/// Free Green's function of the constant hermitian Hamiltonian `h0`
/// (blueprint `from_ham!(G, mu, H0, beta, h)`).
///
/// *Remarks* : *Free Green's Functions From Hamiltonian*
///
/// Let ``K = \mu - H(-i\beta) = R\ \mathrm{diag}(\lambda) R^{\dagger}``, and
/// ``U_n`` be the propagator from 0 to ``t_n``, including the phase
/// ``e^{i\mu t_n}``. For fermions,
///
/// ```math
/// \begin{equation}
/// G^{M}(\tau) = -R\ \mathrm{diag}[e^{\tau\lambda}n_F(\lambda)] R^{\dagger},
/// \quad
/// G^{\rceil}(t_n,\tau) = iU_n R\ \mathrm{diag}[e^{-\tau\lambda}n_F(-\lambda)]
/// R^{\dagger},
/// \end{equation}
/// ```
///
/// ```math
/// \begin{equation}
/// G^{R}(t_n,t_j) = -iU_n U^{\dagger}_j,
/// \quad
/// G^{<}(t_j,t_n) = iU_j N U^{\dagger}_n,
/// \quad
/// N = R\ \mathrm{diag}[n_F(-\lambda)] R^{\dagger}.
/// \end{equation}
/// ```
///
/// For bosons, ``G^{M}`` and ``N`` change sign, ``n_F`` is replaced by
/// ``n_B``, and ``G^{\rceil}`` is multiplied by -1.
///
/// For a constant Hamiltonian, ``U_n = e^{ihnK}``. For a time-dependent
/// Hamiltonian, the commutator-free expansions of order 2 and 4 are used,
///
/// ```math
/// \begin{equation}
/// U_n = e^{-ih H(t_{n-1/2})} U_{n-1}, \quad
/// U_n = e^{-ih(a_1 H_1 + a_2 H_2)} e^{-ih(a_2 H_1 + a_1 H_2)} U_{n-1},
/// \end{equation}
/// ```
///
/// where ``H_{1,2} = H(t_{n-1} + c_{1,2} h)``, ``c_{1,2} = 1/2 \mp
/// \sqrt{3}/6``, ``a_{1,2} = (3 \mp 2\sqrt{3})/12``. The Hamiltonian at the
/// fractional time points is obtained by polynomial interpolation of order
/// `solve_order`. Finally ``U_n \to U_n e^{i\mu t_n}``.
///
/// *References* :
///
/// Please see [`NESSi`] `cntr_equilibrium_impl.hpp`, `green_from_H`,
/// `propagator_exp`.
pub fn from_ham(g: &mut ContourGreen, mu: f64, h0: &DenseMat, beta: f64, h: f64) {
    check_free_hamiltonian(g, h0);
    assert!(beta > 0.0, "from_ham: beta must be positive");
    let d = g.size1;
    let nt = g.nt;

    // U_n = exp(i h (mu - H0))^n with U[0] = I
    let km = mu_minus(h0, mu);
    let udt = expm(&scale_ih(&km, h));
    let mut u = vec![DenseMat::identity(d)];
    for n in 1..=nt {
        u.push(u[n - 1].mat_mul(&udt));
    }
    fill_from_propagator(g, mu, h0, beta, &u);
}

// ---------------------------------------------------------------------------
// Time-dependent Hamiltonian: commutator-free propagators (blueprint
// from_ham.jl, the `from_ham!(G, mu, H::Cf, ...)` entry)
// ---------------------------------------------------------------------------

use crate::math::interpolation;
use crate::types::MATSUBARA;
use crate::types::TimeFunction;
use crate::weights::Integrator;

#[allow(dead_code)]
/// Prepare the Hamiltonian for the propagation from `tstp - 1` to
/// `tstp` (NESSi indices). If `fix = false`, `h[tstp]` is extrapolated
/// from the previous time steps when `tstp > kt`. Returns the 0-based
/// index of the last grid point of the interpolation interval
/// (blueprint `_hamiltonian_cf`).
fn hamiltonian_cf(tstp: i64, h: &mut TimeFunction, kt: i64, fix: bool) -> i64 {
    if !fix && tstp > kt {
        crate::math::extrapolate_timestep_cf(tstp, h, kt);
    }
    tstp.max(kt)
}

#[allow(dead_code)]
/// Second order commutator-free propagation `U_n = e^{-i h H(t_{n-1/2})}
/// U_{n-1}` (blueprint `_propagator_cf2!`).
fn propagator_cf2(
    u: &mut [DenseMat],
    tstp: i64,
    h: &mut TimeFunction,
    hstep: f64,
    kt: i64,
    fix: bool,
) {
    let n1 = hamiltonian_cf(tstp, h, kt, fix);
    let integ = Integrator::new(kt);
    let hm = interpolation(n1, tstp as f64 - 0.5, h, &integ);
    u[tstp as usize] = expm(&scale_ih(&hm, hstep)).mat_mul(&u[(tstp - 1) as usize]);
}

#[allow(dead_code)]
/// Fourth order commutator-free propagation
/// `U_n = e^{-ih(a1 H1 + a2 H2)} e^{-ih(a2 H1 + a1 H2)} U_{n-1}`
/// (blueprint `_propagator_cf4!`).
fn propagator_cf4(
    u: &mut [DenseMat],
    tstp: i64,
    h: &mut TimeFunction,
    hstep: f64,
    kt: i64,
    fix: bool,
) {
    let n1 = hamiltonian_cf(tstp, h, kt, fix);
    let integ = Integrator::new(kt);
    let r3 = 3.0f64.sqrt();
    let h1 = interpolation(n1, tstp as f64 - 0.5 - r3 / 6.0, h, &integ);
    let h2 = interpolation(n1, tstp as f64 - 0.5 + r3 / 6.0, h, &integ);
    let a1 = (3.0 - 2.0 * r3) / 12.0;
    let a2 = (3.0 + 2.0 * r3) / 12.0;
    let mut m1 = h1.clone();
    m1.scale(a1);
    m1.axpy(a2, &h2);
    let mut m2 = h1;
    m2.scale(a2);
    m2.axpy(a1, &h2);
    let e1 = expm(&scale_ih(&m1, hstep));
    let e2 = expm(&scale_ih(&m2, hstep));
    u[tstp as usize] = e1.mat_mul(&e2).mat_mul(&u[(tstp - 1) as usize]);
}

#[allow(dead_code)]
fn check_hermitian_step(h: &TimeFunction, t: i64) {
    let d = h.size1;
    let blk = h.block(t);
    let mut nrm = 0.0f64;
    let mut dev = 0.0f64;
    for i in 0..d {
        for j in 0..d {
            let v = blk[i * d + j];
            nrm += v.abs() * v.abs();
            let dz = v - blk[j * d + i].conj();
            dev += dz.abs() * dz.abs();
        }
    }
    assert!(
        dev.sqrt() <= 1e-12 * 1.0f64.max(nrm.sqrt()),
        "H is not hermitian at time step {t}"
    );
}

#[allow(dead_code)]
/// Calculate the free Green's function of the time-dependent hermitian
/// Hamiltonian `h` (blueprint `from_ham!(G, mu, H::Cf, beta, h;
/// fix_hamiltonian, solve_order, cf_order)`). The Matsubara component
/// is determined by `h` at the Matsubara axis; the propagators use the
/// commutator-free expansion of order `cf_order` (2 or 4) with the
/// Hamiltonian at the fractional time points interpolated to
/// `solve_order`. If `fix_hamiltonian = false`, the values of `h` at
/// `t > solve_order` are extrapolated from the previous time steps and
/// written into `h`.
#[allow(clippy::too_many_arguments)]
fn from_ham_cf(
    g: &mut ContourGreen,
    mu: f64,
    h: &mut TimeFunction,
    beta: f64,
    hstep: f64,
    fix_hamiltonian: bool,
    solve_order: i64,
    cf_order: i64,
) {
    assert!(
        (1..=5).contains(&solve_order),
        "solve_order must be in 1..=5, got {solve_order}"
    );
    assert!(cf_order == 2 || cf_order == 4, "cf_order must be 2 or 4");
    assert!(beta > 0.0, "beta must be positive");
    assert!(hstep > 0.0, "h must be positive");
    let d = g.size1;
    let kt = solve_order;
    let nt = g.nt as i64;
    let h0 = DenseMat::from_slice(d, d, h.block(MATSUBARA));
    check_free_hamiltonian(g, &h0);
    let nmax = if fix_hamiltonian {
        (kt + 1).max(nt)
    } else {
        kt + 1
    };
    assert!(
        (h.nt as i64) >= nmax.max(nt),
        "H must have at least {} real time steps",
        nmax.max(nt)
    );
    for t in 0..=nmax {
        check_hermitian_step(h, t);
    }

    // propagators without the phase
    let mut u = vec![DenseMat::identity(d); (nt + 1) as usize];
    for tstp in 1..=nt {
        if cf_order == 2 {
            propagator_cf2(&mut u, tstp, h, hstep, kt, fix_hamiltonian);
        } else {
            propagator_cf4(&mut u, tstp, h, hstep, kt, fix_hamiltonian);
        }
    }

    // phase e^{i mu t}
    for (n, un) in u.iter_mut().enumerate() {
        let ph = Complex::from_polar(1.0, mu * hstep * n as f64);
        for z in un.raw_mut().iter_mut() {
            *z *= ph;
        }
    }
    fill_from_propagator(g, mu, &h0, beta, &u);
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::math::fermi;
    use crate::types::Contour;

    /// H0 = diag(-0.4, 0.6), mu = 0.1: K = diag(0.5, -0.5) with
    /// lambda = (-0.5, 0.5) ascending and R swapping the two orbitals.
    #[test]
    fn diagonal_hamiltonian_matches_analytic_fermi() {
        let (beta, dt, mu) = (4.0, 0.025, 0.1);
        let c = Contour::new(20, 100, 2, 2, 20.0 * dt, beta);
        let mut g = ContourGreen::new(&c, Sign::Fermi, Complex::ZERO);
        let mut h0 = DenseMat::zeros(2);
        h0.set(0, 0, Complex::new(-0.4, 0.0));
        h0.set(1, 1, Complex::new(0.6, 0.0));
        from_ham(&mut g, mu, &h0, beta, dt);

        let dtau = beta / 100.0;
        for m in [0i64, 25, 50, 100] {
            let tau = m as f64 * dtau;
            // mat = -R diag(e^{tau lam} n_F(lam)) R^dagger with the swap
            let v_l1 = -(-0.5f64 * tau).exp() * fermi(beta, -0.5); // lam = -0.5 -> orbital 1
            let v_l2 = -(0.5f64 * tau).exp() * fermi(beta, 0.5);
            assert!(
                (g.get_mat(m, 0, 0) - Complex::new(v_l2, 0.0)).abs() < 1e-12,
                "mat(0,0) at m = {m}"
            );
            assert!(
                (g.get_mat(m, 1, 1) - Complex::new(v_l1, 0.0)).abs() < 1e-12,
                "mat(1,1) at m = {m}"
            );
            assert!(g.get_mat(m, 0, 1).abs() < 1e-13);
            assert!(g.get_mat(m, 1, 0).abs() < 1e-13);
        }

        // ret(n, n) = -i * Id
        for n in [0i64, 5, 20] {
            assert!((g.get_ret(n, n, 0, 0) - Complex::new(0.0, -1.0)).abs() < 1e-12);
            assert!((g.get_ret(n, n, 1, 1) - Complex::new(0.0, -1.0)).abs() < 1e-12);
            assert!(g.get_ret(n, n, 0, 1).abs() < 1e-13);
        }

        // less(0, 0) = i * N with N = R diag(n_F(-lam)) R^dagger
        let i = Complex::new(0.0, 1.0);
        assert!((g.get_less(0, 0, 0, 0) - i * fermi(beta, -0.5)).abs() < 1e-12);
        assert!((g.get_less(0, 0, 1, 1) - i * fermi(beta, 0.5)).abs() < 1e-12);
        // lmix(0, 0) = i * N as well (G^<(0,0) = G^rangle(0,0))
        assert!((g.get_lmix(0, 0, 0, 0) - i * fermi(beta, -0.5)).abs() < 1e-12);
    }

    /// H(t) = H0 + A t^2 I commutes at all times: the analytic
    /// propagator U(t) = e^{i mu t} e^{-i H0 t} e^{-i phi(t)} with
    /// phi = A t^3 / 3. The polynomial interpolation of a(t) = A t^2
    /// at order 5 is exact, so CF4 (whose Gauss quadrature is exact for
    /// cubics) matches the analytic result to round-off, while CF2
    /// carries the O(h^2) midpoint quadrature error.
    #[test]
    fn cf2_cf4_match_analytic_commuting() {
        let (beta, hstep, mu, k, nt) = (4.0f64, 0.05, 0.1, 5i64, 30usize);
        let c = Contour::new(nt as i64, 60, 2, 2, nt as f64 * hstep, beta);
        let mut h0 = DenseMat::zeros(2);
        h0.set(0, 0, Complex::new(-0.4, 0.0));
        h0.set(0, 1, Complex::new(0.0, 0.1));
        h0.set(1, 0, Complex::new(0.0, -0.1));
        h0.set(1, 1, Complex::new(0.6, 0.0));
        let amp = 0.3f64;

        // H at the Matsubara axis and at the real steps 0..=max(k+1, nt)
        let nt_h = ((k + 1) as usize).max(nt);
        let mut h = TimeFunction::new(nt_h, 2, 2, Complex::ZERO);
        for t in 0..=(nt_h as i64) {
            let a = amp * (t as f64 * hstep) * (t as f64 * hstep);
            for i in 0..2 {
                for j in 0..2 {
                    let mut v = h0.get(i, j);
                    if i == j {
                        v += Complex::new(a, 0.0);
                    }
                    h.set(t, i, j, v);
                }
            }
        }
        for i in 0..2 {
            for j in 0..2 {
                h.set(MATSUBARA, i, j, h0.get(i, j));
            }
        }

        // analytic propagators
        let u_an: Vec<DenseMat> = (0..=(nt as i64))
            .map(|n| {
                let t = n as f64 * hstep;
                let phi = amp * t * t * t / 3.0;
                let mut u = expm(&scale_ih(&h0, t));
                let ph = Complex::from_polar(1.0, mu * t - phi);
                for z in u.raw_mut().iter_mut() {
                    *z *= ph;
                }
                u
            })
            .collect();

        for cf in [2i64, 4] {
            let mut g_cf = ContourGreen::new(&c, Sign::Fermi, Complex::ZERO);
            let mut hh = h.clone();
            from_ham_cf(&mut g_cf, mu, &mut hh, beta, hstep, true, k, cf);
            let mut g_an = ContourGreen::new(&c, Sign::Fermi, Complex::ZERO);
            fill_from_propagator(&mut g_an, mu, &h0, beta, &u_an);

            let mut e = 0.0f64;
            for q in 0..g_an.mat.raw().len() {
                e = e.max((g_cf.mat.raw()[q] - g_an.mat.raw()[q]).abs());
            }
            for n in 0..=(nt as i64) {
                for m in 0..=60 {
                    for q in 0..4 {
                        e = e.max(
                            (g_cf.lmix.raw_block_at(n, m)[q] - g_an.lmix.raw_block_at(n, m)[q])
                                .abs(),
                        );
                    }
                }
                for j in 0..=n {
                    for q in 0..4 {
                        e = e.max(
                            (g_cf.ret.raw_block_at(n, j)[q] - g_an.ret.raw_block_at(n, j)[q]).abs(),
                        );
                        e = e.max(
                            (g_cf.less.raw_block_at(j, n)[q] - g_an.less.raw_block_at(j, n)[q])
                                .abs(),
                        );
                    }
                }
            }
            let tol = if cf == 2 { 5e-4 } else { 1e-9 };
            assert!(e < tol, "cf{cf} vs analytic: {e}");
        }
    }
}
