// Project : KadanoffBaym
// File    : src/equilibrium/ham.rs
// Purpose : equilibrium propagators from a Hamiltonian, and the
//           Hamiltonian checks shared with the Dyson solvers

use crate::complex::Complex;
use crate::linalg::DenseMat;
use crate::types::{ContourGreen, MatComp, Sign, TimeFunction};
use crate::weights::Integrator;

use crate::types::MATSUBARA;

use crate::linalg::{eigh, expm, scale_c};
use crate::math::{bose, bose_tau, extrapolate_timestep_cf, fermi, fermi_tau, interpolation};

/// `factor * R diag(diag) R^dagger` for real diagonal entries `diag`.
fn r_diag_rdag(r: &DenseMat, diag: &[f64], factor: f64) -> DenseMat {
    let n = r.nrows();
    let mut rd = DenseMat::zeros(n);
    for (j, dj) in diag.iter().enumerate().take(n) {
        for i in 0..n {
            rd.set(i, j, r.get(i, j).scale(*dj));
        }
    }
    let mut out = rd.mat_mul(&r.adjoint());
    if factor != 1.0 {
        out.scale(factor);
    }
    out
}

/// `(-i * hstep) * m`.
fn scale_ih(m: &DenseMat, hstep: f64) -> DenseMat {
    scale_c(m, Complex::new(0.0, -hstep))
}

/// `m - mu * I`: for a Hamiltonian, the negative of the `K = mu - H` of
/// the propagators.
pub(crate) fn minus_mu(m: &DenseMat, mu: f64) -> DenseMat {
    let d = m.nrows();
    let mut r = m.clone();
    for i in 0..d {
        let v = r.get(i, i);
        r.set(i, i, v - Complex::new(mu, 0.0));
    }
    r
}

/// Whether the `d x d` block `blk` (row-major) is hermitian within
/// `||H - H^dagger|| <= 1e-12 max(1, ||H||)`.
fn is_hermitian_block(blk: &[Complex], d: usize) -> bool {
    let mut r = 0.0f64;
    let mut norm = 0.0f64;
    for b in 0..d {
        for a in 0..d {
            let dz = blk[a * d + b] - blk[b * d + a].conj();
            r += dz.abs() * dz.abs();
            let e = blk[a * d + b];
            norm += e.abs() * e.abs();
        }
    }
    r.sqrt() <= 1e-12 * 1.0f64.max(norm.sqrt())
}

/// Check that the `d x d` block `blk` (row-major) is finite and
/// hermitian; `at` names the block in the panic messages.
fn check_block(blk: &[Complex], d: usize, at: &str) {
    for z in blk {
        assert!(z.re.is_finite() && z.im.is_finite(), "H is not finite {at}");
    }
    assert!(is_hermitian_block(blk, d), "H is not hermitian {at}");
}

/// Check that `h` is a finite and hermitian `d x d` matrix at all the
/// time steps in `tstps` (`MATSUBARA` for the Matsubara axis). Shared
/// by the Dyson solvers.
pub(crate) fn check_hamiltonian(h: &TimeFunction, tstps: impl Iterator<Item = i64>, d: usize) {
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
        check_block(h.block(n), d, &format!("at time step {n}"));
    }
}

/// Check the static Hamiltonian `h0` of the free Green's function `g`:
/// shapes, finiteness, hermiticity.
fn check_static_hamiltonian(g: &ContourGreen, h0: &DenseMat) {
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
    check_block(h0.raw(), g.size1, "in from_ham");
}

/// Fill the Matsubara component `G^M(tau_m)`, `m = 0 ..= ntau`, of the
/// free propagator with the eigendecomposition
/// `K = mu*I - H = R diag(lambda) R^dagger`: `-R diag[e^{tau lambda}
/// n_F(lambda)] R^dagger` for fermions and `+R diag[e^{tau lambda}
/// n_B(lambda)] R^dagger` for bosons.
pub(crate) fn fill_matsubara(
    mat: &mut MatComp,
    lambda: &[f64],
    r: &DenseMat,
    beta: f64,
    fermion: bool,
) {
    let dtau = beta / mat.ntau as f64;
    for m in 0..=(mat.ntau as i64) {
        let tau = m as f64 * dtau;
        let val = if fermion {
            let f: Vec<f64> = lambda.iter().map(|&w| fermi_tau(beta, tau, w)).collect();
            r_diag_rdag(r, &f, -1.0)
        } else {
            let b: Vec<f64> = lambda.iter().map(|&w| bose_tau(beta, tau, w)).collect();
            r_diag_rdag(r, &b, 1.0)
        };
        for i in 0..mat.size1 {
            for j in 0..mat.size2 {
                mat.set(m as usize, i, j, val.get(i, j));
            }
        }
    }
}

/// Fill all the components of the free Green's function `g` from the
/// Hamiltonian `h0` at the Matsubara axis and the propagators `u[n]`
/// including the phase `e^{i mu t}`.
fn fill_from_propagator(g: &mut ContourGreen, mu: f64, h0: &DenseMat, beta: f64, u: &[DenseMat]) {
    let d = g.size1;
    let nt = g.nt;
    let fermion = g.sign == Sign::Fermi;

    // K = mu*I - H0 and its eigendecomposition
    let mut k = minus_mu(h0, mu);
    k.scale(-1.0);
    let (lambda, r) = eigh(&k);

    // matsubara component
    fill_matsubara(&mut g.mat, &lambda, &r, beta, fermion);

    // lmix components
    let dtau = beta / g.ntau as f64;
    for m in 0..=(g.ntau as i64) {
        let tau = m as f64 * dtau;
        let mut x = if fermion {
            let fneg: Vec<f64> = lambda.iter().map(|&w| fermi_tau(beta, tau, -w)).collect();
            r_diag_rdag(&r, &fneg, 1.0)
        } else {
            let bneg: Vec<f64> = lambda.iter().map(|&w| bose_tau(beta, tau, -w)).collect();
            r_diag_rdag(&r, &bneg, -1.0)
        };
        // X carries a leading +/- i
        for i in 0..d {
            for j in 0..d {
                let v = x.get(i, j);
                x.set(i, j, Complex::new(-v.im, v.re));
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
                fermi(beta, -w)
            } else {
                -bose(beta, -w)
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
/// `solve_order`. Finally ``U_n \to U_n e^{i\mu t_n}``. The time-dependent
/// case is handled by [`from_ham_cf`].
///
///
/// Free Green's function of the constant hermitian Hamiltonian `h0`.
pub fn from_ham(g: &mut ContourGreen, mu: f64, h0: &DenseMat, beta: f64, h: f64) {
    check_static_hamiltonian(g, h0);
    assert!(beta > 0.0, "from_ham: beta must be positive");
    let d = g.size1;
    let nt = g.nt;

    // U_n = exp(i h (mu - H0))^n with U[0] = I
    let km = minus_mu(h0, mu);
    let udt = expm(&scale_ih(&km, h));
    let mut u = vec![DenseMat::identity(d)];
    for n in 1..=nt {
        u.push(u[n - 1].mat_mul(&udt));
    }
    fill_from_propagator(g, mu, h0, beta, &u);
}

// ------------------------------------------------------------------
// Time-dependent Hamiltonian: commutator-free propagators
// ------------------------------------------------------------------

/// Prepare the Hamiltonian for the propagation from `tstp - 1` to
/// `tstp` (0-based indices). If `fix = false`, `h[tstp]` is extrapolated
/// from the previous time steps when `tstp > kt`. Returns the 0-based
/// index of the last grid point of the interpolation interval.
fn prepare_hamiltonian(tstp: i64, h: &mut TimeFunction, kt: i64, fix: bool) -> i64 {
    if !fix && tstp > kt {
        extrapolate_timestep_cf(tstp, h, kt);
    }
    tstp.max(kt)
}

/// Second order commutator-free propagation `U_n = e^{-i h H(t_{n-1/2})}
/// U_{n-1}`.
fn propagator_cf2(
    u: &mut [DenseMat],
    tstp: i64,
    h: &mut TimeFunction,
    hstep: f64,
    kt: i64,
    fix: bool,
) {
    let n1 = prepare_hamiltonian(tstp, h, kt, fix);
    let integ = Integrator::new(kt);
    let hm = interpolation(n1, tstp as f64 - 0.5, h, &integ);
    u[tstp as usize] = expm(&scale_ih(&hm, hstep)).mat_mul(&u[(tstp - 1) as usize]);
}

/// Fourth order commutator-free propagation
/// `U_n = e^{-ih(a1 H1 + a2 H2)} e^{-ih(a2 H1 + a1 H2)} U_{n-1}`.
fn propagator_cf4(
    u: &mut [DenseMat],
    tstp: i64,
    h: &mut TimeFunction,
    hstep: f64,
    kt: i64,
    fix: bool,
) {
    let n1 = prepare_hamiltonian(tstp, h, kt, fix);
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

/// Calculate the free Green's function of the time-dependent hermitian
/// Hamiltonian `h`. The Matsubara component
/// is determined by `h` at the Matsubara axis; the propagators use the
/// commutator-free expansion of order `cf_order` (2 or 4) with the
/// Hamiltonian at the fractional time points interpolated to
/// `solve_order`. If `fix_hamiltonian = false`, the values of `h` at
/// `t > solve_order` are extrapolated from the previous time steps and
/// written into `h`.
#[allow(clippy::too_many_arguments)]
pub fn from_ham_cf(
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
    check_static_hamiltonian(g, &h0);
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
    check_hamiltonian(h, 0..=nmax, d);

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

    use crate::types::Contour;

    const NT: i64 = 100;
    const NTAU: i64 = 100;
    const TMAX: f64 = 1.0;
    const BETA: f64 = 5.0;
    const HSTEP: f64 = TMAX / 100.0;
    const DTAU: f64 = BETA / 100.0;
    const MU: f64 = 0.3;
    const TOL: f64 = 1e-12;

    fn approx(a: Complex, b: Complex) -> bool {
        (a - b).abs() <= TOL * (1.0 + b.abs())
    }

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

    #[test]
    fn from_ham_scalar_fermion_and_boson() {
        let cases = [(Sign::Fermi, 1.0f64), (Sign::Bose, 1.0f64)];
        for &(sign, eps_band) in &cases {
            let cont = Contour::new(NT, NTAU, 1, 1, TMAX, BETA);
            let mut g = ContourGreen::new(&cont, sign, Complex::ZERO);
            let mut h0 = DenseMat::zeros(1);
            h0.set(0, 0, Complex::new(eps_band, 0.0));
            from_ham(&mut g, MU, &h0, BETA, HSTEP);

            let xi = sign.as_i64() as f64;
            let e = eps_band - MU;
            let n = if sign == Sign::Fermi {
                1.0 / ((BETA * e).exp() + 1.0)
            } else {
                1.0 / ((BETA * e).exp() - 1.0)
            };

            // Matsubara: G(tau) = -e^{-eps tau} (1 + xi n)
            for m in 0..=NTAU {
                let tau = m as f64 * DTAU;
                let want = Complex::new(-(-e * tau).exp() * (1.0 + xi * n), 0.0);
                assert!(
                    approx(g.get_mat(m, 0, 0), want),
                    "mat({m}) = {:?} vs {want:?}",
                    g.get_mat(m, 0, 0)
                );
            }

            // Retarded: G(t,t') = -i e^{-i eps (t-t')}
            for i in 0..=NT {
                for j in 0..=i {
                    let want = Complex::new(0.0, -1.0)
                        * Complex::from_polar(1.0, -e * (i - j) as f64 * HSTEP);
                    assert!(approx(g.get_ret(i, j, 0, 0), want), "ret({i},{j})");
                }
            }

            // Left-mixing: G(t,tau) = -i xi n e^{eps tau} e^{-i eps t}
            for i in 0..=NT {
                for m in 0..=NTAU {
                    let tau = m as f64 * DTAU;
                    let want = -Complex::new(0.0, xi * n)
                        * Complex::new((e * tau).exp(), 0.0)
                        * Complex::from_polar(1.0, -e * i as f64 * HSTEP);
                    assert!(approx(g.get_lmix(i, m, 0, 0), want), "lmix({i},{m})");
                }
            }

            // Lesser: G(t',t) = -i xi n e^{-i eps (t-t')}
            for j in 0..=NT {
                for i in 0..=j {
                    // e^{-i eps (i-j) h} with i-j <= 0, i.e. e^{+i eps (j-i) h}
                    let want = Complex::new(0.0, -xi * n)
                        * Complex::from_polar(1.0, e * (j - i) as f64 * HSTEP);
                    assert!(approx(g.get_less(i, j, 0, 0), want), "less({i},{j})");
                }
            }

            // KMS boundary: G(0+) - xi G(beta-) = -1
            assert!(
                approx(
                    g.get_mat(0, 0, 0) - g.get_mat(NTAU, 0, 0) * Complex::new(xi, 0.0),
                    Complex::new(-1.0, 0.0)
                ),
                "KMS"
            );
            // Boundary: Grangle(0,tau) = i xi G^M(beta-tau)
            for m in 0..=NTAU {
                let want =
                    Complex::new(0.0, 1.0) * g.get_mat(NTAU - m, 0, 0) * Complex::new(xi, 0.0);
                assert!(
                    approx(g.get_lmix(0, m, 0, 0), want),
                    "lmix boundary at m = {m}"
                );
            }
            // Equal-time: G<(0,0) = Grangle(0,0)
            assert!(approx(g.get_less(0, 0, 0, 0), g.get_lmix(0, 0, 0, 0)));
        }
    }

    #[test]
    fn from_ham_matrix_hermitian_fermion() {
        let cont = Contour::new(NT, NTAU, 2, 2, TMAX, BETA);
        let mut g = ContourGreen::new(&cont, Sign::Fermi, Complex::ZERO);
        let mut h0 = DenseMat::zeros(2);
        let s = 2.0f64.sqrt();
        h0.set(0, 0, Complex::new(s, 0.0));
        h0.set(0, 1, Complex::new(0.0, s));
        h0.set(1, 0, Complex::new(0.0, -s));
        h0.set(1, 1, Complex::new(-s, 0.0));
        from_ham(&mut g, MU, &h0, BETA, HSTEP);

        // H_mu = H - mu I and rho = f(H - mu)
        let mut hmu = h0.clone();
        for i in 0..2 {
            let v = hmu.get(i, i);
            hmu.set(i, i, v - Complex::new(MU, 0.0));
        }
        let (vals, vecs) = eigh(&hmu);
        let mut rho = DenseMat::zeros(2);
        for a in 0..2 {
            for b in 0..2 {
                let mut v = Complex::ZERO;
                for (l, w) in vals.iter().enumerate().take(2) {
                    v += vecs.get(a, l)
                        * Complex::new(1.0 / ((BETA * *w).exp() + 1.0), 0.0)
                        * vecs.get(b, l).conj();
                }
                rho.set(a, b, v);
            }
        }

        let eye = |i: usize, j: usize| {
            if i == j {
                Complex::new(1.0, 0.0)
            } else {
                Complex::ZERO
            }
        };

        for i in 0..=NT {
            // G^R(t,t) = -i I
            for a in 0..2 {
                for b in 0..2 {
                    assert!(
                        approx(g.get_ret(i, i, a, b), eye(a, b) * Complex::new(0.0, -1.0)),
                        "ret({i},{i})[{a},{b}]"
                    );
                    // G^<(t,t) = i rho, anti-hermitian
                    let v = g.get_less(i, i, a, b);
                    assert!(
                        approx(v, rho.get(a, b) * Complex::new(0.0, 1.0)),
                        "less diag"
                    );
                    assert!(approx(v, -g.get_less(i, i, b, a).conj()), "anti-herm");
                }
            }
        }

        // Retarded: G(t,t') = -i exp(-i (H-mu) (t-t'))
        for i in 0..=NT {
            for j in 0..=i {
                let step = (i - j) as f64 * HSTEP;
                let mut arg = hmu.clone();
                for q in 0..arg.raw().len() {
                    arg.raw_mut()[q] *= Complex::new(0.0, -step);
                }
                let want = {
                    let e = expm(&arg);
                    let mut w = DenseMat::zeros(2);
                    for a in 0..2 {
                        for b in 0..2 {
                            w.set(a, b, e.get(a, b) * Complex::new(0.0, -1.0));
                        }
                    }
                    w
                };
                for a in 0..2 {
                    for b in 0..2 {
                        assert!(
                            approx(g.get_ret(i, j, a, b), want.get(a, b)),
                            "ret({i},{j})[{a},{b}]"
                        );
                    }
                }
            }
        }

        // Lesser: G(t',t) = i e^{-i(H-mu)t'} rho e^{i(H-mu)t}
        // (stored less(i,j) with i <= j, t_i = i h)
        for j in 0..=NT {
            for i in 0..=j {
                let ti = i as f64 * HSTEP;
                let tj = j as f64 * HSTEP;
                let mut argi = hmu.clone();
                let mut argj = hmu.clone();
                for q in 0..4 {
                    argi.raw_mut()[q] *= Complex::new(0.0, -ti);
                    argj.raw_mut()[q] *= Complex::new(0.0, tj);
                }
                let ei = expm(&argi);
                let ej = expm(&argj);
                let prod = ei.mat_mul(&rho).mat_mul(&ej);
                for a in 0..2 {
                    for b in 0..2 {
                        assert!(
                            approx(
                                g.get_less(i, j, a, b),
                                prod.get(a, b) * Complex::new(0.0, 1.0)
                            ),
                            "less({i},{j})[{a},{b}] = {:?} vs {:?}",
                            g.get_less(i, j, a, b),
                            prod.get(a, b) * Complex::new(0.0, 1.0)
                        );
                    }
                }
            }
        }

        // Matsubara: hermitian, G(beta-) = -rho, G(0+) = rho - I
        for m in 0..=NTAU {
            for a in 0..2 {
                for b in 0..2 {
                    assert!(
                        approx(g.get_mat(m, a, b), g.get_mat(m, b, a).conj()),
                        "mat hermitian at m = {m}"
                    );
                }
            }
        }
        for a in 0..2 {
            for b in 0..2 {
                assert!(
                    approx(g.get_mat(NTAU, a, b), -rho.get(a, b)),
                    "mat(beta)[{a},{b}]"
                );
                assert!(
                    approx(g.get_mat(0, a, b), rho.get(a, b) - eye(a, b)),
                    "mat(0)[{a},{b}]"
                );
            }
        }

        // Boundary: Grangle(0,tau) = -i G^M(beta-tau)
        for m in 0..=NTAU {
            for a in 0..2 {
                for b in 0..2 {
                    let want = g.get_mat(NTAU - m, a, b) * Complex::new(0.0, -1.0);
                    assert!(
                        approx(g.get_lmix(0, m, a, b), want),
                        "lmix boundary at m = {m}"
                    );
                }
            }
        }
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

    /// H(t) = H0 + a(t) A + b(t) B with non-commuting H0, A, B: no
    /// closed-form propagator exists. The reference is a classical
    /// fourth-order Runge-Kutta integration of `dU/dt = -i (H - mu) U` on
    /// fine substeps, independent of the exponential propagators and of
    /// the interpolation of [`from_ham_cf`]. Against it the commutator-free
    /// propagators must converge with their formal orders 2 and 4.
    #[test]
    fn cf2_cf4_converge_for_non_commuting_hamiltonian() {
        // m + alpha * k
        fn add_scaled(m: &DenseMat, alpha: f64, k: &DenseMat) -> DenseMat {
            let mut r = m.clone();
            r.axpy(alpha, k);
            r
        }

        // Frobenius norm of m
        fn norm(m: &DenseMat) -> f64 {
            m.raw()
                .iter()
                .map(|z| z.abs() * z.abs())
                .sum::<f64>()
                .sqrt()
        }

        // Reference propagators U(t_n, 0) including the phase e^{i mu t_n}:
        // RK4 on `sub` substeps per time step
        fn rk4_propagators(
            ham: &dyn Fn(f64) -> DenseMat,
            mu: f64,
            hstep: f64,
            nt: usize,
            sub: usize,
            d: usize,
        ) -> Vec<DenseMat> {
            let rhs = |t: f64, u: &DenseMat| scale_ih(&minus_mu(&ham(t), mu).mat_mul(u), 1.0);
            let dt = hstep / sub as f64;
            let mut u = DenseMat::identity(d);
            let mut out = vec![u.clone()];
            for n in 1..=nt {
                let t0 = (n - 1) as f64 * hstep;
                for s in 0..sub {
                    let t = t0 + s as f64 * dt;
                    let k1 = rhs(t, &u);
                    let k2 = rhs(t + 0.5 * dt, &add_scaled(&u, 0.5 * dt, &k1));
                    let k3 = rhs(t + 0.5 * dt, &add_scaled(&u, 0.5 * dt, &k2));
                    let k4 = rhs(t + dt, &add_scaled(&u, dt, &k3));
                    let mut sum = k1;
                    sum.axpy(2.0, &k2);
                    sum.axpy(2.0, &k3);
                    sum.axpy(1.0, &k4);
                    u = add_scaled(&u, dt / 6.0, &sum);
                }
                out.push(u.clone());
            }
            out
        }

        // Largest difference of the elements of two equal-length slices
        fn slice_difference(x: &[Complex], y: &[Complex]) -> f64 {
            x.iter()
                .zip(y)
                .map(|(p, q)| (*p - *q).abs())
                .fold(0.0f64, f64::max)
        }

        // Largest difference of all stored ret / lmix / less elements
        fn max_difference(x: &ContourGreen, y: &ContourGreen) -> f64 {
            slice_difference(x.ret.raw(), y.ret.raw())
                .max(slice_difference(x.lmix.raw(), y.lmix.raw()))
                .max(slice_difference(x.less.raw(), y.less.raw()))
        }

        // three levels; the constant hopping 0 <-> 1 makes H0 non-diagonal
        let d = 3usize;
        let (beta, mu, k, ntau, tmax) = (5.0f64, 0.2, 5i64, 40i64, 2.0f64);
        let mut h0 = DenseMat::zeros(d);
        for (i, e) in [-0.7, 0.1, 0.5].into_iter().enumerate() {
            h0.set(i, i, Complex::new(e, 0.0));
        }
        h0.set(0, 1, Complex::new(0.15, 0.05));
        h0.set(1, 0, Complex::new(0.15, -0.05));
        // A: hopping 1 <-> 2, B: imaginary hopping 0 <-> 2
        let (mut ma, mut mb) = (DenseMat::zeros(d), DenseMat::zeros(d));
        ma.set(1, 2, Complex::ONE);
        ma.set(2, 1, Complex::ONE);
        mb.set(0, 2, Complex::new(0.0, 0.5));
        mb.set(2, 0, Complex::new(0.0, -0.5));
        // cubics: the order-5 interpolation and extrapolation of H is exact
        let a = |t: f64| 0.3 + 0.2 * t - 0.1 * t * t * t;
        let b = |t: f64| -0.1 * t + 0.05 * t * t;
        let ham = |t: f64| {
            let mut m = h0.clone();
            m.axpy(a(t), &ma);
            m.axpy(b(t), &mb);
            m
        };

        // the premise of the test: A and B commute neither with H0 nor
        // with each other
        for (x, y) in [(&h0, &ma), (&h0, &mb), (&ma, &mb)] {
            let mut comm = x.mat_mul(y);
            comm.axpy(-1.0, &y.mat_mul(x));
            assert!(norm(&comm) > 0.1, "the test Hamiltonian must not commute");
        }

        // contour and H(t) for the `nt`-step grid of [0, tmax]; H is exact
        // up to the real step `given` and zero beyond
        let build = |nt: usize, hstep: f64, given: i64| {
            let c = Contour::new(nt as i64, ntau, d as i64, d as i64, tmax, beta);
            // one step more than the contour: the extrapolation of H
            // to the last step reads the step beyond it
            let mut hh = TimeFunction::new(nt + 1, d, d, Complex::ZERO);
            for i in 0..d {
                for j in 0..d {
                    hh.set(MATSUBARA, i, j, h0.get(i, j));
                }
            }
            for t in 0..=given {
                let ht = ham(t as f64 * hstep);
                for i in 0..d {
                    for j in 0..d {
                        hh.set(t, i, j, ht.get(i, j));
                    }
                }
            }
            (c, hh)
        };

        // convergence of the two commutator-free orders
        for cf in [2i64, 4] {
            let mut errors = Vec::new();
            for nt in [20usize, 40] {
                let hstep = tmax / nt as f64;
                let (c, mut hh) = build(nt, hstep, nt as i64);
                let mut g = ContourGreen::new(&c, Sign::Fermi, Complex::ZERO);
                from_ham_cf(&mut g, mu, &mut hh, beta, hstep, true, k, cf);

                let u_ref = rk4_propagators(&ham, mu, hstep, nt, 64, d);
                let mut g_ref = ContourGreen::new(&c, Sign::Fermi, Complex::ZERO);
                fill_from_propagator(&mut g_ref, mu, &h0, beta, &u_ref);
                errors.push(max_difference(&g, &g_ref));
            }
            let (e1, e2) = (errors[0], errors[1]);
            let order = (e1 / e2).log2();
            // measured: cf2 4.3e-4 -> 1.1e-4 (order 2.00),
            //           cf4 4.3e-8 -> 2.7e-9 (order 4.00)
            let (min_order, max_error) = if cf == 2 { (1.9, 5e-4) } else { (3.7, 2e-8) };
            assert!(order > min_order, "cf{cf}: order {order}");
            assert!(e2 < max_error, "cf{cf}: error {e2}");
        }

        // fix_hamiltonian = false: H beyond the first k + 1 steps is
        // extrapolated and must reproduce the exactly given H. The cubic
        // coefficients are extrapolated exactly up to the round-off of the
        // tabulated weights (about 1e-13 at k = 5), which the repeated
        // extrapolation amplifies to about 1e-9 over the 40 steps.
        let (nt, hstep) = (40usize, tmax / 40.0);
        let (c, mut hh) = build(nt, hstep, nt as i64);
        let mut g_exact = ContourGreen::new(&c, Sign::Fermi, Complex::ZERO);
        from_ham_cf(&mut g_exact, mu, &mut hh, beta, hstep, true, k, 4);
        let (_, mut hx) = build(nt, hstep, k + 1);
        let mut g_extrap = ContourGreen::new(&c, Sign::Fermi, Complex::ZERO);
        from_ham_cf(&mut g_extrap, mu, &mut hx, beta, hstep, false, k, 4);
        let e = max_difference(&g_extrap, &g_exact);
        assert!(e < 1e-8, "extrapolated H: {e}");
    }
}
