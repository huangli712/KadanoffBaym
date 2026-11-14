// Project : KadanoffBaym
// File    : src/equilibrium/standard.rs
// Purpose : free equilibrium Green's function from a constant
// Hamiltonian (blueprint: julia/src/equilibrium/from_ham.jl, constant-H
// entry `from_ham!(G, mu, H0, beta, h)`; pulled forward from R5 because
// t_traits uses it as its content generator).
//
//   K = mu*I - H0 = R diag(lambda) R^dagger,  U_n = e^{i h K} ... U_0
//
//   G^M(tau)  = -R diag[e^{tau lambda} n_F(lambda)] R^dagger     (fermions)
//   G^mix(t,tau) = i U_n R diag[e^{-tau lambda} n_F(-lambda)] R^dagger
//   G^R(t_n,t_j) = -i U_n U_j^dagger
//   G^<(t_j,t_n) = i U_j N U_n^dagger,  N = R diag[n_F(-lambda)] R^dagger
//
// For bosons `G^M` and `N` change sign, `n_F` becomes `n_B`, and
// `G^mix` is multiplied by -1.

use crate::complex::c64::C64;
use crate::linalg::eigh::eigh;
use crate::linalg::expm::expm;
use crate::linalg::mat::DenseMat;
use crate::types::contour::Sign;
use crate::types::green::ContourGreen;

/// `fermi_exp(beta, tau, omega)` (blueprint alias of `fermi`).
pub fn fermi_exp(beta: f64, tau: f64, omega: f64) -> f64 {
    crate::math::fermi_tau(beta, tau, omega)
}

/// `bose_exp(beta, tau, omega)`; the zero mode is a pole (panics like
/// the blueprint's `DomainError`).
pub fn bose_exp(beta: f64, tau: f64, omega: f64) -> f64 {
    assert!(omega != 0.0, "bose_exp: zero mode of the Bose function");
    crate::math::bose_tau(beta, tau, omega)
}

/// `sign * R diag(d) R^dagger` for real eigenvalue scales `d`.
fn r_diag_rdag(r: &DenseMat, d: &[f64], sign: f64) -> DenseMat {
    let n = r.nrows();
    let mut rd = DenseMat::zeros(n);
    for j in 0..n {
        for i in 0..n {
            rd.set(i, j, r.get(i, j).scale(d[j]));
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
            assert!(v.re.is_finite() && v.im.is_finite(), "from_ham: H is not finite");
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

/// Free Green's function of the constant hermitian Hamiltonian `h0`
/// (blueprint `from_ham!(G, mu, H0, beta, h)`).
pub fn from_ham(g: &mut ContourGreen, mu: f64, h0: &DenseMat, beta: f64, h: f64) {
    check_free_hamiltonian(g, h0);
    assert!(beta > 0.0, "from_ham: beta must be positive");
    let d = g.size1;
    let nt = g.nt as usize;

    // U_n = exp(i h (mu - H0))^n with U[0] = I
    let km = mu_minus(h0, mu);
    let udt = expm(&scale_ih(&km, h));
    let mut u = vec![DenseMat::identity(d)];
    for n in 1..=nt {
        u.push(u[n - 1].mat_mul(&udt));
    }
    fill_from_propagator(g, mu, h0, beta, &u);
}

/// `(-i * hstep) * m`.
fn scale_ih(m: &DenseMat, hstep: f64) -> DenseMat {
    let w = C64::new(0.0, -hstep);
    DenseMat::from_vec(m.nrows(), m.ncols(), m.raw().iter().map(|&z| z * w).collect())
}

/// `mu * I - m`.
fn mu_minus(m: &DenseMat, mu: f64) -> DenseMat {
    let d = m.nrows();
    let mut r = m.clone();
    for i in 0..d {
        let v = r.get(i, i);
        r.set(i, i, v - C64::new(mu, 0.0));
    }
    r
}

/// Fill all the components of the free Green's function `g` from the
/// Hamiltonian `h0` at the Matsubara axis and the propagators `u[n]`
/// including the phase `e^{i mu t}` (blueprint
/// `_fill_green_from_propagator!`).
pub fn fill_from_propagator(
    g: &mut ContourGreen,
    mu: f64,
    h0: &DenseMat,
    beta: f64,
    u: &[DenseMat],
) {
    let d = g.size1;
    let nt = g.nt as usize;
    let fermion = g.sign == Sign::Fermi;

    // K = mu*I - H0 and its eigendecomposition
    let k = {
        let mut kk = DenseMat::zeros(d);
        for i in 0..d {
            for j in 0..d {
                let diag = if i == j { C64::new(mu, 0.0) } else { C64::ZERO };
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
                x.set(i, j, C64::new(-v.im, v.re));
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
                    g.set_ret(row, col, i, j, C64::new(v.im, -v.re));
                }
            }
            // G^<(t_col, t_row) = i U_col N U_row^dagger
            let un = u[col as usize].mat_mul(&nmat).mat_mul(&u[row as usize].adjoint());
            for i in 0..d {
                for j in 0..d {
                    let v = un.get(i, j);
                    g.set_less(col, row, i, j, C64::new(-v.im, v.re));
                }
            }
        }
    }
}


// ---------------------------------------------------------------------------
// Time-dependent Hamiltonian: commutator-free propagators (blueprint
// from_ham.jl, the `from_ham!(G, mu, H::Cf, ...)` entry)
// ---------------------------------------------------------------------------

use crate::math::interp::interpolation;
use crate::math::{bose, fermi};
use crate::types::green::TimeFunction;
use crate::types::layout::MATSUBARA;
use crate::weights::Integrator;

/// Prepare the Hamiltonian for the propagation from `tstp - 1` to
/// `tstp` (NESSi indices). If `fix = false`, `h[tstp]` is extrapolated
/// from the previous time steps when `tstp > kt`. Returns the 0-based
/// index of the last grid point of the interpolation interval
/// (blueprint `_hamiltonian_cf`).
fn hamiltonian_cf(tstp: i64, h: &mut TimeFunction, kt: i64, fix: bool) -> i64 {
    if !fix && tstp > kt {
        crate::math::extrap::extrapolate_timestep_cf(tstp, h, kt);
    }
    tstp.max(kt)
}

/// Second order commutator-free propagation `U_n = e^{-i h H(t_{n-1/2})}
/// U_{n-1}` (blueprint `_propagator_cf2!`).
fn propagator_cf2(
    u: &mut Vec<DenseMat>,
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

/// Fourth order commutator-free propagation
/// `U_n = e^{-ih(a1 H1 + a2 H2)} e^{-ih(a2 H1 + a1 H2)} U_{n-1}`
/// (blueprint `_propagator_cf4!`).
fn propagator_cf4(
    u: &mut Vec<DenseMat>,
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
    let h0 = DenseMat::from_vec(d, d, h.block(MATSUBARA).to_vec());
    check_free_hamiltonian(g, &h0);
    let nmax = if fix_hamiltonian { (kt + 1).max(nt) } else { kt + 1 };
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
    for n in 0..=(g.nt) {
        let ph = C64::from_polar(1.0, mu * hstep * n as f64);
        for z in u[n].raw_mut().iter_mut() {
            *z *= ph;
        }
    }
    fill_from_propagator(g, mu, &h0, beta, &u);
}

// ---------------------------------------------------------------------------
// Bosonic single pole (blueprint single_pole.jl)
// ---------------------------------------------------------------------------

/// Matsubara component of the bosonic single pole Green's function
/// (blueprint `green_single_pole_mat!`); only the (0,0) element is
/// written.
pub fn green_single_pole_mat(d0: &mut ContourGreen, w: f64, beta: f64) {
    assert!(d0.sign == Sign::Bose, "D0 must be bosonic");
    assert!(w * beta > 0.0, "w*beta must be positive");
    let dtau = beta / d0.ntau as f64;
    for m in 0..=(d0.ntau as i64) {
        let tau = m as f64 * dtau;
        let ewv = bose_exp(beta, tau, w);
        let emwv = -bose_exp(beta, tau, -w);
        d0.set_mat(m, 0, 0, C64::new(-0.5 * (ewv + emwv), 0.0));
    }
}

/// Real time components of the bosonic single pole Green's function at
/// the time step `tstp` (0-based; blueprint `green_single_pole_timestep!`).
pub fn green_single_pole_timestep(
    tstp: i64,
    d0: &mut ContourGreen,
    w: f64,
    beta: f64,
    hstep: f64,
) {
    assert!(d0.sign == Sign::Bose, "D0 must be bosonic");
    assert!(w * beta > 0.0, "w*beta must be positive");
    assert!(0 <= tstp && tstp <= d0.nt as i64, "wrong time step {tstp}");
    let ntau = d0.ntau as i64;
    let dtau = beta / d0.ntau as f64;
    let bw = bose(beta, w);
    let i = C64::new(0.0, 1.0);

    for tp in 0..=tstp {
        let c = (w * (tstp - tp) as f64 * hstep).cos();
        let s = (w * (tstp - tp) as f64 * hstep).sin();
        let d_les = -i * (C64::new(c, -s) + C64::new(2.0 * c * bw, 0.0));
        let d_gtr = d_les;
        let d_ret = d_gtr + d_les.conj();
        d0.set_less(tp, tstp, 0, 0, d_les * C64::new(0.5, 0.0));
        d0.set_ret(tstp, tp, 0, 0, d_ret * C64::new(0.5, 0.0));
    }

    let c = (w * tstp as f64 * hstep).cos();
    let s = (w * tstp as f64 * hstep).sin();
    for m in 0..=ntau {
        let ewv = (w * m as f64 * dtau).exp();
        let emwv = (-w * m as f64 * dtau).exp();
        let d_tv = -i
            * (C64::new(c, s) * emwv
                + (C64::new(c, s) * emwv + C64::new(c, -s) * ewv) * bw);
        d0.set_lmix(tstp, m, 0, 0, d_tv * C64::new(0.5, 0.0));
    }
}

/// All the components of the bosonic single pole Green's function
/// (blueprint `green_single_pole!`).
pub fn green_single_pole(d0: &mut ContourGreen, w: f64, beta: f64, hstep: f64) {
    green_single_pole_mat(d0, w, beta);
    for tstp in 0..=(d0.nt as i64) {
        green_single_pole_timestep(tstp, d0, w, beta, hstep);
    }
}

// ---------------------------------------------------------------------------
// Equilibrium Green's functions from a DOS (blueprint from_dos.jl)
// ---------------------------------------------------------------------------

use crate::fourier::adft::{adft_eval, adft_sample};
use crate::math::dos::BetheDos;

/// The component selector of the DOS integrand.
pub enum DosComp {
    Ret,
    Les,
    Tv,
    Mat,
}

/// Integrand of the equilibrium Green's function (blueprint
/// `_dos_integrand`); the DOS must be finite and nonnegative, and the
/// bosonic zero mode is rejected.
pub fn dos_integrand(
    dos: &dyn Fn(f64) -> f64,
    eps: f64,
    comp: DosComp,
    sign: Sign,
    mu: f64,
    beta: f64,
    tau: f64,
) -> C64 {
    let a = dos(eps);
    assert!(a.is_finite(), "DOS is not finite at eps = {eps}");
    assert!(a >= 0.0, "DOS is negative at eps = {eps}");
    let x = eps - mu;
    let fermion = sign == Sign::Fermi;
    if !fermion && !matches!(comp, DosComp::Ret) && x == 0.0 {
        panic!("zero mode of the Bose function at eps = {eps}");
    }
    let nxi = |y: f64| {
        if fermion {
            fermi(beta, y)
        } else {
            bose(beta, y)
        }
    };
    let xi = sign.as_i64() as f64;
    let fm = match comp {
        DosComp::Ret => 1.0,
        DosComp::Les => xi * nxi(x),
        DosComp::Tv => {
            xi * if fermion {
                fermi_exp(beta, tau, x)
            } else {
                bose_exp(beta, tau, x)
            }
        }
        DosComp::Mat => {
            if x > 0.0 {
                xi * (-tau * x).exp() * nxi(-x)
            } else {
                -((beta - tau) * x).exp() * nxi(x)
            }
        }
    };
    C64::new(a * fm, 0.0)
}

fn check_equilibrium(g: &ContourGreen, beta: f64, limit: i64, nn: i64) {
    assert!(beta > 0.0, "beta must be positive");
    assert!(g.size1 == g.size2, "G must be square");
    assert!(nn % 2 == 0 && nn >= 20, "nn must be even and >= 20, got {nn}");
    assert!(limit >= 1, "limit must be >= 1");
}

/// Matsubara component of the equilibrium Green's function for the
/// density of states `dos` with the support `[lo, hi]` (blueprint
/// `from_dos_mat!`).
#[allow(clippy::too_many_arguments)]
pub fn from_dos_mat(
    g: &mut ContourGreen,
    dos: &dyn Fn(f64) -> f64,
    lo: f64,
    hi: f64,
    beta: f64,
    mu: f64,
    limit: i64,
    nn: i64,
) {
    check_equilibrium(g, beta, limit, nn);
    let sign = g.sign;
    let d = g.size1;
    let dtau = beta / g.ntau as f64;
    for m in 0..=(g.ntau as i64) {
        let tau = m as f64 * dtau;
        let fz = |eps: f64| dos_integrand(dos, eps, DosComp::Mat, sign, mu, beta, tau);
        let p = adft_sample(fz, lo, hi, nn, limit, 0.0, 20);
        let (res, _) = adft_eval(&p, 0.0);
        for i in 0..d {
            for j in 0..d {
                let v = if i == j { res } else { C64::ZERO };
                g.set_mat(m, i, j, v);
            }
        }
    }
}

/// All the components of the equilibrium Green's function for the
/// density of states `dos` (blueprint `from_dos!`).
#[allow(clippy::too_many_arguments)]
pub fn from_dos(
    g: &mut ContourGreen,
    dos: &dyn Fn(f64) -> f64,
    lo: f64,
    hi: f64,
    beta: f64,
    hstep: f64,
    mu: f64,
    limit: i64,
    nn: i64,
) {
    check_equilibrium(g, beta, limit, nn);
    assert!(hstep > 0.0, "h must be positive");
    let sign = g.sign;
    let d = g.size1;
    let nt = g.nt as i64;
    let ntau = g.ntau as i64;
    let dtau = beta / g.ntau as f64;
    let i = C64::new(0.0, 1.0);

    // Matsubara component
    from_dos_mat(g, dos, lo, hi, beta, mu, limit, nn);

    // retarded component, G^R(t,t') with l = t - t'
    let fz_ret = |eps: f64| dos_integrand(dos, eps, DosComp::Ret, sign, mu, beta, 0.0);
    let p = adft_sample(fz_ret, lo, hi, nn, limit, 0.0, 20);
    for l in 0..=nt {
        let (res, _) = adft_eval(&p, -hstep * l as f64);
        let res = -i * res;
        for row in l..=nt {
            for a in 0..d {
                for b in 0..d {
                    let v = if a == b { res } else { C64::ZERO };
                    g.set_ret(row, row - l, a, b, v);
                }
            }
        }
    }

    // left-mixing component
    for m in 0..=ntau {
        let tau = m as f64 * dtau;
        let fz_tv = |eps: f64| dos_integrand(dos, eps, DosComp::Tv, sign, mu, beta, tau);
        let p = adft_sample(fz_tv, lo, hi, nn, limit, 0.0, 20);
        for n in 0..=nt {
            let (res, _) = adft_eval(&p, -n as f64 * hstep);
            let res = -i * res;
            for a in 0..d {
                for b in 0..d {
                    let v = if a == b { res } else { C64::ZERO };
                    g.set_lmix(n, m, a, b, v);
                }
            }
        }
    }

    // lesser component, G^<(t',t) with l = t - t'
    let fz_les = |eps: f64| dos_integrand(dos, eps, DosComp::Les, sign, mu, beta, 0.0);
    let p = adft_sample(fz_les, lo, hi, nn, limit, 0.0, 20);
    for l in 0..=nt {
        let (res, _) = adft_eval(&p, hstep * l as f64);
        let res = -i * res;
        for col in l..=nt {
            for a in 0..d {
                for b in 0..d {
                    let v = if a == b { res } else { C64::ZERO };
                    g.set_less(col - l, col, a, b, v);
                }
            }
        }
    }
}

/// All the components of the equilibrium Green's function of the Bethe
/// lattice `rho(eps) = sqrt(4V^2 - (eps - E0)^2) / (2 pi V^2)`
/// (blueprint `from_dos_bethe!`).
#[allow(clippy::too_many_arguments)]
pub fn from_dos_bethe(
    g: &mut ContourGreen,
    beta: f64,
    hstep: f64,
    v: f64,
    e0: f64,
    mu: f64,
    limit: i64,
    nn: i64,
) {
    assert!(v > 0.0, "V must be positive");
    let rho = BetheDos::new(e0 - 2.0 * v, e0 + 2.0 * v);
    let lo = rho.lo;
    let hi = rho.hi;
    from_dos(
        g,
        &|eps: f64| rho.at(eps),
        lo,
        hi,
        beta,
        hstep,
        mu,
        limit,
        nn,
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::contour::Contour;

    /// H0 = diag(-0.4, 0.6), mu = 0.1: K = diag(0.5, -0.5) with
    /// lambda = (-0.5, 0.5) ascending and R swapping the two orbitals.
    #[test]
    fn diagonal_hamiltonian_matches_analytic_fermi() {
        let (beta, dt, mu) = (4.0, 0.025, 0.1);
        let c = Contour::new(20, 100, 2, 2, 20.0 * dt, beta);
        let mut g = ContourGreen::new(&c, Sign::Fermi, C64::ZERO);
        let mut h0 = DenseMat::zeros(2);
        h0.set(0, 0, C64::new(-0.4, 0.0));
        h0.set(1, 1, C64::new(0.6, 0.0));
        from_ham(&mut g, mu, &h0, beta, dt);

        let dtau = beta / 100.0;
        for m in [0i64, 25, 50, 100] {
            let tau = m as f64 * dtau;
            // mat = -R diag(e^{tau lam} n_F(lam)) R^dagger with the swap
            let v_l1 = -(-0.5f64 * tau).exp() * fermi(beta, -0.5); // lam = -0.5 -> orbital 1
            let v_l2 = -(0.5f64 * tau).exp() * fermi(beta, 0.5);
            assert!(
                (g.get_mat(m, 0, 0) - C64::new(v_l2, 0.0)).abs() < 1e-12,
                "mat(0,0) at m = {m}"
            );
            assert!(
                (g.get_mat(m, 1, 1) - C64::new(v_l1, 0.0)).abs() < 1e-12,
                "mat(1,1) at m = {m}"
            );
            assert!(g.get_mat(m, 0, 1).abs() < 1e-13);
            assert!(g.get_mat(m, 1, 0).abs() < 1e-13);
        }

        // ret(n, n) = -i * Id
        for n in [0i64, 5, 20] {
            assert!((g.get_ret(n, n, 0, 0) - C64::new(0.0, -1.0)).abs() < 1e-12);
            assert!((g.get_ret(n, n, 1, 1) - C64::new(0.0, -1.0)).abs() < 1e-12);
            assert!(g.get_ret(n, n, 0, 1).abs() < 1e-13);
        }

        // less(0, 0) = i * N with N = R diag(n_F(-lam)) R^dagger
        let i = C64::new(0.0, 1.0);
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
        h0.set(0, 0, C64::new(-0.4, 0.0));
        h0.set(0, 1, C64::new(0.0, 0.1));
        h0.set(1, 0, C64::new(0.0, -0.1));
        h0.set(1, 1, C64::new(0.6, 0.0));
        let amp = 0.3f64;

        // H at the Matsubara axis and at the real steps 0..=max(k+1, nt)
        let nt_h = ((k + 1) as usize).max(nt);
        let mut h = TimeFunction::new(nt_h, 2, 2, C64::ZERO);
        for t in 0..=(nt_h as i64) {
            let a = amp * (t as f64 * hstep) * (t as f64 * hstep);
            for i in 0..2 {
                for j in 0..2 {
                    let mut v = h0.get(i, j);
                    if i == j {
                        v += C64::new(a, 0.0);
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
                let ph = C64::from_polar(1.0, mu * t - phi);
                for z in u.raw_mut().iter_mut() {
                    *z *= ph;
                }
                u
            })
            .collect();

        for cf in [2i64, 4] {
            let mut g_cf = ContourGreen::new(&c, Sign::Fermi, C64::ZERO);
            let mut hh = h.clone();
            from_ham_cf(&mut g_cf, mu, &mut hh, beta, hstep, true, k, cf);
            let mut g_an = ContourGreen::new(&c, Sign::Fermi, C64::ZERO);
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

    /// The bosonic single pole against the closed forms of the
    /// blueprint Remarks, including the periodicity of the Matsubara
    /// component.
    #[test]
    fn single_pole_matches_closed_forms() {
        let (w, beta, hstep, nt, ntau) = (1.3f64, 2.0, 0.05, 20usize, 50usize);
        let c = Contour::new(nt as i64, ntau as i64, 1, 1, nt as f64 * hstep, beta);
        let mut d0 = ContourGreen::new(&c, Sign::Bose, C64::ZERO);
        green_single_pole(&mut d0, w, beta, hstep);

        let i = C64::new(0.0, 1.0);
        let bw = bose(beta, w);
        let dtau = beta / ntau as f64;

        // D^M(tau) = -0.5 [e^{tau w} n_B(w) - e^{-tau w} n_B(-w)]
        for m in [0usize, 13, 25, 50] {
            let tau = m as f64 * dtau;
            let want = -0.5
                * ((w * tau).exp() * bw - (-w * tau).exp() * crate::math::bose(beta, -w));
            assert!((d0.get_mat(m as i64, 0, 0) - C64::new(want, 0.0)).abs() < 1e-12);
        }
        // bosonic periodicity D^M(beta - tau) = D^M(tau)
        for m in [1usize, 10, 25] {
            let d = d0.get_mat(m as i64, 0, 0) - d0.get_mat((ntau - m) as i64, 0, 0);
            assert!(d.abs() < 1e-12, "periodicity at m = {m}");
        }
        // D^R(t,t') = -sin w (t - t'), D^<(t',t) = -i/2 [e^{-i w dt} + 2 n_B cos w dt]
        for (tstp, tp) in [(0i64, 0), (5, 0), (20, 7), (20, 20)] {
            let dt = (tstp - tp) as f64 * hstep;
            let cw = (w * dt).cos();
            let sw = (w * dt).sin();
            let ret_want = -sw;
            let les_want = -0.5 * i * (C64::new(cw, -sw) + C64::new(2.0 * cw * bw, 0.0));
            assert!((d0.get_ret(tstp, tp, 0, 0) - C64::new(ret_want, 0.0)).abs() < 1e-12);
            assert!((d0.get_less(tp, tstp, 0, 0) - les_want).abs() < 1e-12);
        }
        // D^rangle(t,tau) closed form
        for (tstp, m) in [(0i64, 0usize), (7, 13), (20, 50)] {
            let t = tstp as f64 * hstep;
            let tau = m as f64 * dtau;
            let c = (w * t).cos();
            let s = (w * t).sin();
            let want = -0.5
                * i
                * (C64::new(c, s) * (-w * tau).exp()
                    + (C64::new(c, s) * (-w * tau).exp() + C64::new(c, -s) * (w * tau).exp())
                        * bw);
            assert!((d0.get_lmix(tstp, m as i64, 0, 0) - want).abs() < 1e-12);
        }
    }

    /// Bethe lattice: G^R(t,t) = -i (the DOS is normalized), and the
    /// fermionic relation mat(0) + mat(beta) = -integral rho = -1 links
    /// the Matsubara endpoints; less(0,0) = i (mat(0) + 1).
    #[test]
    fn from_dos_bethe_normalization() {
        let (beta, hstep, nt, ntau) = (2.0f64, 0.05, 10usize, 100usize);
        let c = Contour::new(nt as i64, ntau as i64, 2, 2, nt as f64 * hstep, beta);
        let mut g = ContourGreen::new(&c, Sign::Fermi, C64::ZERO);
        from_dos_bethe(&mut g, beta, hstep, 1.0, 0.0, 0.0, 100, 20);

        let i = C64::new(0.0, 1.0);
        for n in [0i64, 5, 10] {
            assert!(
                (g.get_ret(n, n, 0, 0) + i).abs() < 1e-6,
                "ret({n},{n}) = {:?}",
                g.get_ret(n, n, 0, 0)
            );
            assert!((g.get_ret(n, n, 1, 1) + i).abs() < 1e-6);
        }
        let m0 = g.get_mat(0, 0, 0);
        let mb = g.get_mat(ntau as i64, 0, 0);
        assert!(
            (m0 + mb - C64::new(-1.0, 0.0)).abs() < 1e-6,
            "mat(0) + mat(beta) = {:?}",
            m0 + mb
        );
        let l00 = g.get_less(0, 0, 0, 0);
        assert!(
            (l00 - i * (m0 + C64::new(1.0, 0.0))).abs() < 1e-6,
            "less(0,0) = {l00:?} vs i(mat(0)+1)"
        );
    }
}
