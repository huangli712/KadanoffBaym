// Project : KadanoffBaym
// File    : src/math/fermi.rs
// Purpose : Fermi-Dirac distribution functions (blueprint: math/fermi.jl)
//!
//! *Remarks* : *Distribution Functions*
//!
//! **Fermionic System**
//!
//! For fermionic system, the basic distribution function reads:
//!
//! ```math
//! \begin{equation}
//! f_1(\beta,\omega) = \frac{1}{1 + e^{\beta\omega}}.
//! \end{equation}
//! ```text
//!
//! We further define ``f_2(\beta,\tau,\omega)``:
//!
//! ```math
//! \begin{equation}
//! f_2(\beta,\tau,\omega) = f_1(\beta,\omega) e^{\tau\omega}.
//! \end{equation}
//! ```text
//!
//! For numerical stability, we should use
//!
//! ```math
//! \begin{equation}
//! f_2(\beta,\tau,\omega) = \frac{e^{\tau\omega}}{ 1 + e^{\beta\omega} },
//! \end{equation}
//! ```text
//!
//! for ``\omega < 0`` and
//!
//! ```math
//! \begin{equation}
//! f_2(\beta,\tau,\omega) = \frac{e^{(\tau-\beta)\omega}}{ 1 + e^{-\beta\omega} },
//! \end{equation}
//! ```text
//!
//! for ``\omega > 0``.

/// Basic Fermi-Dirac distribution `f1(beta, omega) = 1 / (1 + exp(beta*omega))`.
///
/// For `|beta*omega| > 100` returns 0 (`beta*omega > 0`) or 1 to avoid
/// overflow (blueprint convention).
pub(crate) fn fermi(beta: f64, omega: f64) -> f64 {
    let arg = omega * beta;
    if arg.abs() > 100.0 {
        if arg > 0.0 {
            0.0
        } else {
            1.0
        }
    } else {
        1.0 / (1.0 + arg.exp())
    }
}

/// Extended Fermi-Dirac distribution `f2(beta, tau, omega) =
/// f1(beta, omega) * exp(tau*omega)` with stable branching:
///
/// - `omega < 0`: `exp(tau*omega) * f1(beta, omega)`
/// - else: `exp((tau-beta)*omega) * f1(beta, -omega)`
pub fn fermi_tau(beta: f64, tau: f64, omega: f64) -> f64 {
    if omega < 0.0 {
        (omega * tau).exp() * fermi(beta, omega)
    } else {
        ((tau - beta) * omega).exp() * fermi(beta, -omega)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_values_and_limits() {
        assert!((fermi(1.0, 0.0) - 0.5).abs() < 1e-16);
        assert!((fermi(10.0, 1.0) - 1.0 / (1.0 + (10.0f64).exp())).abs() < 1e-17);
        assert_eq!(fermi(10.0, 101.0), 0.0);
        assert_eq!(fermi(10.0, -101.0), 1.0);
    }

    #[test]
    fn extended_branches_match_direct_product() {
        let (beta, tau) = (5.0, 1.3);
        for omega in [-2.0, -0.3, 0.0, 0.7, 3.0] {
            let direct = fermi(beta, omega) * (tau * omega).exp();
            let got = fermi_tau(beta, tau, omega);
            assert!(
                (got - direct).abs() / direct.abs().max(1e-300) < 1e-12,
                "omega={omega}: {got} vs {direct}"
            );
        }
        // Stability: huge arguments stay finite.
        assert!(fermi_tau(5.0, 4.9, 200.0).is_finite());
        assert!(fermi_tau(5.0, 1.0, -200.0).is_finite());
    }
}
