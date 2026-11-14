// Project : KadanoffBaym
// File    : src/math/bose.rs
// Purpose : Bose-Einstein distribution functions (blueprint: math/bose.jl)

/// Basic Bose-Einstein distribution `b1(beta, omega) = 1/(exp(beta*omega)-1)`
/// with stable branches:
///
/// - `omega < 0`: `-1 - b1(beta, -omega)`;
/// - `|arg| > 100`: `0`;
/// - `arg < 1e-10`: `1/arg`;
/// - else `1/(exp(arg) - 1)`.
pub fn bose(beta: f64, omega: f64) -> f64 {
    let arg = omega * beta;
    if arg < 0.0 {
        return -1.0 - bose(beta, -omega);
    }
    if arg.abs() > 100.0 {
        0.0
    } else if arg < 1.0e-10 {
        1.0 / arg
    } else {
        1.0 / (arg.exp() - 1.0)
    }
}

/// Extended Bose-Einstein distribution `b2(beta, tau, omega)` with the
/// stable branching:
///
/// - `omega < 0`: `exp(tau*omega) * b1(beta, omega)`
/// - else: `-exp((tau-beta)*omega) * b1(beta, -omega)`
pub fn bose_tau(beta: f64, tau: f64, omega: f64) -> f64 {
    if omega < 0.0 {
        (tau * omega).exp() * bose(beta, omega)
    } else {
        -((tau - beta) * omega).exp() * bose(beta, -omega)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_values_and_branches() {
        // negative argument reflection: b(β,-ω) = -1 - b(β,ω)
        let b1 = bose(2.0, 0.3);
        let bm = bose(2.0, -0.3);
        assert!((bm - (-1.0 - b1)).abs() < 1e-15);
        // tiny argument -> 1/arg
        assert!((bose(2.0, 1e-12) - 1.0 / 2.0e-12).abs() / (1.0 / 2.0e-12) < 1e-12);
        // huge argument -> 0
        assert_eq!(bose(2.0, 100.0), 0.0);
        // direct formula
        assert!((b1 - 1.0 / ((0.6f64).exp() - 1.0)).abs() < 1e-16);
    }

    #[test]
    fn extended_branch_matches_product() {
        let (beta, tau) = (5.0, 1.3);
        for omega in [-2.0, -0.3, 0.7, 3.0] {
            let got = bose_tau(beta, tau, omega);
            // direct (possibly less stable) evaluation for comparison
            let direct = if omega < 0.0 {
                (tau * omega).exp() / (((beta * omega).exp()) - 1.0)
            } else {
                -((tau - beta) * omega).exp() / (((-beta * omega).exp()) - 1.0)
            };
            assert!(
                (got - direct).abs() / direct.abs().max(1e-300) < 1e-10,
                "omega={omega}: {got} vs {direct}"
            );
        }
        assert!(bose_tau(5.0, 4.9, 200.0).is_finite());
        assert!(bose_tau(5.0, 1.0, -200.0).is_finite());
    }
}
