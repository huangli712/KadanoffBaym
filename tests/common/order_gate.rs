/// Shared convergence-order gate (blueprint julia/test/common/order_gate.jl).

/// Outcome of [`check_order_gate`].
pub struct OrderGate {
    pub ok: bool,
    pub observed: f64,
    pub msg: String,
}

#[allow(dead_code)]
/// Shared convergence-order gate (criteria identical to the blueprint):
///
/// 1. at least 4 grids, strictly increasing grid sizes `ns`;
/// 2. all errors are positive and finite;
/// 3. at least 3 errors above the plateau threshold (round-off floor);
/// 4. the least-squares slope of `-log(err)` versus `log(n)` over the
///    non-plateau points is `>= expected - order_tol`;
/// 5. the error on the finest grid is `<= cap`.
#[allow(clippy::too_many_arguments)]
pub fn check_order_gate(
    ns: &[i64],
    errs: &[f64],
    expected: f64,
    cap: f64,
    plateau: f64,
    order_tol: f64,
) -> OrderGate {
    let fail = |observed: f64, msg: &str| OrderGate {
        ok: false,
        observed,
        msg: msg.to_string(),
    };
    if ns.len() != errs.len() {
        return fail(f64::NAN, "length(ns) != length(errs)");
    }
    if ns.len() < 4 {
        return fail(
            f64::NAN,
            &format!("at least 4 grids are required, got {}", ns.len()),
        );
    }
    if ns.windows(2).any(|w| w[1] <= w[0]) {
        return fail(f64::NAN, "grid sizes must be strictly increasing");
    }
    if errs.iter().any(|e| !e.is_finite() || *e <= 0.0) {
        return fail(f64::NAN, "errors must be positive and finite");
    }
    let sel: Vec<usize> = errs
        .iter()
        .enumerate()
        .filter(|(_, e)| **e > plateau)
        .map(|(i, _)| i)
        .collect();
    if sel.len() < 3 {
        return fail(
            f64::NAN,
            &format!(
                "at least 3 non-plateau errors are required, got {}",
                sel.len()
            ),
        );
    }
    let x: Vec<f64> = sel.iter().map(|&i| ns[i] as f64).map(f64::ln).collect();
    let y: Vec<f64> = sel.iter().map(|&i| errs[i].ln()).collect();
    let xm = x.iter().sum::<f64>() / x.len() as f64;
    let ym = y.iter().sum::<f64>() / y.len() as f64;
    let mut num = 0.0;
    let mut den = 0.0;
    for i in 0..x.len() {
        num += (x[i] - xm) * (y[i] - ym);
        den += (x[i] - xm) * (x[i] - xm);
    }
    let observed = -num / den;
    if observed < expected - order_tol {
        return fail(
            observed,
            &format!("observed order {observed} < expected {expected} - {order_tol}"),
        );
    }
    let last = errs[errs.len() - 1];
    if last > cap {
        return fail(
            observed,
            &format!("error on finest grid {last} > cap {cap}"),
        );
    }
    OrderGate {
        ok: true,
        observed,
        msg: String::new(),
    }
}

#[allow(dead_code)]
/// Assert form of [`check_order_gate`] for use in tests.
pub fn assert_order_gate(ns: &[i64], errs: &[f64], expected: f64, cap: f64) {
    let g = check_order_gate(ns, errs, expected, cap, 1e-13, 0.3);
    assert!(
        g.ok,
        "order gate failed: {} (observed {})",
        g.msg, g.observed
    );
}
