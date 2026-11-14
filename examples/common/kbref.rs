//
// Reference-data comparison helpers of the tutor validations (the Rust
// port of the reading/compare parts of julia/tutor/common/kbref.jl,
// example_tools.jl, holstein_common.jl, and order_gate.jl). The kbref
// reading itself is the shared `tools/kbref.rs` (included as
// `kbref_raw` by this module and by the integration tests).
//

use kadanoffbaym::complex::Complex;
use super::kbref_raw::{read_kbref, KbrefData, KbrefRow};
use kadanoffbaym::types::ness::NessHermMat;
use std::path::Path;

/// The tutor data directory (`data/tutor` at the repository root).
pub fn tutor_data_dir() -> String {
    // examples/ runs from the crate root; also work from any cwd by
    // walking up to the Cargo.toml
    let mut dir = std::env::current_dir().unwrap();
    loop {
        if dir.join("Cargo.toml").is_file() && dir.join("data/tutor").is_dir() {
            return dir.join("data/tutor").to_string_lossy().to_string();
        }
        if !dir.pop() {
            break;
        }
    }
    "data/tutor".to_string()
}

/// Read a kbref file from `data/tutor`.
pub fn read_tutor_kbref(name: &str) -> Result<KbrefData, String> {
    let p = format!("{}/{}", tutor_data_dir(), name);
    if !Path::new(&p).is_file() {
        return Err(format!("{p}: no such file"));
    }
    read_kbref(&p).map_err(|e| e.to_string())
}

/// Numeric rows of a text file (comment lines are skipped).
pub fn read_rows(fname: &str) -> Result<Vec<Vec<f64>>, String> {
    let mut rows = Vec::new();
    for line in std::fs::read_to_string(fname).map_err(|e| format!("{fname}: {e}"))?.lines() {
        if line.starts_with('#') || line.trim().is_empty() {
            continue;
        }
        let mut row = Vec::new();
        for w in line.split_whitespace() {
            row.push(w.parse::<f64>().map_err(|_| format!("{fname}: bad value `{w}`"))?);
        }
        rows.push(row);
    }
    Ok(rows)
}

/// Compare every reference record with the value `kbval(row)`
/// (atol = rtol = 1e-10). Returns the failure messages.
pub fn example_compare<F>(refdata: &KbrefData, kbval: F) -> Vec<String>
where
    F: Fn(&KbrefRow) -> Option<Complex>,
{
    let mut fails = Vec::new();
    for r in refdata.rows.iter() {
        let x = kbval(r);
        let ok = match &x {
            Some(v) => (*v - r.value).abs() <= 1e-10 + 1e-10 * r.value.abs(),
            None => false,
        };
        if !ok {
            fails.push(format!(
                "{} ({}, {}, {}, {}): kb = {:?}, nessi = {}",
                r.component, r.i, r.j, r.a, r.b, x, r.value
            ));
        }
    }
    if !fails.is_empty() {
        println!(
            "example reference failures:\n  {}",
            fails.iter().take(10).cloned().collect::<Vec<_>>().join("\n  ")
        );
    }
    fails
}

/// `G^R`/`G^<` of a steady-state Green's function, component
/// `gf_{ret,les}_{time,freq}` (NESSi time step in `r.j`).
pub fn example_gf(g: &NessHermMat, r: &KbrefRow, d: usize) -> Complex {
    let time = r.component.ends_with("time");
    let arr = if r.component.contains("ret") {
        if time {
            &g.ret.time
        } else {
            &g.ret.freq
        }
    } else if time {
        &g.les.time
    } else {
        &g.les.freq
    };
    arr[(r.j as usize) * d * d]
}

//
// the shared convergence-order gate (order_gate.jl)
//

/// Evaluate the convergence-order gate. Returns `(ok, observed, msg)`.
pub fn check_order_gate(
    ns: &[i64],
    errs: &[f64],
    expected: f64,
    cap: f64,
    plateau: f64,
    order_tol: f64,
) -> (bool, f64, String) {
    if ns.len() != errs.len() {
        return (false, f64::NAN, "length(Ns) != length(errs)".to_string());
    }
    if ns.len() < 4 {
        return (
            false,
            f64::NAN,
            format!("at least 4 grids are required, got {}", ns.len()),
        );
    }
    if !ns.windows(2).all(|w| w[1] > w[0]) {
        return (false, f64::NAN, "grid sizes must be strictly increasing".to_string());
    }
    if !errs.iter().all(|e| e.is_finite() && *e > 0.0) {
        return (false, f64::NAN, "errors must be positive and finite".to_string());
    }
    let sel: Vec<usize> = errs
        .iter()
        .enumerate()
        .filter(|(_, e)| **e > plateau)
        .map(|(i, _)| i)
        .collect();
    if sel.len() < 3 {
        return (
            false,
            f64::NAN,
            format!(
                "at least 3 non-plateau errors are required, got {}",
                sel.len()
            ),
        );
    }
    let x: Vec<f64> = sel.iter().map(|&i| ns[i] as f64).map(|v| v.ln()).collect();
    let y: Vec<f64> = sel.iter().map(|&i| errs[i]).map(|v: f64| v.ln()).collect();
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
        return (
            false,
            observed,
            format!("observed order {observed} < expected {expected} - {order_tol}"),
        );
    }
    if errs[errs.len() - 1] > cap {
        return (
            false,
            observed,
            format!(
                "finest-grid error {} > cap {cap}",
                errs[errs.len() - 1]
            ),
        );
    }
    (true, observed, "ok".to_string())
}

/// Record the convergence-order gate as a panic on failure; returns the
/// observed order.
pub fn test_order_gate(ns: &[i64], errs: &[f64], expected: f64, cap: f64) -> f64 {
    let (ok, observed, msg) = check_order_gate(ns, errs, expected, cap, 1e-13, 0.3);
    assert!(
        ok,
        "order gate failed: {msg} (Ns = {ns:?}, errs = {errs:?}, expected = {expected})"
    );
    observed
}

//
// the Holstein comparison helpers (holstein_common.jl)
//

/// Value of a kbref row in a result: `scalars` maps the component to
/// the vector indexed by the NESSi time step + 1, `rho(t, a, b)` gives
/// the density matrix element.
pub fn hol_value(
    row: &KbrefRow,
    scalars: &std::collections::BTreeMap<String, Vec<f64>>,
    rho: &dyn Fn(i64, usize, usize) -> f64,
) -> f64 {
    if row.component == "rho" {
        return rho(row.i, row.a as usize, row.b as usize);
    }
    scalars[&row.component][(row.i + 2) as usize]
}

/// Pointwise comparison; returns (pass, maximal error, component, step).
pub fn hol_compare(
    d: &KbrefData,
    scalars: &std::collections::BTreeMap<String, Vec<f64>>,
    rho: &dyn Fn(i64, usize, usize) -> f64,
) -> (bool, (f64, String, i64)) {
    let mut ok = true;
    let mut worst = (0.0f64, String::new(), -2i64);
    for row in d.rows.iter() {
        let v = hol_value(row, scalars, rho);
        let e = (v - row.value).abs();
        ok &= e <= 1e-10 + 1e-10 * row.value.abs();
        if e > worst.0 {
            worst = (e, row.component.clone(), row.i);
        }
    }
    (ok, worst)
}

/// The same data with the rows of the component `c` at the NESSi time
/// step 1 changed by `delta` (for `rho` only the off-diagonal elements).
pub fn hol_tamper(d: &KbrefData, c: &str, delta: f64) -> KbrefData {
    let rows = d
        .rows
        .iter()
        .map(|x| {
            let hit = x.component == c && x.i == 1 && (c != "rho" || x.a != x.b);
            if hit {
                KbrefRow {
                    component: x.component.clone(),
                    i: x.i,
                    j: x.j,
                    a: x.a,
                    b: x.b,
                    value: x.value + Complex::new(delta, 0.0),
                }
            } else {
                x.clone()
            }
        })
        .collect();
    KbrefData {
        path: d.path.clone(),
        metadata: d.metadata.clone(),
        rows,
    }
}
