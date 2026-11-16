// Project : KadanoffBaym
// File    : util/compare.rs
// Purpose : reference-data comparison against kbref rows
//

use std::collections::BTreeMap;
use std::env;
use std::fs;
use std::path::Path;

use kadanoffbaym::complex::Complex;
use kadanoffbaym::types::SteadyGreen;

use super::kbref::{KbrefData, KbrefRow, read_kbref};

/// The tutor data directory (`data/tutor` at the repository root).
pub fn tutor_data_dir() -> String {
    // examples/ runs from the crate root; also work from any cwd by
    // walking up to the Cargo.toml
    let mut dir = env::current_dir().unwrap();
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
    for line in fs::read_to_string(fname)
        .map_err(|e| format!("{fname}: {e}"))?
        .lines()
    {
        if line.starts_with('#') || line.trim().is_empty() {
            continue;
        }
        let mut row = Vec::new();
        for w in line.split_whitespace() {
            row.push(
                w.parse::<f64>()
                    .map_err(|_| format!("{fname}: bad value `{w}`"))?,
            );
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
            fails
                .iter()
                .take(10)
                .cloned()
                .collect::<Vec<_>>()
                .join("\n  ")
        );
    }
    fails
}

/// `G^R`/`G^<` of a steady-state Green's function, component
/// `gf_{ret,les}_{time,freq}` (NESSi time step in `r.j`).
pub fn example_gf(g: &SteadyGreen, r: &KbrefRow, d: usize) -> Complex {
    let time = r.component.ends_with("time");
    let arr = if r.component.contains("ret") {
        if time { &g.ret.time } else { &g.ret.freq }
    } else if time {
        &g.les.time
    } else {
        &g.les.freq
    };
    arr[(r.j as usize) * d * d]
}

//
// the Holstein comparison helpers
//

/// Value of a kbref row in a result: `scalars` maps the component to
/// the vector indexed by the NESSi time step + 1, `rho(t, a, b)` gives
/// the density matrix element.
pub fn hol_value(
    row: &KbrefRow,
    scalars: &BTreeMap<String, Vec<f64>>,
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
    scalars: &BTreeMap<String, Vec<f64>>,
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
