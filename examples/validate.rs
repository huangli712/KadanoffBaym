// Project : KadanoffBaym
// File    : examples/validate.rs
// Purpose : validation runner comparing example observables with the reference data.
//
// Usage:
//   cargo run --release --example validate -- [filter]
//

#[path = "../util/mod.rs"]
mod util;

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Command;

use std::env::{current_exe, temp_dir};
use std::fs::{create_dir_all, read_to_string, remove_dir_all, write};
use std::process::{exit, id};

use util::{ExampleRun, RefData, RefRow};

use util::{find_run, get_f64, load_toml_config, read_ref, redirected_config, workspace_root};

use util::prelude::*;

// ------------------------------------------------------------------
// constants
// ------------------------------------------------------------------

/// Relative tolerance of every comparison: `|got - want| <= TOL + TOL |want|`.
const TOL: f64 = 1e-10;

// ------------------------------------------------------------------
// types
// ------------------------------------------------------------------

/// The result of one check.
struct Check {
    name: String,
    passed: bool,
    detail: String,
}

/// The counted comparison of one example with its reference data.
#[derive(Default)]
struct Tally {
    rows: usize,
    failed: usize,
    worst: f64,
}

impl Tally {
    /// Account a real value written by the example.
    fn push(&mut self, got: f64, want: f64) {
        let e = (got - want).abs();
        self.record(e, want.abs());
    }

    /// Account a complex value, given by its real and imaginary parts.
    fn push_complex(&mut self, got: (f64, f64), want: &Complex) {
        let e = (got.0 - want.re).hypot(got.1 - want.im);
        self.record(e, want.abs());
    }

    /// Account a reference row without a counterpart in the output.
    fn missing(&mut self) {
        self.failed += 1;
    }

    fn record(&mut self, e: f64, scale: f64) {
        self.worst = self.worst.max(e);
        if e > TOL + TOL * scale {
            self.failed += 1;
        }
        self.rows += 1;
    }

    /// The check result of the usual data rows.
    fn check(self, name: &str) -> Check {
        self.labeled(name, "rows")
    }

    /// The check result; `label` names the compared rows.
    fn labeled(self, name: &str, label: &str) -> Check {
        let detail = if self.failed == 0 {
            format!("{} {label}, worst {:.2e}", self.rows, self.worst)
        } else {
            format!(
                "{}/{} rows fail, worst {:.2e}",
                self.failed, self.rows, self.worst
            )
        };
        Check {
            name: name.to_string(),
            passed: self.failed == 0,
            detail,
        }
    }
}

// ------------------------------------------------------------------
// the reference data and the example output
// ------------------------------------------------------------------

/// Read one reference data file of `data/tutor`.
fn reference(data: &Path, file: &str) -> RefData {
    read_ref(data.join(file).to_str().unwrap()).expect("reference file")
}

/// Read a whitespace-separated numeric table (skipping `#` lines).
fn read_table(path: &Path) -> Vec<Vec<f64>> {
    read_to_string(path)
        .expect("output file")
        .lines()
        .filter(|l| !l.starts_with('#') && !l.trim().is_empty())
        .map(|l| l.split_whitespace().map(|x| x.parse().unwrap()).collect())
        .collect()
}

/// The Green's function tables `prefix_G_time.dat` and `prefix_G_freq.dat`
/// written by a steady-state example.
fn green_tables(dir: &Path, prefix: &str) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    (
        read_table(&dir.join(format!("{prefix}_G_time.dat"))),
        read_table(&dir.join(format!("{prefix}_G_freq.dat"))),
    )
}

/// The value of a Green's function reference row in the tables of
/// `green_tables`, whose columns are x Re G^R Im G^R Re G^< Im G^<; the
/// index of a row is its `j` column.
fn green_pair(time: &[Vec<f64>], freq: &[Vec<f64>], r: &RefRow) -> Option<(f64, f64)> {
    let j = r.j as usize;
    match r.component.as_str() {
        "gf_ret_time" => time.get(j).map(|row| (row[1], row[2])),
        "gf_les_time" => time.get(j).map(|row| (row[3], row[4])),
        "gf_ret_freq" => freq.get(j).map(|row| (row[1], row[2])),
        "gf_les_freq" => freq.get(j).map(|row| (row[3], row[4])),
        _ => None,
    }
}

/// Index the rows of a table by their rounded first column.
fn key_rows(table: &[Vec<f64>]) -> HashMap<i64, &[f64]> {
    table
        .iter()
        .map(|r| ((r[0] * 1e8).round() as i64, r.as_slice()))
        .collect()
}

/// The key of the time step `i` in a table indexed by `key_rows`.
fn time_key(i: i64, h: f64) -> i64 {
    (i as f64 * h * 1e8).round() as i64
}

/// Read the selected Green's function components written by the examples as
/// `(component, i, j, a, b, re, im)`.
fn selected_green_rows(path: &Path) -> Vec<(String, i64, i64, i64, i64, f64, f64)> {
    read_to_string(path)
        .expect("output file")
        .lines()
        .filter(|l| !l.starts_with('#') && !l.trim().is_empty())
        .map(|l| {
            let p: Vec<&str> = l.split_whitespace().collect();
            (
                p[0].to_string(),
                p[1].parse().unwrap(),
                p[2].parse().unwrap(),
                p[3].parse().unwrap(),
                p[4].parse().unwrap(),
                p[5].parse().unwrap(),
                p[6].parse().unwrap(),
            )
        })
        .collect()
}

// ------------------------------------------------------------------
// the comparisons
// ------------------------------------------------------------------

/// Compare the real-valued rows of `refs`: `got` maps a reference row to the
/// value written by the example, or to `None` when the row has no counterpart
/// in the output (which counts as a failure).
fn compare_rows(
    refs: &RefData,
    skip: impl Fn(&RefRow) -> bool,
    got: impl Fn(&RefRow) -> Option<f64>,
) -> Tally {
    let mut t = Tally::default();
    for r in &refs.rows {
        if skip(r) {
            continue;
        }
        match got(r) {
            Some(value) => t.push(value, r.value.re),
            None => t.missing(),
        }
    }
    t
}

/// The complex-valued counterpart of `compare_rows`.
fn compare_complex(refs: &RefData, got: impl Fn(&RefRow) -> Option<(f64, f64)>) -> Tally {
    let mut t = Tally::default();
    for r in &refs.rows {
        match got(r) {
            Some(value) => t.push_complex(value, &r.value),
            None => t.missing(),
        }
    }
    t
}

/// Compare the rows of `refs` with a table keyed by the time in its first
/// column. The Matsubara rows (negative index) are skipped; a reference row
/// whose component has no column, or which is missing from the table, counts
/// as a failure.
fn compare_reference(
    refs: &RefData,
    table: &[Vec<f64>],
    h: f64,
    cols: impl Fn(&RefRow) -> Option<usize>,
) -> Tally {
    let keyed = key_rows(table);
    let mut t = Tally::default();
    for r in &refs.rows {
        if r.i < 0 {
            // the Matsubara branch rows live in the state, not the tables
            continue;
        }
        let Some(col) = cols(r) else {
            eprintln!(
                "validate: no column for the reference component {}",
                r.component
            );
            t.missing();
            continue;
        };
        let Some(row) = keyed.get(&time_key(r.i, h)) else {
            t.missing();
            continue;
        };
        t.push(row[col], r.value.re);
    }
    t
}

/// The column of a `*_obs.dat` table of the Bethe Holstein model.
fn bethe_columns(r: &RefRow) -> Option<usize> {
    match r.component.as_str() {
        "rho" => Some(1),
        "Xph" => Some(3),
        "Pph" => Some(4),
        "XX" => Some(5),
        "Ekin" => Some(6),
        "Enx_MF" => Some(7),
        "Enx_corr" => Some(8),
        "Eph_cl" => Some(9),
        "Eph_corr" => Some(10),
        "Etot" => Some(11),
        _ => None,
    }
}

/// The column of a `*_obs.dat` table of the Nambu Holstein model: a `rho`
/// row carries its orbital indices `a` and `b`, the other rows -1.
fn nambu_columns(r: &RefRow) -> Option<usize> {
    let (a, b) = (r.a.max(0) as usize, r.b.max(0) as usize);
    match r.component.as_str() {
        "rho" => Some(1 + 2 * (2 * a + b)),
        "XX" => Some(9),
        "Ekin" => Some(10),
        "Enx_MF" => Some(11),
        "Enx_corr" => Some(12),
        "Eph" => Some(13),
        "Etot" => Some(14),
        _ => None,
    }
}

/// The column of a `*_obs.dat` table of the impurity Holstein model: a
/// `rho` row carries the orbital indices `a` and `b`, the other rows -1.
fn impurity_columns(r: &RefRow) -> Option<usize> {
    match (r.component.as_str(), r.a, r.b) {
        ("rho", 0, 0) => Some(1),
        ("rho", 1, 1) => Some(3),
        ("Xph", _, _) => Some(5),
        ("Pph", _, _) => Some(6),
        ("XX", _, _) => Some(7),
        ("Ekin", _, _) => Some(8),
        ("Enx_MF", _, _) => Some(9),
        ("Enx_corr", _, _) => Some(10),
        ("Eph_cl", _, _) => Some(11),
        ("Eph_corr", _, _) => Some(12),
        ("Etot", _, _) => Some(13),
        _ => None,
    }
}

// ------------------------------------------------------------------
// running the examples
// ------------------------------------------------------------------

fn binary_dir() -> PathBuf {
    let exe = current_exe().expect("current exe");
    exe.parent().expect("exe dir").to_path_buf()
}

/// Read the time step `h` of an example TOML.
fn config_h(root: &Path, toml_rel: &str) -> f64 {
    let path = root.join(toml_rel);
    let p = load_toml_config(path.to_string_lossy().as_ref()).expect("toml");
    for sec in ["contour", "grid", ""] {
        if !(sec.is_empty() || p.contains_key(sec)) {
            continue;
        }
        if let Ok(v) = get_f64(&p, sec, "h") {
            return v;
        }
    }
    panic!("no h in {toml_rel}")
}

/// Request the selected Green's function components in an example config:
/// the Hubbard chain writes them only when asked, and the green reference
/// data is checked. Only a section header line `[output]` is extended, so
/// that the same words inside a comment are left alone.
fn request_green_components(text: String) -> String {
    if text.contains("save_green_components") {
        return text;
    }
    let mut out = String::new();
    for line in text.lines() {
        out.push_str(line);
        out.push('\n');
        if line == "[output]" {
            out.push_str("save_green_components = true\n");
        }
    }
    out
}

/// Run the example of `run` with its configuration redirected into
/// `outdir`.
fn run_case(run: &ExampleRun, outdir: &Path) -> Result<(), String> {
    let root = workspace_root();
    let redirected = redirected_config(&root.join(run.toml), outdir)?;
    let redirected = request_green_components(redirected);
    let cfg = outdir.join("case.toml");
    write(&cfg, redirected).map_err(|e| e.to_string())?;
    let mut cmd = Command::new(binary_dir().join(run.exe));
    if !run.selector.is_empty() {
        cmd.arg(run.selector);
    }
    let out = cmd
        .arg(&cfg)
        .args(run.args)
        .output()
        .map_err(|e| format!("{}: {e}", run.exe))?;
    if !out.status.success() {
        return Err(format!(
            "{} failed: {}",
            run.exe,
            String::from_utf8_lossy(&out.stderr)
        ));
    }
    Ok(())
}

/// Run the example `run` with the output redirected into `tmp/<name>`. A
/// failed run is reported at once and `None` is returned.
fn run_example(
    checks: &mut Vec<Check>,
    tmp: &Path,
    name: &str,
    run: &ExampleRun,
) -> Option<PathBuf> {
    let dir = tmp.join(name);
    create_dir_all(&dir).unwrap();
    match run_case(run, &dir) {
        Ok(()) => Some(dir),
        Err(e) => {
            checks.push(Check {
                name: name.to_string(),
                passed: false,
                detail: e,
            });
            None
        }
    }
}

fn main() {
    let root = workspace_root();
    let data = root.join("data/tutor");
    let tmp = temp_dir().join(format!("kb-validate-{}", id()));
    create_dir_all(&tmp).expect("temp dir");
    let mut checks: Vec<Check> = Vec::new();

    // ---- build_chain (2B / GW / TPP) -------------------------------------
    let toml = "examples/hubbard/chain.toml";
    let hh = config_h(&root, toml);
    for approx in ["2b", "gw", "tpp"] {
        let sel = format!("chain_{approx}");
        if let Some(dir) = run_example(&mut checks, &tmp, &sel, find_run("hubbard", &sel)) {
            let refs = reference(&data, &format!("hubbard_{approx}_quick.dat"));
            let occupation = read_table(&dir.join(format!("{sel}_occupation.dat")));
            let energy = read_table(&dir.join(format!("{sel}_energy.dat")));
            let occ = key_rows(&occupation);
            let en = key_rows(&energy);
            let energies = ["Ekin", "Epot", "Etot"];
            let mut t = Tally::default();
            for r in &refs.rows {
                let (table, col) = if r.component == "n1" {
                    (&occ, 1)
                } else {
                    (
                        &en,
                        1 + energies.iter().position(|c| *c == r.component).unwrap(),
                    )
                };
                let Some(row) = table.get(&time_key(r.i, hh)) else {
                    t.missing();
                    continue;
                };
                t.push(row[col], r.value.re);
            }
            // the selected Green's function components (G_11)
            let green = reference(&data, &format!("hubbard_{approx}_quick_green.dat"));
            let written = selected_green_rows(&dir.join(format!("{sel}_green.dat")));
            for r in &green.rows {
                let hit = written.iter().find(|o| {
                    o.0 == r.component && o.1 == r.i && o.2 == r.j && o.3 == r.a && o.4 == r.b
                });
                match hit {
                    Some(o) => t.push_complex((o.5, o.6), &r.value),
                    None => t.missing(),
                }
            }
            checks.push(t.check(&sel));
        }
    }

    // ---- falicov_kimball ------------------------------------------------
    {
        let toml = "examples/falicov_kimball/falicov_kimball.toml";
        if let Some(dir) = run_example(
            &mut checks,
            &tmp,
            "falicov_kimball",
            find_run("falicov_kimball", ""),
        ) {
            let refs = reference(&data, "fkm_bethe_quench_quick.dat");
            let tab = read_table(&dir.join("falicov_kimball_density.dat"));
            let cols = |r: &RefRow| match r.component.as_str() {
                "den0" => Some(1),
                "den1" => Some(2),
                "total" => Some(3),
                _ => None,
            };
            let t = compare_reference(&refs, &tab, config_h(&root, toml), cols);
            checks.push(t.labeled("falicov_kimball", "rows (real time)"));
        }
    }

    // ---- holstein_bethe (4 variants) -------------------------------------
    for (approx, refn, nambu) in [
        ("migdal", "holstein_bethe_migdal", false),
        ("umig", "holstein_bethe_umig", false),
        ("nambu_migdal", "holstein_nambu_migdal", true),
        ("nambu_umig", "holstein_nambu_umig", true),
    ] {
        let sel = format!("bethe_{approx}");
        let toml = format!("examples/holstein/{sel}.toml");
        let name = format!("holstein_bethe_{approx}");
        if let Some(dir) = run_example(&mut checks, &tmp, &name, find_run("holstein", &sel)) {
            let refs = reference(&data, &format!("{refn}_quick.dat"));
            let tab = read_table(&dir.join(format!("{sel}_obs.dat")));
            let cols: fn(&RefRow) -> Option<usize> =
                if nambu { nambu_columns } else { bethe_columns };
            let t = compare_reference(&refs, &tab, config_h(&root, &toml), cols);
            checks.push(t.check(&name));
        }
    }

    // ---- holstein_impurity (2 variants) ----------------------------------
    for approx in ["migdal", "umig"] {
        let sel = format!("impurity_{approx}");
        let toml = format!("examples/holstein/{sel}.toml");
        let name = format!("holstein_impurity_{approx}");
        if let Some(dir) = run_example(&mut checks, &tmp, &name, find_run("holstein", &sel)) {
            let refs = reference(&data, &format!("holstein_impurity_{approx}_quick.dat"));
            let tab = read_table(&dir.join(format!("{sel}_obs.dat")));
            let t = compare_reference(&refs, &tab, config_h(&root, &toml), impurity_columns);
            checks.push(t.check(&name));
        }
    }

    // ---- steady (density / Eint from the observables file) ----------------
    {
        if let Some(dir) = run_example(
            &mut checks,
            &tmp,
            "bethe_steady",
            find_run("hubbard", "bethe_steady"),
        ) {
            // the columns of the observables file: density Eint iterations
            let observables = read_table(&dir.join("bethe_steady_observables.dat"));
            let obs = &observables[0];
            // the columns: iteration error density
            let iterations = read_table(&dir.join("bethe_steady_iterations.dat"));
            let (time, freq) = green_tables(&dir, "bethe_steady");
            let refs = reference(&data, "examples_bethe.dat");
            // the index of every reference row is its `j` column
            let t = compare_complex(&refs, |r| match r.component.as_str() {
                "density" => Some((obs[0], 0.0)),
                "eint" => Some((obs[1], 0.0)),
                "iteration_error" => {
                    r.j.checked_sub(1)
                        .and_then(|i| iterations.get(i as usize))
                        .map(|row| (row[1], 0.0))
                }
                _ => green_pair(&time, &freq, r),
            });
            checks.push(t.check("bethe_steady"));
        }
    }

    // ---- bethe_prop (reference examples_bethe_prop.dat) --------------------
    {
        if let Some(dir) = run_example(
            &mut checks,
            &tmp,
            "bethe_prop",
            find_run("hubbard", "bethe_prop"),
        ) {
            // the columns of the density file: tstp t density
            let density = read_table(&dir.join("bethe_steady_propagation_density.dat"));
            // the columns of the error file: stage tstp iteration error
            let errors = read_table(&dir.join("bethe_steady_propagation_errors.dat"));
            let refs = reference(&data, "examples_bethe_prop.dat");
            // a density row is keyed by the KB time step `j` (0 = Matsubara), an
            // error row by the stage `a`, the time step `b`, and the iteration `j`
            let t = compare_rows(
                &refs,
                // the steady-state view of the last time slice is not written
                |r| matches!(r.component.as_str(), "gf_ret_time" | "gf_les_time"),
                |r| match r.component.as_str() {
                    "density" => density.iter().find(|x| x[0] as i64 == r.j).map(|x| x[2]),
                    "iteration_error" => errors
                        .iter()
                        .find(|x| x[0] as i64 == r.a && x[1] as i64 == r.b && x[2] as i64 == r.j)
                        .map(|x| x[3]),
                    _ => None,
                },
            );
            checks.push(t.check("bethe_prop"));
        }
    }

    // ---- bethe_trunc (reference trunc_bethe.dat) ---------------------------
    {
        if let Some(dir) = run_example(
            &mut checks,
            &tmp,
            "bethe_trunc",
            find_run("hubbard", "bethe_trunc"),
        ) {
            // the columns of the density file: tstp t density densk...
            let density = read_table(&dir.join("bethe_trunc_density.dat"));
            // the columns: tstp t ekin_trunc corrector_error
            let energy = read_table(&dir.join("bethe_trunc_energy.dat"));
            let refs = reference(&data, "trunc_bethe.dat");
            let tc: i64 = refs.metadata["tc"].parse().unwrap();
            // The density of the start-up steps 0..tc is the one of the window
            // rows at the handoff (output row tstp = step); the records of the
            // propagation are the NESSi steps (output row tstp = step + 1). The
            // leading rows of G and Sigma are not written by the example.
            let t = compare_rows(
                &refs,
                |r| r.j >= 0,
                |r| {
                    let (row, col, table) = match r.component.as_str() {
                        "density" => (if r.i <= tc { r.i } else { r.i + 1 }, 2, &density),
                        "ekin" => (r.i + 1, 2, &energy),
                        "error" => (r.i + 1, 3, &energy),
                        _ => return None,
                    };
                    table
                        .iter()
                        .find(|x| x[0] as i64 == row)
                        .map(|line| line[col])
                },
            );
            checks.push(t.check("bethe_trunc"));
        }
    }

    // ---- bethe_steady_truncated (reference steady_bethe_trunc.dat) ---------
    {
        if let Some(dir) = run_example(
            &mut checks,
            &tmp,
            "bethe_steady_truncated",
            find_run("hubbard", "bethe_steady_trunc"),
        ) {
            // the columns: rel tstp density corrector_error
            let density = read_table(&dir.join("bethe_steady_truncated_trunc_density.dat"));
            let refs = reference(&data, "steady_bethe_trunc.dat");
            // the reference index is the NESS relative step, the `rel` column;
            // the leading rows of G are not written by the example
            let t = compare_rows(
                &refs,
                |r| r.j >= 0,
                |r| {
                    density
                        .iter()
                        .find(|x| x[0] as i64 == r.i)
                        .map(|line| line[2])
                },
            );
            checks.push(t.check("bethe_steady_truncated"));
        }
    }

    // ---- gw (reference gw.dat) ----------------------------------------
    {
        if let Some(dir) = run_example(&mut checks, &tmp, "gw", find_run("gw", "")) {
            // the columns: tstp t E A Ekin Eint Etot density current
            let tab = read_table(&dir.join("gw_observables.dat"));
            let refs = reference(&data, "gw.dat");
            // the reference index is the NESSi time step; the output rows carry
            // the KB time step (0 = Matsubara, step + 1). The per-k-point
            // components are not written by the example.
            let t = compare_rows(
                &refs,
                |r| r.j >= 0,
                |r| {
                    let col = match r.component.as_str() {
                        "afield" => 3,
                        "ekin" => 4,
                        "eint" => 5,
                        "density" => 7,
                        "current" => 8,
                        _ => return None,
                    };
                    tab.get(r.i as usize + 1).map(|row| row[col])
                },
            );
            checks.push(t.check("gw"));
        }
    }

    // ---- anderson (reference examples_siam.dat) ----------------------------
    {
        if let Some(dir) = run_example(&mut checks, &tmp, "anderson", find_run("anderson", "")) {
            // the columns: density Eint current current_GSL - current_SLG
            let observables = read_table(&dir.join("anderson_observables.dat"));
            let obs = &observables[0];
            let (time, freq) = green_tables(&dir, "anderson");
            let refs = reference(&data, "examples_siam.dat");
            // the index of every reference row is its `j` column
            let t = compare_complex(&refs, |r| match (r.component.as_str(), r.j) {
                ("density", 0) => Some((obs[0], 0.0)),
                ("eint", 0) => Some((obs[1], 0.0)),
                ("current", 0) => Some((obs[2], 0.0)),
                ("current", 1) => Some((obs[2] + obs[3], 0.0)),
                _ => green_pair(&time, &freq, r),
            });
            checks.push(t.check("anderson"));
        }
    }

    // ---- summary -----------------------------------------------------------
    let mut failed = 0;
    println!();
    for c in &checks {
        let mark = if c.passed { "PASS" } else { "FAIL" };
        if !c.passed {
            failed += 1;
        }
        println!("[{mark}] {:<28} {}", c.name, c.detail);
    }
    let _ = remove_dir_all(&tmp);
    println!();
    if failed == 0 {
        println!("validate: {} checks passed", checks.len());
    } else {
        println!("validate: {failed}/{} checks FAILED", checks.len());
        exit(1);
    }
}
