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

use util::{get_f64, load_toml_config, read_ref, workspace_root};

struct Check {
    name: String,
    passed: bool,
    detail: String,
}

impl Check {
    fn ok(name: &str, detail: &str) -> Self {
        Check {
            name: name.to_string(),
            passed: true,
            detail: detail.to_string(),
        }
    }
    fn fail(name: &str, detail: &str) -> Self {
        Check {
            name: name.to_string(),
            passed: false,
            detail: detail.to_string(),
        }
    }
}

fn binary_dir() -> PathBuf {
    let exe = current_exe().expect("current exe");
    exe.parent().expect("exe dir").to_path_buf()
}

/// Read the reference rows as `(component, i, value)`.
fn reference_rows(path: &Path) -> Vec<(String, i64, f64)> {
    read_ref(path.to_string_lossy().as_ref())
        .expect("reference file")
        .rows
        .iter()
        .map(|r| (r.component.clone(), r.i, r.value.re))
        .collect()
}

/// Read a whitespace-separated numeric table (skipping `#` lines).
fn read_table(path: &Path) -> Vec<Vec<f64>> {
    let text = read_to_string(path).expect("output file");
    text.lines()
        .filter(|l| !l.starts_with('#') && !l.trim().is_empty())
        .map(|l| l.split_whitespace().map(|x| x.parse().unwrap()).collect())
        .collect()
}

/// Compare reference rows against a keyed output table with the given
/// column map; returns (checked, failed, worst).
fn compare_reference(
    refs: &[(String, i64, f64)],
    table: &[Vec<f64>],
    key_col: usize,
    cols: fn(&str) -> Option<usize>,
    h: f64,
    tol: f64,
) -> (usize, usize, f64) {
    let mut keyed = HashMap::new();
    for r in table {
        keyed.insert((r[key_col] * 1e8).round() as i64, r.clone());
    }
    let (mut n, mut bad, mut worst) = (0usize, 0usize, 0f64);
    for (comp, i, rv) in refs {
        let Some(col) = cols(&comp.replace(".re", "")) else {
            continue;
        };
        if *i < 0 {
            // the Matsubara branch rows live in the state, not the tables
            continue;
        }
        let Some(row) = keyed.get(&(((*i as f64) * h * 1e8).round() as i64)) else {
            bad += 1;
            continue;
        };
        n += 1;
        let e = (row[col] - rv).abs();
        worst = worst.max(e);
        if e > tol + tol * rv.abs() {
            bad += 1;
        }
    }
    (n, bad, worst)
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

/// Run one example binary with a config redirected into `outdir`.
fn run_case(
    name: &str,
    pre: &[&str],
    toml_rel: &str,
    outdir: &Path,
    extra: &[&str],
) -> Result<(), String> {
    let root = workspace_root();
    let src = root.join(toml_rel);
    let text = read_to_string(&src).map_err(|e| e.to_string())?;
    // redirect dir = "." (and any relative dir) into the temp directory
    let redirected = text.replace(
        "dir = \".\"",
        &format!("dir = {:?}", outdir.display().to_string()),
    );
    let cfg = outdir.join("case.toml");
    write(&cfg, redirected).map_err(|e| e.to_string())?;
    let exe = binary_dir().join(name);
    let out = Command::new(&exe)
        .args(pre)
        .arg(&cfg)
        .args(extra)
        .output()
        .map_err(|e| format!("{name}: {e}"))?;
    if !out.status.success() {
        return Err(format!(
            "{name} failed: {}",
            String::from_utf8_lossy(&out.stderr)
        ));
    }
    Ok(())
}

fn main() {
    let root = workspace_root();
    let data = root.join("data/tutor");
    let tmp = temp_dir().join(format!("kb-validate-{}", id()));
    create_dir_all(&tmp).expect("temp dir");
    let mut checks: Vec<Check> = Vec::new();

    // ---- build_chain (2B / GW / TPP) ---------------------------------
    for approx in ["2b", "gw", "tpp"] {
        let dir = tmp.join(format!("hc_{approx}"));
        create_dir_all(&dir).unwrap();
        match run_case(
            "hubbard",
            &[&format!("chain_{approx}")],
            "examples/hubbard/chain.toml",
            &dir,
            &["--profile", "quick"],
        ) {
            Ok(()) => {
                let hh = config_h(&root, "examples/hubbard/chain.toml");
                let refs = reference_rows(&data.join(format!("hubbard_{approx}_quick.dat")));
                let occ = read_table(&dir.join(format!("chain_{approx}_occupation.dat")));
                let en = read_table(&dir.join(format!("chain_{approx}_energy.dat")));
                let lookup = |t: &[Vec<f64>]| {
                    let mut m = HashMap::new();
                    for r in t {
                        m.insert((r[0] * 1e8).round() as i64, r.clone());
                    }
                    m
                };
                let (om, em) = (lookup(&occ), lookup(&en));
                let (mut n, mut bad, mut worst) = (0, 0, 0f64);
                for (comp, i, rv) in &refs {
                    let m = if comp == "n1" { &om } else { &em };
                    let col = if comp == "n1" {
                        1
                    } else {
                        1 + ["Ekin", "Epot", "Etot"]
                            .iter()
                            .position(|c| c == comp)
                            .unwrap()
                    };
                    let key = ((*i as f64) * hh * 1e8).round() as i64;
                    let Some(row) = m.get(&key) else {
                        bad += 1;
                        continue;
                    };
                    n += 1;
                    let e = (row[col] - rv).abs();
                    worst = worst.max(e);
                    bad += (e > 1e-10 + 1e-10 * rv.abs()) as usize;
                }
                checks.push(if bad == 0 {
                    Check::ok(
                        &format!("chain_{approx}"),
                        &format!("{n} rows, worst {worst:.2e}"),
                    )
                } else {
                    Check::fail(
                        &format!("chain_{approx}"),
                        &format!("{bad}/{n} rows fail, worst {worst:.2e}"),
                    )
                });
            }
            Err(e) => checks.push(Check::fail(&format!("chain_{approx}"), &e)),
        }
    }

    // ---- falicov_kimball ------------------------------------------------
    {
        let dir = tmp.join("fkm");
        create_dir_all(&dir).unwrap();
        match run_case(
            "falicov_kimball",
            &[],
            "examples/falicov_kimball/falicov_kimball.toml",
            &dir,
            &["--profile", "quick"],
        ) {
            Ok(()) => {
                let hf = config_h(&root, "examples/falicov_kimball/falicov_kimball.toml");
                let refs = reference_rows(&data.join("fkm_bethe_quench_quick.dat"));
                let tab = read_table(&dir.join("falicov_kimball_density.dat"));
                let cols = |c: &str| match c {
                    "den0" => Some(1),
                    "den1" => Some(2),
                    "total" => Some(3),
                    _ => None,
                };
                let (n, bad, worst) = compare_reference(&refs, &tab, 0, cols, hf, 1e-10);
                let c = if bad == 0 {
                    Check::ok(
                        "falicov_kimball",
                        &format!("{n} rows (real time), worst {worst:.2e}"),
                    )
                } else {
                    Check::fail(
                        "falicov_kimball",
                        &format!("{bad}/{n} rows fail, worst {worst:.2e}"),
                    )
                };
                checks.push(c);
            }
            Err(e) => checks.push(Check::fail("falicov_kimball", &e)),
        }
    }

    // ---- holstein_bethe (4 variants) -------------------------------------
    for (approx, refn, nambu) in [
        ("migdal", "holstein_bethe_migdal", false),
        ("umig", "holstein_bethe_umig", false),
        ("nambu_migdal", "holstein_nambu_migdal", true),
        ("nambu_umig", "holstein_nambu_umig", true),
    ] {
        let dir = tmp.join(format!("hb_{approx}"));
        create_dir_all(&dir).unwrap();
        let sel = format!("bethe_{approx}");
        let toml = format!("examples/holstein/bethe_{approx}.toml");
        match run_case("holstein", &[&sel], &toml, &dir, &["--profile", "quick"]) {
            Ok(()) => {
                let hb_h = config_h(&root, &toml);
                let tab = read_table(&dir.join(format!("bethe_{approx}_obs.dat")));
                if nambu {
                    // the reference carries the rho(a,b) orbital indices
                    let text = read_to_string(data.join(format!("{refn}_quick.dat"))).unwrap();
                    let mut keyed = HashMap::new();
                    for r in &tab {
                        keyed.insert((r[0] * 1e8).round() as i64, r.clone());
                    }
                    let (mut n, mut bad, mut worst) = (0usize, 0usize, 0f64);
                    for l in text
                        .lines()
                        .filter(|l| !l.starts_with('#') && !l.trim().is_empty())
                    {
                        let p: Vec<&str> = l.split_whitespace().collect();
                        let i: f64 = p[1].parse().unwrap();
                        // the non-rho rows carry -1 placeholders
                        let (a, b): (usize, usize) = (
                            p[3].parse::<i64>().unwrap().max(0) as usize,
                            p[4].parse::<i64>().unwrap().max(0) as usize,
                        );
                        let rv: f64 = p[5].parse().unwrap();
                        let col = match p[0] {
                            "rho" => Some(1 + 2 * (2 * a + b)),
                            "XX" => Some(9),
                            "Ekin" => Some(10),
                            "Enx_MF" => Some(11),
                            "Enx_corr" => Some(12),
                            "Eph" => Some(13),
                            "Etot" => Some(14),
                            _ => None,
                        };
                        let Some(col) = col else { continue };
                        let Some(row) = keyed.get(&((i * hb_h * 1e8).round() as i64)) else {
                            bad += 1;
                            continue;
                        };
                        n += 1;
                        let e = (row[col] - rv).abs();
                        worst = worst.max(e);
                        bad += (e > 1e-10 + 1e-10 * rv.abs()) as usize;
                    }
                    let c = if bad == 0 {
                        Check::ok(
                            &format!("holstein_bethe_{approx}"),
                            &format!("{n} rows, worst {worst:.2e}"),
                        )
                    } else {
                        Check::fail(
                            &format!("holstein_bethe_{approx}"),
                            &format!("{bad}/{n} rows fail, worst {worst:.2e}"),
                        )
                    };
                    checks.push(c);
                    continue;
                }
                let refs = reference_rows(&data.join(format!("{refn}_quick.dat")));
                let cols_n: fn(&str) -> Option<usize> = |c: &str| match c {
                    "rho" => Some(1),
                    "XX" => Some(9),
                    "Ekin" => Some(10),
                    "Enx_MF" => Some(11),
                    "Enx_corr" => Some(12),
                    "Eph" => Some(13),
                    "Etot" => Some(14),
                    _ => None,
                };
                let cols: fn(&str) -> Option<usize> = |c: &str| match c {
                    "rho" => Some(1),
                    "Xph" => Some(3),
                    "Pph" => Some(4),
                    "XX" => Some(5),
                    "Ekin" => Some(6),
                    "Exn_MF" => Some(7),
                    "Exn_corr" => Some(8),
                    "Eph_cl" => Some(9),
                    "Eph_corr" => Some(10),
                    "Etot" => Some(11),
                    _ => None,
                };
                let (f, w) = if nambu { (hb_h, cols_n) } else { (hb_h, cols) };
                let (n, bad, worst) = compare_reference(&refs, &tab, 0, w, f, 1e-10);
                let c = if bad == 0 {
                    Check::ok(
                        &format!("holstein_bethe_{approx}"),
                        &format!("{n} rows, worst {worst:.2e}"),
                    )
                } else {
                    Check::fail(
                        &format!("holstein_bethe_{approx}"),
                        &format!("{bad}/{n} rows fail, worst {worst:.2e}"),
                    )
                };
                checks.push(c);
            }
            Err(e) => checks.push(Check::fail(&format!("holstein_bethe_{approx}"), &e)),
        }
    }

    // ---- holstein_impurity (2 variants) ----------------------------------
    for approx in ["migdal", "umig"] {
        let dir = tmp.join(format!("hi_{approx}"));
        create_dir_all(&dir).unwrap();
        let sel = format!("impurity_{approx}");
        let toml = format!("examples/holstein/impurity_{approx}.toml");
        match run_case("holstein", &[&sel], &toml, &dir, &["--profile", "quick"]) {
            Ok(()) => {
                let refs =
                    reference_rows(&data.join(format!("holstein_impurity_{approx}_quick.dat")));
                let tab = read_table(&dir.join(format!("impurity_{approx}_obs.dat")));
                let cols: fn(&str) -> Option<usize> = |c: &str| match c {
                    "rho00" => Some(1),
                    "rho11" => Some(3),
                    "Xph" => Some(5),
                    "Pph" => Some(6),
                    "XX" => Some(7),
                    "Ekin" => Some(8),
                    "Exn_MF" => Some(9),
                    "Exn_corr" => Some(10),
                    "Eph_cl" => Some(11),
                    "Eph_corr" => Some(12),
                    "Etot" => Some(13),
                    _ => None,
                };
                let h = config_h(&root, &toml);
                let (n, bad, worst) = compare_reference(&refs, &tab, 0, cols, h, 1e-10);
                let c = if bad == 0 {
                    Check::ok(
                        &format!("holstein_impurity_{approx}"),
                        &format!("{n} rows, worst {worst:.2e}"),
                    )
                } else {
                    Check::fail(
                        &format!("holstein_impurity_{approx}"),
                        &format!("{bad}/{n} rows fail, worst {worst:.2e}"),
                    )
                };
                checks.push(c);
            }
            Err(e) => checks.push(Check::fail(&format!("holstein_impurity_{approx}"), &e)),
        }
    }

    // ---- steady (density / Eint from the observables file) ----------------
    {
        let dir = tmp.join("steady");
        create_dir_all(&dir).unwrap();
        match run_case(
            "hubbard",
            &["bethe_steady"],
            "examples/hubbard/bethe_steady.toml",
            &dir,
            &[],
        ) {
            Ok(()) => {
                let obs = read_to_string(dir.join("bethe_steady_observables.dat")).unwrap();
                let vals: Vec<f64> = obs
                    .lines()
                    .find(|l| !l.starts_with('#') && !l.trim().is_empty())
                    .unwrap()
                    .split_whitespace()
                    .map(|x| x.parse().unwrap())
                    .collect();
                let (dens, eint) = (vals[0], vals[1]);
                // the steady state of the (8 digits)
                let ok = (dens - 0.99993334).abs() < 5e-8 && (eint - (-0.1717343)).abs() < 5e-7;
                let c = if ok {
                    Check::ok(
                        "bethe_steady",
                        &format!("density {dens:.8}, Eint {eint:.8}"),
                    )
                } else {
                    Check::fail(
                        "bethe_steady",
                        &format!("density {dens:.8}, Eint {eint:.8}"),
                    )
                };
                checks.push(c);
            }
            Err(e) => checks.push(Check::fail("bethe_steady", &e)),
        }
    }

    // ---- trunc / steady_truncated: row-structural checks -----------------------
    for (name, sel, toml, out, nrows) in [
        (
            "bethe_trunc",
            "bethe_trunc",
            "examples/hubbard/bethe_trunc.toml",
            "bethe_trunc_density.dat",
            42usize,
        ),
        (
            "bethe_steady_truncated",
            "bethe_steady_trunc",
            "examples/hubbard/bethe_steady_truncated.toml",
            "bethe_steady_truncated_trunc_density.dat",
            11,
        ),
    ] {
        let dir = tmp.join(name);
        create_dir_all(&dir).unwrap();
        match run_case("hubbard", &[sel], toml, &dir, &[]) {
            Ok(()) => {
                let tab = read_table(&dir.join(out));
                let finite = tab.iter().all(|r| r.iter().all(|x| x.is_finite()));
                let ok = tab.len() == nrows && finite;
                let c = if ok {
                    Check::ok(name, &format!("{} rows, all finite", tab.len()))
                } else {
                    Check::fail(name, &format!("{} rows, finite={}", tab.len(), finite))
                };
                checks.push(c);
            }
            Err(e) => checks.push(Check::fail(name, &e)),
        }
    }

    // ---- gw (reference gw.dat) ----------------------------------------
    {
        let dir = tmp.join("gw");
        create_dir_all(&dir).unwrap();
        match run_case("gw", &[], "examples/gw/gw.toml", &dir, &[]) {
            Ok(()) => {
                let tab = read_table(&dir.join("gw_observables.dat"));
                // the columns: tstp t E A Ekin Eint Etot density current
                // the reference run values (bit-identical output, W7)
                let want_last = [-0.5921373673f64, 0.2303739311, 0.0000006866];
                let last = tab.last().unwrap();
                let ok = (last[4] - want_last[0]).abs() < 1e-9
                    && (last[5] - want_last[1]).abs() < 1e-9
                    && (last[8] - want_last[2]).abs() < 1e-9;
                let c = if ok {
                    Check::ok(
                        "gw",
                        &format!(
                            "Ekin {:.10}, Eint {:.10}, current {:.10}",
                            last[4], last[5], last[8]
                        ),
                    )
                } else {
                    Check::fail(
                        "gw",
                        &format!(
                            "Ekin {:.10}, Eint {:.10}, current {:.10}",
                            last[4], last[5], last[8]
                        ),
                    )
                };
                checks.push(c);
            }
            Err(e) => checks.push(Check::fail("gw", &e)),
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
