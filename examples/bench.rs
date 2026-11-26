// Project : KadanoffBaym
// File    : examples/bench.rs
// Purpose : timing driver of the tutorial examples; writes a wall-time report outside the repository.
//
// Usage:
//   cargo run --release --example bench -- [--profile quick|release] [--runs N]
//

#[path = "../util/mod.rs"]
mod util;

use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::{Instant, SystemTime};

use std::time::UNIX_EPOCH;

use std::env::{args, current_exe, temp_dir};
use std::fs::{create_dir_all, read_to_string, remove_dir_all, write};
use std::process::id;

use util::{find_run, redirected_config, workspace_root};

/// The timed entries: (report name, example binary, model selector); the
/// runs are the shared table `TIMED_RUNS` of the utilities.
const ENTRIES: [(&str, &str, &str); 10] = [
    ("hubbard_2b", "hubbard", "chain_2b"),
    ("hubbard_gw", "hubbard", "chain_gw"),
    ("hubbard_tpp", "hubbard", "chain_tpp"),
    ("fkm_bethe_quench", "falicov_kimball", ""),
    ("holstein_bethe_migdal", "holstein", "bethe_migdal"),
    ("holstein_bethe_umig", "holstein", "bethe_umig"),
    ("holstein_nambu_migdal", "holstein", "bethe_nambu_migdal"),
    ("holstein_nambu_umig", "holstein", "bethe_nambu_umig"),
    ("holstein_impurity_migdal", "holstein", "impurity_migdal"),
    ("holstein_impurity_umig", "holstein", "impurity_umig"),
];

fn cpu_model() -> String {
    read_to_string("/proc/cpuinfo")
        .unwrap_or_default()
        .lines()
        .find(|l| l.starts_with("model name"))
        .and_then(|l| l.split(':').nth(1))
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|| "unknown".to_string())
}

fn rustc_version() -> String {
    Command::new("rustc")
        .arg("--version")
        .output()
        .ok()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_else(|| "unknown".to_string())
}

fn main() {
    let args: Vec<String> = args().skip(1).collect();
    let profile = args
        .iter()
        .find_map(|a| a.strip_prefix("--profile="))
        .map(|s| s.to_string())
        .unwrap_or_else(|| "quick".to_string());
    let runs: usize = args
        .iter()
        .find_map(|a| a.strip_prefix("--runs="))
        .and_then(|s| s.parse().ok())
        .unwrap_or(2);

    let root = workspace_root();
    let exedir = current_exe()
        .expect("exe")
        .parent()
        .expect("dir")
        .to_path_buf();
    let tmp = temp_dir().join(format!("kb-bench-{}", id()));

    // the entry arguments for a given grid profile: the quick profile is
    // the frozen default of the TOMLs; the release profile overrides
    let mut results: Vec<(String, f64, &str)> = Vec::new();
    let t_all = Instant::now();
    for (name, exe, sel) in ENTRIES {
        let run = find_run(exe, sel);
        let dir = tmp.join(name);
        create_dir_all(&dir).expect("temp dir");
        let redirected = redirected_config(&root.join(run.toml), &dir).expect("toml");
        let cfg = dir.join("case.toml");
        write(&cfg, redirected).expect("write cfg");
        let mut cmdline: Vec<String> = Vec::new();
        if !run.selector.is_empty() {
            cmdline.push(run.selector.to_string());
        }
        cmdline.push(cfg.display().to_string());
        for a in run.args {
            // the release grid profile replaces the quick default
            if profile == "release" && *a == "quick" {
                cmdline.push("release".to_string());
            } else {
                cmdline.push(a.to_string());
            }
        }
        // warm-up (untimed)
        let ok = Command::new(exedir.join(run.exe))
            .args(&cmdline)
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .map(|s| s.success())
            .unwrap_or(false);
        if !ok {
            results.push((name.to_string(), f64::NAN, "failed"));
            continue;
        }
        // timed runs; the last one counts
        let mut wall = f64::NAN;
        for _ in 0..runs {
            let t = Instant::now();
            let ok = Command::new(exedir.join(run.exe))
                .args(&cmdline)
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .status()
                .map(|s| s.success())
                .unwrap_or(false);
            wall = t.elapsed().as_secs_f64();
            if !ok {
                wall = f64::NAN;
                break;
            }
        }
        let status = if wall.is_finite() { "ok" } else { "failed" };
        println!("{name:<26} {wall:10.6} s  [{status}]");
        results.push((name.to_string(), wall, status));
    }
    let total = t_all.elapsed().as_secs_f64();
    let _ = remove_dir_all(&tmp);

    // the TOML report in the baseline format (kept outside the repo)
    let date = date_stamp();
    let outdir = PathBuf::from("/tmp/opencode");
    create_dir_all(&outdir).expect("create /tmp/opencode");
    let report = outdir.join(format!("bench-rust-{profile}-{date}.toml"));
    let mut out = String::new();
    out.push_str(&format!("cpu = {:?}\n", cpu_model()));
    out.push_str(&format!("rustc = {:?}\n", rustc_version()));
    out.push_str(&format!("profile = {profile:?}\n"));
    out.push_str("threads = 1\n");
    out.push_str(
        "protocol = \"warmup 1 + timed runs, last run counts (subprocess, stdout discarded)\"\n",
    );
    out.push_str(&format!("total_wall_s = {total}\n"));
    for (name, wall, status) in &results {
        out.push_str("\n[[entries]]\n");
        out.push_str(&format!("name = {name:?}\n"));
        out.push_str(&format!("status = {status:?}\n"));
        out.push_str(&format!("wall_s = {wall}\n"));
    }
    write(&report, out).expect("write report");
    println!("\nreport: {}", report.display());
}

/// Local date string (YYYYMMDD) without external crates.
fn date_stamp() -> String {
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock")
        .as_secs();
    let days = secs / 86400;
    // civil-from-days (Howard Hinnant's algorithm)
    let z = days as i64 + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    format!("{y:04}{m:02}{d:02}")
}
