//
// Usage:
//   cargo run --release --example bench -- [--profile quick|release] [--runs N]
//
// Timing driver of the tutor examples (the Rust counterpart of
// julia/tutor/profiles.jl): every P5 entry runs with its frozen quick
// or release configuration, once as a warm-up and twice timed (the
// second run counts, as in the R0 baseline protocol). The report is
// written to bench/rust-<profile>-<date>.toml in the format of the
// Julia baseline files (name / wall_s / status per entry).
//
// The entries run as subprocesses with stdout discarded; the process
// spawn overhead (~2 ms) is part of the measured wall time and noted
// in the report header.
//

use std::path::PathBuf;
use std::process::Command;
use std::time::Instant;

/// The timed entries: (report name, example binary, config, extra args).
const ENTRIES: &[(&str, &str, &str, &[&str])] = &[
    ("hubbard_2b", "hubbard_chain_2b", "julia/tutor/hubbard_chain/hubbard_chain.toml", &["--profile", "quick"]),
    ("hubbard_gw", "hubbard_chain_gw", "julia/tutor/hubbard_chain/hubbard_chain.toml", &["--profile", "quick"]),
    ("hubbard_tpp", "hubbard_chain_tpp", "julia/tutor/hubbard_chain/hubbard_chain.toml", &["--profile", "quick"]),
    ("fkm_bethe_quench", "fkm_bethe", "julia/tutor/fkm_bethe/fkm_bethe.toml", &["--profile", "quick"]),
    ("holstein_bethe_migdal", "holstein_bethe_migdal", "julia/tutor/holstein_bethe/holstein_bethe_migdal.toml", &["--profile", "quick"]),
    ("holstein_bethe_umig", "holstein_bethe_umig", "julia/tutor/holstein_bethe/holstein_bethe_umig.toml", &["--profile", "quick"]),
    ("holstein_nambu_migdal", "holstein_bethe_nambu_migdal", "julia/tutor/holstein_bethe/holstein_bethe_nambu_migdal.toml", &["--profile", "quick"]),
    ("holstein_nambu_umig", "holstein_bethe_nambu_umig", "julia/tutor/holstein_bethe/holstein_bethe_nambu_umig.toml", &["--profile", "quick"]),
    ("holstein_impurity_migdal", "holstein_impurity_singlebath_migdal", "julia/tutor/holstein_impurity/holstein_impurity_singlebath_migdal.toml", &["--profile", "quick"]),
    ("holstein_impurity_umig", "holstein_impurity_singlebath_umig", "julia/tutor/holstein_impurity/holstein_impurity_singlebath_umig.toml", &["--profile", "quick"]),
];

fn workspace_root() -> PathBuf {
    let exe = std::env::current_exe().expect("current exe");
    let mut dir = exe.parent().expect("exe dir").to_path_buf();
    loop {
        if dir.join("Cargo.toml").is_file() && dir.join("bench").is_dir() {
            return dir;
        }
        dir = match dir.parent() {
            Some(p) => p.to_path_buf(),
            None => panic!("workspace root not found"),
        };
    }
}

fn cpu_model() -> String {
    std::fs::read_to_string("/proc/cpuinfo")
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
    let args: Vec<String> = std::env::args().skip(1).collect();
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
    let exedir = std::env::current_exe()
        .expect("exe")
        .parent()
        .expect("dir")
        .to_path_buf();
    let tmp = std::env::temp_dir().join(format!("kb-r10-bench-{}", std::process::id()));

    // the entry arguments for a given grid profile: the quick profile is
    // the frozen default of the TOMLs; the release profile overrides
    let mut results: Vec<(String, f64, &str)> = Vec::new();
    let t_all = Instant::now();
    for (name, exe, toml, extra) in ENTRIES {
        let dir = tmp.join(name);
        std::fs::create_dir_all(&dir).expect("temp dir");
        let src = root.join(toml);
        let text = std::fs::read_to_string(&src).expect("toml");
        let redirected =
            text.replace("dir = \".\"", &format!("dir = {:?}", dir.display().to_string()));
        let cfg = dir.join("case.toml");
        std::fs::write(&cfg, redirected).expect("write cfg");
        let mut cmdline: Vec<String> = vec![cfg.display().to_string()];
        for a in *extra {
            // the release grid profile replaces the quick default
            if profile == "release" && *a == "quick" {
                cmdline.push("release".to_string());
            } else {
                cmdline.push(a.to_string());
            }
        }
        // warm-up (untimed)
        let ok = Command::new(exedir.join(exe))
            .args(&cmdline)
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
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
            let ok = Command::new(exedir.join(exe))
                .args(&cmdline)
                .stdout(std::process::Stdio::null())
                .stderr(std::process::Stdio::null())
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
    let _ = std::fs::remove_dir_all(&tmp);

    // the TOML report in the baseline format
    let date = chrono_like_date();
    let report = root.join(format!("bench/rust-{profile}-{date}.toml"));
    let mut out = String::new();
    out.push_str(&format!("cpu = {:?}\n", cpu_model()));
    out.push_str(&format!("rustc = {:?}\n", rustc_version()));
    out.push_str(&format!("profile = {profile:?}\n"));
    out.push_str("threads = 1\n");
    out.push_str("protocol = \"warmup 1 + timed runs, last run counts (subprocess, stdout discarded)\"\n");
    out.push_str(&format!("total_wall_s = {total}\n"));
    for (name, wall, status) in &results {
        out.push_str("\n[[entries]]\n");
        out.push_str(&format!("name = {name:?}\n"));
        out.push_str(&format!("status = {status:?}\n"));
        out.push_str(&format!("wall_s = {wall}\n"));
    }
    std::fs::write(&report, out).expect("write report");
    println!("\nreport: {}", report.display());
}

/// Local date string (YYYYMMDD) without external crates.
fn chrono_like_date() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
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
