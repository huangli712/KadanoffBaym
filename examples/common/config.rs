//
// Validated configuration of the tutor examples (the Rust port of
// julia/tutor/common/example_config.jl).
//

use super::toml::{parse_toml_file, TomlValue};
use kadanoffbaym::complex::Complex;
use kadanoffbaym::types::contour::Contour;
use kadanoffbaym::types::green::TimeFunction;
use kadanoffbaym::basics::MATSUBARA;
use kadanoffbaym::weights::Integrator;
use std::collections::BTreeMap;

pub const EXAMPLE_SECTIONS: [&str; 6] = ["contour", "solver", "model", "drive", "output", "profile"];
pub const EXAMPLE_CONTOUR_KEYS: [&str; 5] = ["nt", "ntau", "h", "beta", "solve_order"];
pub const EXAMPLE_SOLVER_KEYS: [&str; 5] = [
    "mats_max_iter",
    "mats_max_error",
    "bootstrap_max_iter",
    "bootstrap_max_error",
    "corrector_steps",
];
pub const EXAMPLE_SOLVER_OPTIONAL: [&str; 1] = ["output_every"];
pub const EXAMPLE_OUTPUT_KEYS: [&str; 3] = ["dir", "prefix", "save_green_components"];
pub const EXAMPLE_PROFILES: [&str; 2] = ["quick", "release"];
pub const DRIVE_KEYS: [&str; 5] = ["mode", "value", "path", "amplitude", "end"];

/// Validated configuration of an example.
#[derive(Clone, Debug)]
pub struct ExampleConfig {
    pub path: String,
    pub dir: String,
    pub profile: String,
    pub contour: BTreeMap<String, TomlValue>,
    pub solver: BTreeMap<String, TomlValue>,
    pub model: BTreeMap<String, TomlValue>,
    pub drive: BTreeMap<String, BTreeMap<String, TomlValue>>,
    pub output: BTreeMap<String, TomlValue>,
}

/// Contour parameters (the NESSi `nt`/`ntau` are my library counts).
#[derive(Clone, Copy, Debug)]
pub struct ContourParams {
    pub nt: i64,
    pub ntau: i64,
    pub h: f64,
    pub beta: f64,
    pub k: i64,
}

/// Solver parameters.
#[derive(Clone, Copy, Debug)]
pub struct SolverParams {
    pub mats_max_iter: i64,
    pub mats_max_error: f64,
    pub bootstrap_max_iter: i64,
    pub bootstrap_max_error: f64,
    pub corrector_steps: i64,
    pub output_every: i64,
}

fn check_value(v: &TomlValue) -> Result<(), String> {
    match v {
        TomlValue::Int(_) | TomlValue::Float(_) => {
            if let Some(x) = v.as_f64() {
                if !x.is_finite() {
                    return Err("must be finite".to_string());
                }
            }
            Ok(())
        }
        TomlValue::Array(items) => {
            for x in items {
                check_value(x)?;
            }
            Ok(())
        }
        TomlValue::Str(_) | TomlValue::Bool(_) => Ok(()),
        TomlValue::Table(_) => Err("unsupported type".to_string()),
    }
}

fn merge_table(
    dst: &mut BTreeMap<String, TomlValue>,
    src: &BTreeMap<String, TomlValue>,
    sec: &str,
    allowed: &[&str],
    path: &str,
) -> Result<(), String> {
    for (k, v) in src {
        if !allowed.contains(&k.as_str()) {
            return Err(format!("{path}: unknown key `{k}` in [{sec}]"));
        }
        check_value(v).map_err(|e| format!("{path}: [{sec}] {k} {e}"))?;
        dst.insert(k.clone(), v.clone());
    }
    Ok(())
}

fn merge_drive(
    dst: &mut BTreeMap<String, BTreeMap<String, TomlValue>>,
    src: &BTreeMap<String, TomlValue>,
    allowed: &[&str],
    path: &str,
) -> Result<(), String> {
    for (name, t) in src {
        if !allowed.contains(&name.as_str()) {
            return Err(format!("{path}: unknown drive `{name}`"));
        }
        let table = t
            .as_table()
            .ok_or_else(|| format!("{path}: [drive.{name}] must be a table"))?;
        let mut d = BTreeMap::new();
        merge_drive_table(&mut d, table, path, name)?;
        dst.insert(name.clone(), d);
    }
    Ok(())
}

fn merge_drive_table(
    d: &mut BTreeMap<String, TomlValue>,
    t: &BTreeMap<String, TomlValue>,
    path: &str,
    name: &str,
) -> Result<(), String> {
    for (k, v) in t {
        if !DRIVE_KEYS.contains(&k.as_str()) {
            return Err(format!("{path}: unknown key `{k}` in [drive.{name}]"));
        }
        check_value(v).map_err(|e| format!("{path}: [drive.{name}] {k} {e}"))?;
        d.insert(k.clone(), v.clone());
    }
    let mode = d.get("mode").and_then(|v| v.as_str()).unwrap_or("");
    match mode {
        "constant" => {
            let ok = d.get("value").map(|v| v.is_real()).unwrap_or(false)
                && !d.contains_key("path");
            if !ok {
                return Err(format!(
                    "{path}: [drive.{name}] constant mode needs a numeric `value`"
                ));
            }
        }
        "file" => {
            let ok = d.get("path").map(|v| v.as_str().is_some()).unwrap_or(false)
                && !d.contains_key("value");
            if !ok {
                return Err(format!("{path}: [drive.{name}] file mode needs a `path`"));
            }
        }
        "sin2" => {
            let ok = d.get("amplitude").map(|v| v.is_real()).unwrap_or(false)
                && d.get("end").map(|v| v.as_f64().map(|e| e > 0.0).unwrap_or(false)).unwrap_or(false);
            if !ok {
                return Err(format!(
                    "{path}: [drive.{name}] sin2 mode needs a numeric `amplitude` and a positive `end`"
                ));
            }
        }
        _ => {
            return Err(format!(
                "{path}: [drive.{name}] mode must be constant, file, or sin2"
            ));
        }
    }
    Ok(())
}

fn get_table<'a>(
    root: &'a BTreeMap<String, TomlValue>,
    key: &str,
) -> Option<&'a BTreeMap<String, TomlValue>> {
    root.get(key).and_then(|v| v.as_table())
}

/// Read and validate the TOML file and merge the profile overrides (the
/// port of `load_example_config`).
#[allow(clippy::too_many_arguments)]
pub fn load_example_config(
    path: &str,
    profile: &str,
    model_required: &[&str],
    model_optional: &[&str],
    drives: &[&str],
    drives_required: &[&str],
) -> Result<ExampleConfig, String> {
    if !std::path::Path::new(path).is_file() {
        return Err(format!("{path}: no such file"));
    }
    if !EXAMPLE_PROFILES.contains(&profile) {
        return Err(format!("profile must be quick or release, got {profile}"));
    }
    let raw = parse_toml_file(path).map_err(|e| e.to_string())?;
    for k in raw.keys() {
        if !EXAMPLE_SECTIONS.contains(&k.as_str()) {
            return Err(format!("{path}: unknown section [{k}]"));
        }
    }
    let mut model_keys: Vec<&str> = model_required.to_vec();
    model_keys.extend_from_slice(model_optional);
    let mut contour = BTreeMap::new();
    let mut solver = BTreeMap::new();
    let mut model = BTreeMap::new();
    let mut output = BTreeMap::new();
    let mut drive: BTreeMap<String, BTreeMap<String, TomlValue>> = BTreeMap::new();
    // apply one level of tables (base or profile)
    let apply = |t: &BTreeMap<String, TomlValue>,
                     contour: &mut BTreeMap<String, TomlValue>,
                     solver: &mut BTreeMap<String, TomlValue>,
                     model: &mut BTreeMap<String, TomlValue>,
                     output: &mut BTreeMap<String, TomlValue>,
                     drive: &mut BTreeMap<String, BTreeMap<String, TomlValue>>|
     -> Result<(), String> {
        for (sec, v) in t {
            let table = v
                .as_table()
                .ok_or_else(|| format!("{path}: [{sec}] must be a table"))?;
            match sec.as_str() {
                "drive" => merge_drive(drive, table, drives, path)?,
                "contour" => merge_table(contour, table, sec, &EXAMPLE_CONTOUR_KEYS, path)?,
                "solver" => {
                    let mut allowed = EXAMPLE_SOLVER_KEYS.to_vec();
                    allowed.extend_from_slice(&EXAMPLE_SOLVER_OPTIONAL);
                    merge_table(solver, table, sec, &allowed, path)?
                }
                "model" => merge_table(model, table, sec, &model_keys, path)?,
                "output" => merge_table(output, table, sec, &EXAMPLE_OUTPUT_KEYS, path)?,
                _ => return Err(format!("{path}: unknown section [{sec}] in the profile")),
            }
        }
        Ok(())
    };
    // the base tables (everything but [profile])
    let base: BTreeMap<String, TomlValue> = raw
        .iter()
        .filter(|(k, _)| k.as_str() != "profile")
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();
    apply(&base, &mut contour, &mut solver, &mut model, &mut output, &mut drive)?;
    // both profiles must be present and validate
    let prof = get_table(&raw, "profile");
    let prof = match prof {
        Some(t) => t,
        None => return Err(format!("{path}: missing [profile]")),
    };
    for p in prof.keys() {
        if !EXAMPLE_PROFILES.contains(&p.as_str()) {
            return Err(format!("{path}: unknown profile `{p}`"));
        }
    }
    for p in EXAMPLE_PROFILES {
        let t = prof
            .get(p)
            .and_then(|v| v.as_table())
            .ok_or_else(|| format!("{path}: missing [profile.{p}]"))?;
        let mut scratch = (
            BTreeMap::new(),
            BTreeMap::new(),
            BTreeMap::new(),
            BTreeMap::new(),
            BTreeMap::new(),
        );
        apply(
            t,
            &mut scratch.0,
            &mut scratch.1,
            &mut scratch.2,
            &mut scratch.3,
            &mut scratch.4,
        )?;
    }
    // the selected profile's overrides
    let selected = prof
        .get(profile)
        .and_then(|v| v.as_table())
        .ok_or_else(|| format!("{path}: missing [profile.{profile}]"))?;
    apply(selected, &mut contour, &mut solver, &mut model, &mut output, &mut drive)?;
    // required keys
    for (sec, d, keys) in [
        ("contour", &contour, EXAMPLE_CONTOUR_KEYS.as_slice()),
        ("solver", &solver, EXAMPLE_SOLVER_KEYS.as_slice()),
        ("model", &model, model_required),
        ("output", &output, ["dir", "prefix"].as_slice()),
    ] {
        for k in keys {
            if !d.contains_key(*k) {
                return Err(format!("{path}: missing key `{k}` in [{sec}]"));
            }
        }
    }
    for k in ["nt", "ntau", "solve_order"] {
        if contour.get(k).and_then(|v| v.as_i64()).is_none() {
            return Err(format!("{path}: [contour] {k} must be an integer"));
        }
    }
    for k in ["mats_max_iter", "bootstrap_max_iter", "corrector_steps"] {
        if solver.get(k).and_then(|v| v.as_i64()).is_none() {
            return Err(format!("{path}: [solver] {k} must be an integer"));
        }
    }
    solver
        .entry("output_every".to_string())
        .or_insert(TomlValue::Int(1));
    output
        .entry("save_green_components".to_string())
        .or_insert(TomlValue::Bool(false));
    if output
        .get("save_green_components")
        .and_then(|v| v.as_bool())
        .is_none()
    {
        return Err(format!(
            "{path}: [output] save_green_components must be a boolean"
        ));
    }
    let p = output
        .get("prefix")
        .and_then(|v| v.as_str())
        .ok_or_else(|| format!("{path}: [output] prefix must be a string"))?;
    if p.is_empty() || p.ends_with(".h5") {
        return Err(format!("{path}: [output] bad prefix `{p}`"));
    }
    for d in drives_required {
        if !drive.contains_key(*d) {
            return Err(format!("{path}: missing [drive.{d}]"));
        }
    }
    let dir = std::path::Path::new(path)
        .parent()
        .map(|p| p.to_string_lossy().to_string())
        .unwrap_or_default();
    let dir = if dir.is_empty() { ".".to_string() } else { dir };
    let cfg = ExampleConfig {
        path: std::fs::canonicalize(path)
            .map(|p| p.to_string_lossy().to_string())
            .unwrap_or_else(|_| path.to_string()),
        dir,
        profile: profile.to_string(),
        contour,
        solver,
        model,
        drive,
        output,
    };
    contour_params(&cfg)?;
    Ok(cfg)
}

/// Contour parameters (validated).
pub fn contour_params(cfg: &ExampleConfig) -> Result<ContourParams, String> {
    let nt = cfg.contour["nt"].as_i64().unwrap();
    let ntau = cfg.contour["ntau"].as_i64().unwrap();
    let k = cfg.contour["solve_order"].as_i64().unwrap();
    let h = cfg.contour["h"].as_f64().unwrap();
    let beta = cfg.contour["beta"].as_f64().unwrap();
    if !(1..=5).contains(&k) {
        return Err(format!(
            "{}: [contour] solve_order must be in 1:5, got {k}",
            cfg.path
        ));
    }
    if nt < k {
        return Err(format!("{}: [contour] nt must be >= solve_order", cfg.path));
    }
    if ntau < 2 {
        return Err(format!("{}: [contour] ntau must be >= 2", cfg.path));
    }
    if h <= 0.0 || beta <= 0.0 {
        return Err(format!(
            "{}: [contour] h and beta must be positive",
            cfg.path
        ));
    }
    Ok(ContourParams { nt, ntau, h, beta, k })
}

/// The contour of the configuration.
pub fn contour_of(cfg: &ExampleConfig) -> Result<Contour, String> {
    let p = contour_params(cfg)?;
    Ok(Contour::new(p.nt, p.ntau, 1, 1, p.nt as f64 * p.h, p.beta))
}

/// The integrator of the configuration.
pub fn integrator_of(cfg: &ExampleConfig) -> Result<Integrator, String> {
    Ok(Integrator::new(contour_params(cfg)?.k))
}

/// Solver parameters (validated).
pub fn solver_params(cfg: &ExampleConfig) -> Result<SolverParams, String> {
    let s = &cfg.solver;
    let v = SolverParams {
        mats_max_iter: s["mats_max_iter"].as_i64().unwrap(),
        mats_max_error: s["mats_max_error"].as_f64().unwrap(),
        bootstrap_max_iter: s["bootstrap_max_iter"].as_i64().unwrap(),
        bootstrap_max_error: s["bootstrap_max_error"].as_f64().unwrap(),
        corrector_steps: s["corrector_steps"].as_i64().unwrap(),
        output_every: s["output_every"].as_i64().unwrap(),
    };
    if !(v.mats_max_iter >= 0
        && v.bootstrap_max_iter >= 0
        && v.corrector_steps >= 1
        && v.mats_max_error > 0.0
        && v.bootstrap_max_error > 0.0)
    {
        return Err(format!("{}: [solver] invalid values", cfg.path));
    }
    Ok(v)
}

/// Fill the 1x1 single-time function with the drive `[drive.<name>]`
/// (the port of `load_tvector!`).
pub fn load_tvector(
    cfg: &ExampleConfig,
    name: &str,
) -> Result<TimeFunction, String> {
    let d = cfg
        .drive
        .get(name)
        .ok_or_else(|| format!("{}: missing [drive.{name}]", cfg.path))?;
    let cp = contour_params(cfg)?;
    let ntime = cp.nt + 1; // the KB time points
    let mut f = TimeFunction::new(cp.nt as usize, 1, 1, kadanoffbaym::complex::Complex::ZERO);
    let mode = d["mode"].as_str().unwrap();
    let v: Vec<f64> = if mode == "constant" {
        vec![d["value"].as_f64().unwrap(); (ntime + 1) as usize]
    } else if mode == "sin2" {
        // the sin^2 pulse of the NESSi Holstein demos, evaluated at
        // t = 0, h, .., (ntime - 1) h; the Matsubara point is zero; the
        // multiplication order follows the Python reference
        let a = d["amplitude"].as_f64().unwrap();
        let tend = d["end"].as_f64().unwrap();
        let mut v = vec![0.0f64; (ntime + 1) as usize];
        // v[i] (i >= 1) is the real time t = (i - 1) h (KB step i)
        for i in 1..=(ntime as usize) {
            let t = (i as i64 - 1) as f64 * cp.h;
            v[i] = if t > 0.0 && t < tend {
                a * (t * std::f64::consts::PI / tend).sin() * (t * std::f64::consts::PI / tend).sin()
            } else {
                0.0
            };
        }
        v
    } else {
        // file: nt + 2 values in the NESSi step order -1, 0, .., nt
        let p = d["path"].as_str().unwrap();
        let full = if std::path::Path::new(p).is_absolute() {
            p.to_string()
        } else {
            format!("{}/{}", cfg.dir, p)
        };
        if !std::path::Path::new(&full).is_file() {
            return Err(format!("{}: [drive.{name}] no such file {full}", cfg.path));
        }
        let mut v = Vec::new();
        for line in std::fs::read_to_string(&full)
            .map_err(|e| format!("{}: {full}: {e}", cfg.path))?
            .lines()
        {
            let s = line.trim();
            if s.is_empty() || s.starts_with('#') {
                continue;
            }
            for w in s.split_whitespace() {
                let x: f64 = w
                    .parse()
                    .map_err(|_| format!("{}: [drive.{name}] bad value `{w}` in {full}", cfg.path))?;
                if !x.is_finite() {
                    return Err(format!("{}: [drive.{name}] bad value `{w}` in {full}", cfg.path));
                }
                v.push(x);
            }
        }
        if v.len() != (ntime + 1) as usize {
            return Err(format!(
                "{}: [drive.{name}] {full} must hold nt + 2 = {} values, got {}",
                cfg.path,
                ntime + 1,
                v.len()
            ));
        }
        v
    };
    // the NESSi step -1 is the Matsubara value, the steps 0..nt follow
    f.block_mut(MATSUBARA)[0] = Complex::new(v[0], 0.0);
    for i in 0..=cp.nt {
        f.block_mut(i)[0] = Complex::new(v[(i + 1) as usize], 0.0);
    }
    Ok(f)
}

/// Parsed command line of an example entry.
#[derive(Clone, Debug)]
pub struct ExampleArgs {
    pub cfgfile: String,
    pub profile: String,
    pub validate_only: bool,
}

/// Parse `<config.toml> [--profile quick|release] [--validate-only]`.
pub fn parse_example_args(args: &[String]) -> Result<ExampleArgs, String> {
    if args.is_empty() {
        return Err(
            "usage: <config.toml> [--profile quick|release] [--validate-only]".to_string(),
        );
    }
    let cfgfile = args[0].clone();
    let mut profile = "quick".to_string();
    let mut validate = false;
    let mut i = 1;
    while i < args.len() {
        let a = &args[i];
        if a == "--profile" {
            if i + 1 >= args.len() {
                return Err("--profile needs a value".to_string());
            }
            profile = args[i + 1].clone();
            i += 2;
        } else if let Some(v) = a.strip_prefix("--profile=") {
            profile = v.to_string();
            i += 1;
        } else if a == "--validate-only" {
            validate = true;
            i += 1;
        } else {
            return Err(format!("unknown argument `{a}`"));
        }
    }
    Ok(ExampleArgs {
        cfgfile,
        profile,
        validate_only: validate,
    })
}

/// Format the command line arguments of the process for `parse_example_args`.
pub fn args_from_env() -> Vec<String> {
    std::env::args().skip(1).collect()
}
