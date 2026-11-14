// Project : KadanoffBaym
// File    : src/kbref.rs
// Purpose : kbref-v1 plain-text reference format (reader/writer)
//
// Port of julia/tutor/common/kbref.jl (reader, strict schema validation)
// and write_kbref (schema round-trip). Index translation (NESSi is
// 0-based): mat[q] -> G.mat[q], ret[i,j] -> G.ret[i,j] with 0 <= j <= i
// <= nt, tv[i,q] -> G.lmix[i,q], les[i,j] -> G.less[i,j] with
// 0 <= i <= j <= nt.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::fs;

use crate::complex::Complex;

/// Format version.
pub(crate) const KBREF_VERSION: &str = "1";
/// Column specification line.
pub(crate) const KBREF_COLUMNS: &str = "component i j a b re im";

/// All known data kinds.
pub(crate) const KBREF_KINDS: [&str; 15] = [
    "weights",
    "herm_matrix",
    "scalar",
    "function",
    "element",
    "labels",
    "iteration",
    "pseudo",
    "phase7_fourier",
    "phase7_ness",
    "phase7_example",
    "moving",
    "series",
    "observable",
    "selected_green",
];

/// P2 data kinds (require `generator` and `case_set` metadata).
const KBREF_P2_KINDS: [&str; 5] = ["scalar", "function", "element", "labels", "iteration"];

/// P7 data kinds.
const KBREF_P7_KINDS: [&str; 3] = ["phase7_fourier", "phase7_ness", "phase7_example"];

/// P7 `phase7_example` cases.
const KBREF_P7_CASES: [&str; 3] = ["siam", "bethe", "bethe_prop"];

/// Metadata of the P8 `moving` kind.
const KBREF_MOVING_META: [&str; 8] =
    ["generator", "case_set", "case_id", "source", "tc", "size", "sign", "tstp"];

/// Generators allowed for `moving`.
const KBREF_MOVING_GENERATORS: [&str; 2] = ["p8_moving_ref", "p8_moving_solver_ref"];

/// Metadata of the P8 `series` kind.
const KBREF_SERIES_META: [&str; 7] =
    ["generator", "case_set", "components", "nsteps", "nk", "size", "source"];

/// Metadata of the P6 `pseudo` kind.
const KBREF_PSEUDO_META: [&str; 8] =
    ["generator", "case", "sig", "nt", "ntau", "size1", "size2", "k"];

/// CG iteration scalars.
pub const KBREF_CG_SCALARS: [&str; 5] = ["rsold", "c2", "alpha", "rsnew", "beta_cg"];

/// Metadata of the P5 kinds.
pub const KBREF_P5_META: [&str; 15] = [
    "generator",
    "case_set",
    "model",
    "approximation",
    "profile",
    "nt",
    "ntau",
    "h",
    "beta",
    "solve_order",
    "mats_max_iter",
    "bootstrap_max_iter",
    "corrector_steps",
    "size",
    "params",
];

/// Holstein scalar observables (class 0).
const KBREF_P5_HOLSTEIN_SCALARS: [&str; 9] = [
    "Xph", "Pph", "XX", "Ekin", "Enx_MF", "Enx_corr", "Eph_cl", "Eph_corr", "Etot",
];

/// Allowed components per data kind (None for `series` and `observable`,
/// which use per-case sets).
pub(crate) fn kbref_components(kind: &str) -> Option<&'static [&'static str]> {
    match kind {
        "weights" => Some(&["greg", "bd", "pint", "pdiff", "rcorr"]),
        "herm_matrix" => Some(&["mat", "ret", "tv", "les"]),
        "scalar" => Some(&["distance_ret", "distance_tv", "distance_les", "distance_total", "response"]),
        "function" => Some(&["extrapolate", "interpolate"]),
        "element" => Some(&["les_timediag", "density_matrix", "startup_less_adv", "startup_lmix_rmix"]),
        "labels" => Some(&["order", "alpha"]),
        "iteration" => Some(&["g", "rsold", "c2", "alpha", "rsnew", "beta_cg"]),
        "pseudo" => Some(&["mat", "gtr", "tv", "les", "density", "les_start_source"]),
        "phase7_fourier" => Some(&[
            "fft_input",
            "fft_forward",
            "fft_backward",
            "dftcorr",
            "adaptive",
            "matsubara_dft_result",
            "matsubara_ft_result",
            "vie2_pcf20",
            "vie2_pcf10",
        ]),
        "phase7_ness" => Some(&[
            "ret_time",
            "ret_freq",
            "les_time",
            "les_freq",
            "bath_ret",
            "bath_les",
            "density",
            "convolution_density",
            "bubble1",
            "bubble2",
            "resample",
            "force_equilibrium",
            "cntr_conversion",
        ]),
        "phase7_example" => Some(&[
            "gf_ret_time",
            "gf_ret_freq",
            "gf_les_time",
            "gf_les_freq",
            "density",
            "current",
            "eint",
            "iteration_error",
        ]),
        "moving" => Some(&["ret", "less"]),
        "selected_green" => Some(&["mat", "ret", "tv", "les"]),
        _ => None,
    }
}

/// P5 `observable`: index class (0 real scalar, 2 density matrix) of a
/// component for a model.
pub fn observable_class(model: &str, comp: &str) -> Option<u8> {
    match model {
        "hubbard" => match comp {
            "n1" | "Ekin" | "Epot" | "Etot" => Some(0),
            _ => None,
        },
        "fkm" => match comp {
            "den0" | "den1" | "total" => Some(0),
            _ => None,
        },
        "holstein_bethe" | "holstein_impurity" => {
            if comp == "rho" {
                Some(2)
            } else if KBREF_P5_HOLSTEIN_SCALARS.contains(&comp)
                || matches!(comp, "XX" | "Ekin" | "Enx_MF" | "Enx_corr" | "Eph" | "Etot")
            {
                Some(0)
            } else {
                None
            }
        }
        "holstein_nambu" => {
            if comp == "rho" {
                Some(2)
            } else if matches!(comp, "XX" | "Ekin" | "Enx_MF" | "Enx_corr" | "Eph" | "Etot") {
                Some(0)
            } else {
                None
            }
        }
        _ => None,
    }
}

/// Approximations allowed for a P5 model.
pub fn p5_approximations(model: &str) -> Option<&'static [&'static str]> {
    match model {
        "hubbard" => Some(&["2B", "GW", "TPP"]),
        "fkm" => Some(&["none"]),
        "holstein_bethe" | "holstein_nambu" | "holstein_impurity" => Some(&["Migdal", "uMig"]),
        _ => None,
    }
}

/// P8 `series`: index class of a component for a case_set.
pub fn series_class(case_set: &str, comp: &str) -> Option<u8> {
    match case_set {
        "moving_solver" => match comp {
            "kinetic" => Some(0),
            "dyson_ret" | "dyson_less" | "dyson0_ret" | "dyson0_less" | "bubble1_ret"
            | "bubble1_less" | "bubble2_ret" | "bubble2_less" | "vie2_ret" | "vie2_less" => Some(2),
            _ => None,
        },
        "trunc_bethe" => match comp {
            "density" | "error" | "ekin" => Some(0),
            "g_ret" | "g_less" | "sigma_ret" | "sigma_less" => Some(2),
            _ => None,
        },
        "ness2_bethe_trunc" => match comp {
            "density" => Some(0),
            "g_ret" | "g_less" => Some(2),
            _ => None,
        },
        "gw" => match comp {
            "afield" | "density" | "current_nessi" | "ekin" | "eint" => Some(0),
            "rho" | "sigma_hf" | "g_ret" | "g_less" | "w_ret" | "w_less" | "sigma_gw_ret"
            | "sigma_gw_less" | "p_ret" | "p_less" => Some(2),
            _ => None,
        },
        _ => None,
    }
}

/// The generator that owns a `series` case_set.
pub fn series_generator(case_set: &str) -> Option<&'static str> {
    match case_set {
        "moving_solver" => Some("p8_moving_solver_ref"),
        "trunc_bethe" => Some("p8_trunc_bethe_ref"),
        "ness2_bethe_trunc" => Some("p8_ness2_bethe_trunc_ref"),
        "gw" => Some("p8_gw_ref"),
        _ => None,
    }
}

/// The four fixed keys of `selected_green` (hubbard only).
pub const KBREF_SELECTED_GREEN_KEYS: [(&str, i64, i64, i64, i64); 4] = [
    ("mat", 0, -1, 0, 0),
    ("ret", 3, 1, 0, 0),
    ("les", 1, 3, 0, 0),
    ("tv", 3, 2, 0, 0),
];

/// One data line of a kbref-v1 file.
#[derive(Clone, Debug, PartialEq)]
pub struct KbrefRow {
    pub component: String,
    pub i: i64,
    pub j: i64,
    pub a: i64,
    pub b: i64,
    pub value: Complex,
}

/// Content of a kbref-v1 file.
#[derive(Clone, Debug)]
pub struct KbrefData {
    pub path: String,
    pub metadata: BTreeMap<String, String>,
    pub rows: Vec<KbrefRow>,
}

/// Parse/validation error (`kbref: <path>:<line>: <msg>`).
#[derive(Clone, Debug, PartialEq)]
pub struct KbrefError {
    pub path: String,
    pub line: usize,
    pub msg: String,
}

impl fmt::Display for KbrefError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "kbref: {}:{}: {}", self.path, self.line, self.msg)
    }
}

impl std::error::Error for KbrefError {}

/// C-style `%.17e` formatting (two-digit zero-padded exponent with sign,
/// e.g. `4.16666666666666685e-01`). Rust's `{:e}` formatting is exact
/// and correctly rounded like C's printf; only the exponent shape is
/// adapted (`e0` -> `e+00`, `e-1` -> `e-01`).
pub fn fmt_c_e17(x: f64) -> String {
    if !x.is_finite() {
        // C printf style; Rust's f64 parser accepts these tokens back
        if x.is_nan() {
            return "nan".to_string();
        }
        return if x < 0.0 { "-inf" } else { "inf" }.to_string();
    }
    let s = format!("{:.17e}", x);
    let (mant, exp) = s.split_once('e').expect("{:e} always contains 'e'");
    let e: i32 = exp.parse().expect("{:e} exponent is an integer");
    let esign = if e < 0 { '-' } else { '+' };
    format!("{mant}e{esign}{:02}", e.abs())
}

impl KbrefData {
    /// Integer metadata entry.
    pub fn kbref_int(&self, key: &str) -> Result<i64, KbrefError> {
        let v = self
            .metadata
            .get(key)
            .ok_or_else(|| self.err(0, &format!("missing metadata `{key}`")))?;
        v.parse::<i64>()
            .map_err(|_| self.err(0, &format!("metadata `{key}` is not an integer")))
    }

    fn err(&self, line: usize, msg: &str) -> KbrefError {
        KbrefError {
            path: self.path.clone(),
            line,
            msg: msg.to_string(),
        }
    }
}

/// Read and strictly validate a kbref-v1 file. Returns an error for any
/// violation of the schema (version, data_kind, columns, component
/// names, index domain, finite values, duplicated keys, completeness).
pub fn read_kbref(path: &str) -> Result<KbrefData, KbrefError> {
    let text = fs::read_to_string(path)
        .map_err(|e| KbrefError {
            path: path.to_string(),
            line: 0,
            msg: format!("cannot read file: {e}"),
        })?;
    let mut data = KbrefData {
        path: path.to_string(),
        metadata: BTreeMap::new(),
        rows: Vec::new(),
    };
    let mut keys: BTreeSet<(String, i64, i64, i64, i64)> = BTreeSet::new();
    let mut header = false;
    let mut data_started = false;
    let mut kind = String::new();

    for (ln, line) in text.lines().enumerate() {
        let ln = ln + 1;
        let s = line.trim();
        if s.is_empty() {
            return Err(data.err(ln, "empty line"));
        }
        if let Some(body) = s.strip_prefix('#') {
            if data_started {
                return Err(data.err(ln, "metadata after data"));
            }
            let body = body.trim();
            let (key, val) = body
                .split_once(char::is_whitespace)
                .ok_or_else(|| data.err(ln, "bad metadata line"))?;
            let val = val.trim();
            if !header {
                if key != "kbref" || val != KBREF_VERSION {
                    return Err(data.err(ln, "first line must be `# kbref 1`"));
                }
                header = true;
                continue;
            }
            if data.metadata.contains_key(key) {
                return Err(data.err(ln, &format!("duplicated metadata `{key}`")));
            }
            data.metadata.insert(key.to_string(), val.to_string());
            continue;
        }
        if !header {
            return Err(data.err(ln, "missing `# kbref 1` header"));
        }
        if !data_started {
            kind = data
                .metadata
                .get("data_kind")
                .cloned()
                .unwrap_or_default();
            if !KBREF_KINDS.contains(&kind.as_str()) {
                return Err(data.err(ln, &format!("unknown data_kind `{kind}`")));
            }
            if KBREF_P2_KINDS.contains(&kind.as_str())
                || data
                    .metadata
                    .get("generator")
                    .map(|g| g.starts_with("p2_"))
                    .unwrap_or(false)
            {
                for key in ["generator", "case_set"] {
                    if !data.metadata.contains_key(key) {
                        return Err(data.err(ln, &format!("missing metadata `{key}`")));
                    }
                }
            }
            // P4 files: one case per file, the case and the statistics
            // are metadata only
            if data
                .metadata
                .get("generator")
                .map(|g| g.starts_with("phase4_"))
                .unwrap_or(false)
            {
                for key in ["case", "sig"] {
                    if !data.metadata.contains_key(key) {
                        return Err(data.err(ln, &format!("missing metadata `{key}`")));
                    }
                }
                let sig = data.metadata.get("sig").map(|s| s.as_str()).unwrap_or("");
                if sig != "-1" && sig != "1" {
                    return Err(data.err(ln, "bad metadata `sig`"));
                }
            }
            if kind == "herm_matrix"
                && data
                    .metadata
                    .get("generator")
                    .map(|g| g.starts_with("p2_"))
                    .unwrap_or(false)
            {
                match data
                    .metadata
                    .get("case_id")
                    .and_then(|v| v.parse::<i64>().ok())
                {
                    Some(v) if v >= 0 => {}
                    _ => return Err(data.err(ln, "missing or bad metadata `case_id`")),
                }
            }
            if KBREF_P7_KINDS.contains(&kind.as_str()) {
                for key in ["generator", "case_set"] {
                    if !data.metadata.contains_key(key) {
                        return Err(data.err(ln, &format!("missing metadata `{key}`")));
                    }
                }
                if kind == "phase7_example" {
                    let case = data.metadata.get("case").map(|s| s.as_str()).unwrap_or("");
                    if !KBREF_P7_CASES.contains(&case) {
                        return Err(data.err(ln, "missing or bad metadata `case`"));
                    }
                } else if data.kbref_int("size_max")? < 1 {
                    return Err(data.err(ln, "bad metadata `size_max`"));
                }
                if kind == "phase7_ness" && data.kbref_int("nft_max")? < 2 {
                    return Err(data.err(ln, "bad metadata `nft_max`"));
                }
            }
            if kind == "moving" {
                for key in KBREF_MOVING_META {
                    if !data.metadata.contains_key(key) {
                        return Err(data.err(ln, &format!("missing metadata `{key}`")));
                    }
                }
                let generator = data.metadata["generator"].as_str();
                if !KBREF_MOVING_GENERATORS.contains(&generator) {
                    return Err(data.err(ln, "bad generator for moving data"));
                }
                if data.kbref_int("tc")? < 0 || data.kbref_int("size")? < 1 {
                    return Err(data.err(ln, "bad metadata `tc`/`size`"));
                }
                if data.kbref_int("case_id")? < 0 {
                    return Err(data.err(ln, "bad metadata `case_id`"));
                }
                if data.kbref_int("tstp")? < 0 {
                    return Err(data.err(ln, "bad metadata `tstp`"));
                }
                let sign = data.metadata["sign"].as_str();
                if sign != "-1" && sign != "1" {
                    return Err(data.err(ln, "bad metadata `sign`"));
                }
            }
            if kind == "series" {
                for key in KBREF_SERIES_META {
                    if !data.metadata.contains_key(key) {
                        return Err(data.err(ln, &format!("missing metadata `{key}`")));
                    }
                }
                let cs = data.metadata["case_set"].as_str();
                if series_class(cs, "").is_none() && !KBREF_SERIES_CASE_SETS.contains(&cs) {
                    return Err(data.err(ln, &format!("unknown series case_set `{cs}`")));
                }
                if !KBREF_SERIES_CASE_SETS.contains(&cs) {
                    return Err(data.err(ln, &format!("unknown series case_set `{cs}`")));
                }
                if data.metadata["generator"] != series_generator(cs).unwrap_or("") {
                    return Err(data.err(ln, "generator does not match the case_set"));
                }
                let comps: Vec<&str> = data.metadata["components"].split(',').collect();
                let mut unique = BTreeSet::new();
                let mut all_known = true;
                for c in &comps {
                    unique.insert(*c);
                    if series_class(cs, c).is_none() {
                        all_known = false;
                    }
                }
                if unique.len() != comps.len() || !all_known || comps.is_empty() {
                    return Err(data.err(ln, "bad metadata `components`"));
                }
                if data.kbref_int("nsteps")? < 1
                    || data.kbref_int("nk")? < 1
                    || data.kbref_int("size")? < 1
                {
                    return Err(data.err(ln, "bad metadata `nsteps`/`nk`/`size`"));
                }
            }
            if kind == "observable" || kind == "selected_green" {
                for key in KBREF_P5_META {
                    if !data.metadata.contains_key(key) {
                        return Err(data.err(ln, &format!("missing metadata `{key}`")));
                    }
                }
                let model = data.metadata["model"].as_str();
                let approx = data.metadata["approximation"].as_str();
                let ok = p5_approximations(model).map(|a| a.contains(&approx)).unwrap_or(false);
                if !ok {
                    return Err(data.err(ln, "bad metadata `model`/`approximation`"));
                }
                if kind != "observable" && model != "hubbard" {
                    return Err(data.err(ln, "selected_green is only defined for hubbard"));
                }
                let profile = data.metadata["profile"].as_str();
                if profile != "quick" && profile != "release" {
                    return Err(data.err(ln, "bad metadata `profile`"));
                }
                if data.kbref_int("nt")? < 1 || data.kbref_int("size")? < 1 {
                    return Err(data.err(ln, "bad metadata `nt`/`size`"));
                }
            }
            if kind == "pseudo" {
                for key in KBREF_PSEUDO_META {
                    if !data.metadata.contains_key(key) {
                        return Err(data.err(ln, &format!("missing metadata `{key}`")));
                    }
                }
                let sig = data.metadata["sig"].as_str();
                if sig != "-1" && sig != "1" {
                    return Err(data.err(ln, "bad metadata `sig`"));
                }
            }
            if data.metadata.get("columns").map(|c| c.as_str()) != Some(KBREF_COLUMNS) {
                return Err(data.err(ln, "bad columns specification"));
            }
            data_started = true;
        }
        // data line
        let f: Vec<&str> = s.split_whitespace().collect();
        if f.len() != 7 {
            return Err(data.err(ln, &format!("expected 7 columns, got {}", f.len())));
        }
        let comp = f[0].to_string();
        let known = if kind == "series" {
            KBREF_SERIES_CASE_SETS
                .iter()
                .any(|cs| series_class(cs, &comp).is_some())
        } else if kind == "observable" {
            P5_MODELS
                .iter()
                .any(|m| observable_class(m, &comp).is_some())
        } else {
            kbref_components(&kind)
                .map(|c| c.contains(&comp.as_str()))
                .unwrap_or(false)
        };
        if !known {
            return Err(data.err(ln, &format!("unknown component `{comp}`")));
        }
        let parse_idx = |t: &str| -> Result<i64, KbrefError> {
            t.parse::<i64>()
                .map_err(|_| data.err(ln, "bad integer index"))
        };
        let i = parse_idx(f[1])?;
        let j = parse_idx(f[2])?;
        let a = parse_idx(f[3])?;
        let b = parse_idx(f[4])?;
        let re: f64 = f[5]
            .parse()
            .map_err(|_| data.err(ln, "bad number"))?;
        let im: f64 = f[6]
            .parse()
            .map_err(|_| data.err(ln, "bad number"))?;
        if !re.is_finite() || !im.is_finite() {
            return Err(data.err(ln, "non-finite value"));
        }
        let row = KbrefRow {
            component: comp,
            i,
            j,
            a,
            b,
            value: Complex::new(re, im),
        };
        // per-kind index-domain validation
        match kind.as_str() {
            "weights" => check_weights(&row, &data, ln)?,
            "iteration" => check_iteration(&row, &data, ln)?,
            "pseudo" => check_pseudo(&row, &data, ln)?,
            k if KBREF_P7_KINDS.contains(&k) => check_phase7(&row, &data, ln)?,
            "moving" => check_moving(&row, &data, ln)?,
            "series" => check_series(&row, &data, ln)?,
            "observable" | "selected_green" => check_p5(&row, &data, ln)?,
            k if KBREF_P2_KINDS.contains(&k) => check_p2(&row, &data, ln)?,
            _ => check_herm(&row, &data, ln)?,
        }
        let key = (row.component.clone(), row.i, row.j, row.a, row.b);
        if !keys.insert(key) {
            return Err(data.err(ln, "duplicated key"));
        }
        data.rows.push(row);
    }
    if !header {
        return Err(data.err(0, "missing `# kbref 1` header"));
    }
    if !data_started {
        return Err(data.err(0, "no data rows"));
    }
    // Completeness of the independent storage domain
    if kind == "herm_matrix" {
        let (nt, ntau) = (data.kbref_int("nt")?, data.kbref_int("ntau")?);
        let s = data.kbref_int("size1")? * data.kbref_int("size2")?;
        let ntri = (nt + 1) * (nt + 2) / 2;
        let nexp = s * ((ntau + 1) + 2 * ntri + (nt + 1) * (ntau + 1));
        if data.rows.len() as i64 != nexp {
            return Err(data.err(
                0,
                &format!("incomplete data: {} rows, expected {nexp}", data.rows.len()),
            ));
        }
    }
    if kind == "pseudo" {
        let (nt, ntau) = (data.kbref_int("nt")?, data.kbref_int("ntau")?);
        let s = data.kbref_int("size1")? * data.kbref_int("size2")?;
        let ntri = (nt + 1) * (nt + 2) / 2;
        let cnt = |c: &str| data.rows.iter().filter(|r| r.component == c).count() as i64;
        let comps = ["mat", "gtr", "tv", "les"];
        let nexp = |c: &str| -> i64 {
            match c {
                "mat" => s * (ntau + 1),
                "gtr" => s * ntri,
                "tv" => s * (nt + 1) * (ntau + 1),
                _ => s * ntri,
            }
        };
        if comps.iter().any(|c| cnt(c) > 0) && comps.iter().any(|c| cnt(c) != nexp(c)) {
            return Err(data.err(0, "incomplete pseudo storage"));
        }
    }
    if kind == "phase7_fourier" {
        for c in ["fft_input", "fft_forward", "fft_backward"] {
            let mut ns: BTreeSet<i64> = BTreeSet::new();
            for r in &data.rows {
                if r.component == c {
                    ns.insert(r.i);
                }
            }
            for n in ns {
                let js: BTreeSet<i64> = data
                    .rows
                    .iter()
                    .filter(|r| r.component == c && r.i == n)
                    .map(|r| r.j)
                    .collect();
                let want: BTreeSet<i64> = (0..n).collect();
                if js != want {
                    return Err(data.err(0, &format!("incomplete `{c}` data for N = {n}")));
                }
            }
        }
    }
    if kind == "moving" {
        let (tc, s) = (data.kbref_int("tc")?, data.kbref_int("size")?);
        let nexp = 2 * (tc + 1).pow(2) * s.pow(2);
        if data.rows.len() as i64 != nexp {
            return Err(data.err(0, "incomplete moving window"));
        }
    }
    if kind == "selected_green" && data.rows.len() != KBREF_SELECTED_GREEN_KEYS.len() {
        return Err(data.err(0, "incomplete selected_green data"));
    }
    if kind == "series" {
        let want: BTreeSet<&str> = data.metadata["components"].split(',').collect();
        let got: BTreeSet<&str> = data.rows.iter().map(|r| r.component.as_str()).collect();
        if want != got {
            return Err(data.err(0, "the component set does not match the metadata"));
        }
    }
    if kind == "iteration" {
        let (niter, ntau) = (data.kbref_int("niter")?, data.kbref_int("ntau")?);
        let s = data.kbref_int("size1")? * data.kbref_int("size2")?;
        let cg = data.metadata["algorithm"] == "cg";
        let nexp = (if cg { niter } else { niter + 1 }) * (ntau + 1) * s;
        let gcount = data.rows.iter().filter(|r| r.component == "g").count() as i64;
        if gcount != nexp {
            return Err(data.err(0, "incomplete iteration data `g`"));
        }
        if cg {
            for c in KBREF_CG_SCALARS {
                let is: BTreeSet<i64> = data
                    .rows
                    .iter()
                    .filter(|r| r.component == c)
                    .map(|r| r.i)
                    .collect();
                let want: BTreeSet<i64> = (1..=niter).collect();
                if is != want {
                    return Err(data.err(0, &format!("incomplete iteration data `{c}`")));
                }
            }
        }
    }
    Ok(data)
}

/// All `series` case sets.
pub const KBREF_SERIES_CASE_SETS: [&str; 4] =
    ["moving_solver", "trunc_bethe", "ness2_bethe_trunc", "gw"];

/// All P5 observable models.
pub const P5_MODELS: [&str; 5] = [
    "hubbard",
    "fkm",
    "holstein_bethe",
    "holstein_nambu",
    "holstein_impurity",
];

/// Components (unsorted) of a `series` case set.
pub fn kbref_series_components(cs: &str) -> Option<Vec<&'static str>> {
    let mut out = Vec::new();
    for c in [
        "kinetic", "density", "error", "ekin", "afield", "current_nessi", "eint",
        "dyson_ret", "dyson_less", "dyson0_ret", "dyson0_less", "bubble1_ret",
        "bubble1_less", "bubble2_ret", "bubble2_less", "vie2_ret", "vie2_less",
        "g_ret", "g_less", "sigma_ret", "sigma_less", "rho", "sigma_hf", "w_ret",
        "w_less", "sigma_gw_ret", "sigma_gw_less", "p_ret", "p_less",
    ] {
        if series_class(cs, c).is_some() {
            out.push(c);
        }
    }
    if out.is_empty() { None } else { Some(out) }
}

/// Components (unsorted) of a P5 observable model.
pub fn kbref_observable_components(model: &str) -> Option<Vec<&'static str>> {
    let mut out = Vec::new();
    for c in [
        "n1", "Ekin", "Epot", "Etot", "den0", "den1", "total", "rho", "Xph",
        "Pph", "XX", "Enx_MF", "Enx_corr", "Eph_cl", "Eph_corr", "Eph",
    ] {
        if observable_class(model, c).is_some() {
            out.push(c);
        }
    }
    if out.is_empty() { None } else { Some(out) }
}

fn check_weights(r: &KbrefRow, d: &KbrefData, ln: usize) -> Result<(), KbrefError> {
    let k = r.i;
    let e = |m: &str| Err(d.err(ln, m));
    if !(1..=5).contains(&k) {
        return e(&format!("k = {k} out of range 1:5"));
    }
    if r.value.im != 0.0 {
        return e("weights must be real");
    }
    let ok = if r.component == "greg" {
        let (n, j) = (r.j, r.a);
        0 <= n && n <= 2 * k + 3 && 0 <= j && j <= n.max(k) && r.b == -1
    } else if r.component == "bd" {
        0 <= r.j && r.j <= k + 1 && r.a == -1 && r.b == -1
    } else if r.component == "pint" {
        0 <= r.j && r.j <= k && 0 <= r.a && r.a <= k && 0 <= r.b && r.b <= k
    } else if r.component == "pdiff" {
        0 <= r.j && r.j <= k && 0 <= r.a && r.a <= k && r.b == -1
    } else {
        k >= 2 && 1 <= r.j && r.j <= k - 1 && 0 <= r.a && r.a <= k && 0 <= r.b && r.b <= k
    };
    if !ok {
        return e(&format!("index out of range for `{}`", r.component));
    }
    Ok(())
}

fn check_herm(r: &KbrefRow, d: &KbrefData, ln: usize) -> Result<(), KbrefError> {
    let nt = d.kbref_int("nt")?;
    let ntau = d.kbref_int("ntau")?;
    let s1 = d.kbref_int("size1")?;
    let s2 = d.kbref_int("size2")?;
    if !(0 <= r.a && r.a < s1 && 0 <= r.b && r.b < s2) {
        return Err(d.err(ln, "orbital index out of range"));
    }
    let (c, i, j) = (r.component.as_str(), r.i, r.j);
    let ok = match c {
        "mat" => 0 <= i && i <= ntau && j == -1,
        "ret" => 0 <= j && j <= i && i <= nt,
        "tv" => 0 <= i && i <= nt && 0 <= j && j <= ntau,
        _ => 0 <= i && i <= j && j <= nt, // les
    };
    if !ok {
        return Err(d.err(ln, &format!("index out of range for `{c}`")));
    }
    Ok(())
}

fn check_p2(r: &KbrefRow, d: &KbrefData, ln: usize) -> Result<(), KbrefError> {
    let kind = &d.metadata["data_kind"];
    let c = &r.component;
    if r.i < 0 {
        return Err(d.err(ln, "negative case_id"));
    }
    if kind == "scalar" || kind == "labels" {
        if !(r.j == -1 && r.a == -1 && r.b == -1) {
            return Err(d.err(ln, &format!("index out of range for `{c}`")));
        }
        if c.starts_with("distance") {
            if r.value.im != 0.0 {
                return Err(d.err(ln, "distance must be real"));
            }
        } else if c == "order" {
            let v = r.value.re;
            if !(r.value.im == 0.0 && (v - v.round()).abs() < 1e-12 && (0.0..=5.0).contains(&v)) {
                return Err(d.err(ln, "bad order label"));
            }
        }
    } else {
        // function, element
        let nt = d.kbref_int("nt")?;
        let s1 = d.kbref_int("size1")?;
        let s2 = d.kbref_int("size2")?;
        if !(0 <= r.a && r.a < s1 && 0 <= r.b && r.b < s2) {
            return Err(d.err(ln, "orbital index out of range"));
        }
        if c == "interpolate" {
            if !d.metadata.contains_key("tinter") {
                return Err(d.err(ln, "missing metadata `tinter`"));
            }
            if r.j != -1 {
                return Err(d.err(ln, &format!("index out of range for `{c}`")));
            }
        } else if c.starts_with("startup") {
            if !(0 <= r.j && r.j <= nt) {
                return Err(d.err(ln, &format!("index out of range for `{c}`")));
            }
        } else if !(-1 <= r.j && r.j <= nt) {
            return Err(d.err(ln, &format!("index out of range for `{c}`")));
        }
    }
    Ok(())
}

fn check_pseudo(r: &KbrefRow, d: &KbrefData, ln: usize) -> Result<(), KbrefError> {
    let nt = d.kbref_int("nt")?;
    let ntau = d.kbref_int("ntau")?;
    let s1 = d.kbref_int("size1")?;
    let s2 = d.kbref_int("size2")?;
    let k = d.kbref_int("k")?;
    if !(0 <= r.a && r.a < s1 && 0 <= r.b && r.b < s2) {
        return Err(d.err(ln, "orbital index out of range"));
    }
    let (c, i, j) = (r.component.as_str(), r.i, r.j);
    let ok = match c {
        "mat" => 0 <= i && i <= ntau && j == -1,
        "gtr" => 0 <= j && j <= i && i <= nt,
        "tv" => 0 <= i && i <= nt && 0 <= j && j <= ntau,
        "les" => 0 <= i && i <= j && j <= nt,
        "density" => -1 <= i && i <= nt && j == -1,
        _ => 0 <= j && j < i && i <= k, // les_start_source
    };
    if !ok {
        return Err(d.err(ln, &format!("index out of range for `{c}`")));
    }
    Ok(())
}

fn check_iteration(r: &KbrefRow, d: &KbrefData, ln: usize) -> Result<(), KbrefError> {
    let alg = d.metadata.get("algorithm").cloned().unwrap_or_default();
    if alg != "fixpoint" && alg != "cg" {
        return Err(d.err(ln, "bad metadata `algorithm`"));
    }
    let niter = d.kbref_int("niter")?;
    let ntau = d.kbref_int("ntau")?;
    let s1 = d.kbref_int("size1")?;
    let s2 = d.kbref_int("size2")?;
    let imin = if alg == "fixpoint" { 0 } else { 1 };
    if r.component == "g" {
        if !(imin <= r.i && r.i <= niter && 0 <= r.j && r.j <= ntau && 0 <= r.a && r.a < s1 && 0 <= r.b && r.b < s2) {
            return Err(d.err(ln, "index out of range for `g`"));
        }
    } else {
        if alg != "cg" {
            return Err(d.err(ln, &format!("scalar `{}` is only allowed for cg", r.component)));
        }
        if !(1 <= r.i && r.i <= niter && r.j == -1 && r.a == -1 && r.b == -1) {
            return Err(d.err(ln, &format!("index out of range for `{}`", r.component)));
        }
        if r.value.im != 0.0 {
            return Err(d.err(ln, "iteration scalar must be real"));
        }
    }
    Ok(())
}

fn check_phase7(r: &KbrefRow, d: &KbrefData, ln: usize) -> Result<(), KbrefError> {
    let kind = &d.metadata["data_kind"];
    let (c, i, j, a, b, v) = (&r.component, r.i, r.j, r.a, r.b, r.value);
    let (cid, smax) = if kind == "phase7_example" {
        let cid = KBREF_P7_CASES
            .iter()
            .position(|x| *x == d.metadata["case"])
            .unwrap_or(usize::MAX) as i64;
        (cid, 1i64)
    } else {
        (-1i64, d.kbref_int("size_max")?)
    };
    let orb = 0 <= a && a < smax && 0 <= b && b < smax;
    let ok = if kind == "phase7_fourier" {
        if c == "fft_input" || c == "fft_forward" || c == "fft_backward" {
            i >= 1 && 0 <= j && j < i && a == 0 && b == 0
        } else if c == "dftcorr" {
            i >= 0 && (j == 1 || j == 3)
                && b == 0
                && 0 <= a
                && a <= if j == 1 { 3 } else { 5 }
                && (a >= 2 || v.im == 0.0)
        } else if c == "adaptive" {
            0 <= i && i <= 2 && j >= 0 && 0 <= a && a <= 2 && b == 0 && (a != 0 || v.im == 0.0)
        } else if c == "matsubara_ft_result" {
            i >= 0 && (i % 10 == 0 || i % 10 == 1 || i % 10 == 3) && orb
        } else {
            i >= 0 && j >= 0 && orb
        }
    } else if kind == "phase7_ness" {
        let nft = d.kbref_int("nft_max")?;
        if c == "ret_time" || c == "les_time" || c == "ret_freq" || c == "les_freq" {
            i >= 0 && 0 <= j && j < nft && orb
        } else if c == "bath_ret" || c == "bath_les" {
            0 <= i && i <= 3 && j >= 0 && a == 0 && b == 0
        } else if c == "density" || c == "convolution_density" {
            i >= 0 && j == 0 && orb
        } else {
            i >= 0 && 0 <= j && j < 2 * nft && orb
        }
    } else {
        // phase7_example
        let prop = cid == 2;
        if i != cid {
            false
        } else if c == "gf_ret_time" || c == "gf_les_time" {
            j >= 0 && a == 0 && b == 0
        } else if c == "gf_ret_freq" || c == "gf_les_freq" {
            !prop && j >= 0 && a == 0 && b == 0
        } else if c == "density" {
            (if prop { j >= 0 } else { j == 0 }) && a == 0 && b == 0 && v.im == 0.0
        } else if c == "current" {
            cid == 0 && (j == 0 || j == 1) && a == 0 && b == 0 && v.im == 0.0
        } else if c == "eint" {
            cid <= 1 && j == 0 && a == 0 && b == 0 && v.im == 0.0
        } else {
            // iteration_error
            v.im == 0.0
                && ((cid == 1 && j >= 1 && a == 0 && b == 0)
                    || (cid == 2 && j >= 0 && 0 <= a && a <= 2 && b >= 0))
        }
    };
    if !ok {
        return Err(d.err(ln, &format!("index out of range for `{c}`")));
    }
    Ok(())
}

fn check_moving(r: &KbrefRow, d: &KbrefData, ln: usize) -> Result<(), KbrefError> {
    let tc = d.kbref_int("tc")?;
    let s = d.kbref_int("size")?;
    if !(0 <= r.i && r.i <= tc && 0 <= r.j && r.j <= tc && 0 <= r.a && r.a < s && 0 <= r.b && r.b < s) {
        return Err(d.err(ln, &format!("index out of range for `{}`", r.component)));
    }
    Ok(())
}

fn check_series(r: &KbrefRow, d: &KbrefData, ln: usize) -> Result<(), KbrefError> {
    let cs = d.metadata["case_set"].as_str();
    let cls = series_class(cs, &r.component);
    if cls.is_none() {
        return Err(d.err(ln, &format!("component `{}` not allowed in this case_set", r.component)));
    }
    if !d.metadata["components"].split(',').any(|x| x == r.component) {
        return Err(d.err(ln, &format!("component `{}` not listed in the metadata", r.component)));
    }
    let nsteps = d.kbref_int("nsteps")?;
    let nk = d.kbref_int("nk")?;
    let s = d.kbref_int("size")?;
    let c = cls.unwrap();
    let mut ok = 0 <= r.i && r.i < nsteps;
    if c == 0 {
        ok = ok && r.j == -1 && r.a == -1 && r.b == -1;
    } else if c == 1 {
        ok = ok && 0 <= r.j && r.j < nk && r.a == -1 && r.b == -1;
    } else {
        ok = ok && 0 <= r.j && r.j < nk && 0 <= r.a && r.a < s && 0 <= r.b && r.b < s;
    }
    if !ok {
        return Err(d.err(ln, &format!("index out of range for `{}`", r.component)));
    }
    Ok(())
}

fn check_p5(r: &KbrefRow, d: &KbrefData, ln: usize) -> Result<(), KbrefError> {
    if d.metadata["data_kind"] == "selected_green" {
        let is_key = KBREF_SELECTED_GREEN_KEYS
            .iter()
            .any(|(c, i, j, a, b)| *c == r.component && *i == r.i && *j == r.j && *a == r.a && *b == r.b);
        if !is_key {
            return Err(d.err(ln, "bad selected_green key"));
        }
        return Ok(());
    }
    let model = d.metadata["model"].as_str();
    let cls = observable_class(model, &r.component);
    if cls.is_none() {
        return Err(d.err(ln, &format!("component `{}` not allowed for this model", r.component)));
    }
    let nt = d.kbref_int("nt")?;
    let s = d.kbref_int("size")?;
    let mut ok = -1 <= r.i && r.i <= nt && r.j == -1;
    if cls == Some(0) {
        ok = ok && r.a == -1 && r.b == -1 && r.value.im == 0.0;
    } else {
        ok = ok && 0 <= r.a && r.a < s && 0 <= r.b && r.b < s;
    }
    if !ok {
        return Err(d.err(ln, &format!("index or value out of range for `{}`", r.component)));
    }
    Ok(())
}

/// Write a kbref-v1 file (schema round-trip tests only).
pub fn write_kbref(path: &str, data_kind: &str, meta: &[(String, String)], rows: &[KbrefRow]) -> std::io::Result<()> {
    use std::io::Write;
    let mut out = String::new();
    out.push_str("# kbref 1\n");
    out.push_str(&format!("# data_kind {data_kind}\n"));
    for (k, v) in meta {
        out.push_str(&format!("# {k} {v}\n"));
    }
    out.push_str(&format!("# columns {KBREF_COLUMNS}\n"));
    for r in rows {
        out.push_str(&format!(
            "{} {} {} {} {} {} {}\n",
            r.component,
            r.i,
            r.j,
            r.a,
            r.b,
            fmt_c_e17(r.value.re),
            fmt_c_e17(r.value.im)
        ));
    }
    fs::File::create(path)?.write_all(out.as_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fmt_c_e17_matches_c_printout() {
        // Values and their C printf("%.17e") outputs (gcc probe)
        assert_eq!(fmt_c_e17(0.0), "0.00000000000000000e+00");
        assert_eq!(fmt_c_e17(-0.0), "-0.00000000000000000e+00");
        assert_eq!(fmt_c_e17(0.5), "5.00000000000000000e-01");
        assert_eq!(fmt_c_e17(4.16666666666666685e-01), "4.16666666666666685e-01");
        assert_eq!(fmt_c_e17(1.0), "1.00000000000000000e+00");
        assert_eq!(fmt_c_e17(-2.5), "-2.50000000000000000e+00");
        assert_eq!(fmt_c_e17(10.0), "1.00000000000000000e+01");
        assert_eq!(fmt_c_e17(0.1), "1.00000000000000006e-01");
        assert_eq!(fmt_c_e17(1.0e-7), "9.99999999999999955e-08");
        assert_eq!(fmt_c_e17(5.0e-324), "4.94065645841246544e-324");
        assert_eq!(fmt_c_e17(1.7976931348623157e308), "1.79769313486231571e+308");
        assert_eq!(fmt_c_e17(6.02e23), "6.01999999999999996e+23");
        assert_eq!(fmt_c_e17(3.141592653589793), "3.14159265358979312e+00");
        assert_eq!(fmt_c_e17(1.0e100), "1.00000000000000002e+100");
        // roundtrip
        for x in [
            0.1f64,
            3.141592653589793,
            -6.02e23,
            5.0e-324,
            1.7976931348623157e308,
        ] {
            let s = fmt_c_e17(x);
            let back: f64 = s.parse().unwrap();
            assert_eq!(back, x);
        }
    }
}
