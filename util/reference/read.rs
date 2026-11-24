// Project : KadanoffBaym
// File    : util/reference/read.rs
// Purpose : the ref-v1 reader: the row and file types, and the
//           strict validation of the metadata block, the single rows,
//           and the completeness of the stored domain
//

use std::collections::{BTreeMap, BTreeSet};

use std::fs::read_to_string;

use kadanoffbaym::complex::Complex;

use super::error::RefError;

use super::schema::{
    REF_CG_SCALARS, REF_COLUMNS, REF_KINDS, REF_MOVING_GENERATORS, REF_MOVING_META,
    REF_OBSERVABLE_META, REF_PSEUDO_META, REF_REFERENCE_CASES, REF_REFERENCE_KINDS,
    REF_SELECTED_GREEN_KEYS, REF_SERIES_CASE_SETS, REF_SERIES_META, REF_SINGLE_CASE_GENERATORS,
    REF_TOOL_GENERATORS, REF_TOOL_KINDS, REF_VERSION,
};

use super::check::check_row;
use super::schema::{approximations, known_component, series_class, series_generator};

// ------------------------------------------------------------------
// the row and file types
// ------------------------------------------------------------------

/// One data line of a ref-v1 file.
#[derive(Clone, Debug, PartialEq)]
pub struct RefRow {
    pub component: String,
    pub i: i64,
    pub j: i64,
    pub a: i64,
    pub b: i64,
    pub value: Complex,
}

/// Content of a ref-v1 file.
#[derive(Clone, Debug, PartialEq)]
pub struct RefData {
    pub path: String,
    pub metadata: BTreeMap<String, String>,
    pub rows: Vec<RefRow>,
}

impl RefData {
    /// Integer metadata entry.
    pub(super) fn int_meta(&self, key: &str) -> Result<i64, RefError> {
        let v = self
            .metadata
            .get(key)
            .ok_or_else(|| self.err(0, &format!("missing metadata `{key}`")))?;
        v.parse::<i64>()
            .map_err(|_| self.err(0, &format!("metadata `{key}` is not an integer")))
    }

    pub(super) fn err(&self, line: usize, msg: &str) -> RefError {
        RefError {
            path: self.path.clone(),
            line,
            msg: msg.to_string(),
        }
    }
}

// ------------------------------------------------------------------
// the reader
// ------------------------------------------------------------------

/// Read and strictly validate a ref-v1 file. Returns an error for any
/// violation of the schema (version, data_kind, columns, component
/// names, index domain, finite values, duplicated keys, completeness).
pub fn read_ref(path: &str) -> Result<RefData, RefError> {
    let text = read_to_string(path).map_err(|e| RefError {
        path: path.to_string(),
        line: 0,
        msg: format!("cannot read file: {e}"),
    })?;
    let mut data = RefData {
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
                if key != "ref" || val != REF_VERSION {
                    return Err(data.err(ln, "first line must be `# ref 1`"));
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
            return Err(data.err(ln, "missing `# ref 1` header"));
        }
        if !data_started {
            kind = validate_metadata(&data, ln)?;
            data_started = true;
        }
        // data line
        let row = parse_row(s, &kind, &data, ln)?;
        check_row(&kind, &row, &data, ln)?;
        let key = (row.component.clone(), row.i, row.j, row.a, row.b);
        if !keys.insert(key) {
            return Err(data.err(ln, "duplicated key"));
        }
        data.rows.push(row);
    }
    if !header {
        return Err(data.err(0, "missing `# ref 1` header"));
    }
    if !data_started {
        return Err(data.err(0, "no data rows"));
    }
    check_completeness(&data, &kind)?;
    Ok(data)
}

// ------------------------------------------------------------------
// the metadata block (checked once, at the first data line)
// ------------------------------------------------------------------

/// The data kind, the keys the generator families and the data kinds
/// require, the value domains, and the column specification. Returns the
/// data kind.
fn validate_metadata(data: &RefData, ln: usize) -> Result<String, RefError> {
    let kind = data.metadata.get("data_kind").cloned().unwrap_or_default();
    if !REF_KINDS.contains(&kind.as_str()) {
        return Err(data.err(ln, &format!("unknown data_kind `{kind}`")));
    }
    check_generator_meta(data, ln, &kind)?;
    if REF_REFERENCE_KINDS.contains(&kind.as_str()) {
        check_reference_meta(data, ln, &kind)?;
    }
    if kind == "moving" {
        check_moving_meta(data, ln)?;
    }
    if kind == "series" {
        check_series_meta(data, ln)?;
    }
    if kind == "observable" || kind == "selected_green" {
        check_observable_meta(data, ln, &kind)?;
    }
    if kind == "pseudo" {
        check_pseudo_meta(data, ln)?;
    }
    if data.metadata.get("columns").map(|c| c.as_str()) != Some(REF_COLUMNS) {
        return Err(data.err(ln, "bad columns specification"));
    }
    Ok(kind)
}

/// Every key of `keys` must be present in the metadata.
fn require_meta(data: &RefData, ln: usize, keys: &[&str]) -> Result<(), RefError> {
    for key in keys {
        if !data.metadata.contains_key(*key) {
            return Err(data.err(ln, &format!("missing metadata `{key}`")));
        }
    }
    Ok(())
}

/// The generator families: the tool kinds (and the tool generators)
/// carry `generator` and `case_set`; the single-case reference
/// generators carry `case` and the statistics instead; the
/// `herm_matrix` files of the tool generators carry a non-negative
/// `case_id`.
fn check_generator_meta(data: &RefData, ln: usize, kind: &str) -> Result<(), RefError> {
    let generator = data
        .metadata
        .get("generator")
        .map(|g| g.as_str())
        .unwrap_or("");
    let single_case = REF_SINGLE_CASE_GENERATORS.contains(&generator);
    let tool = REF_TOOL_KINDS.contains(&kind)
        || REF_TOOL_GENERATORS.contains(&generator)
        || data.metadata.contains_key("case_set");
    if tool {
        require_meta(data, ln, &["generator", "case_set"])?;
    }
    if single_case {
        require_meta(data, ln, &["case", "sig"])?;
        let sig = data.metadata.get("sig").map(|s| s.as_str()).unwrap_or("");
        if sig != "-1" && sig != "1" {
            return Err(data.err(ln, "bad metadata `sig`"));
        }
    }
    if kind == "herm_matrix" && REF_TOOL_GENERATORS.contains(&generator) {
        match data
            .metadata
            .get("case_id")
            .and_then(|v| v.parse::<i64>().ok())
        {
            Some(v) if v >= 0 => {}
            _ => return Err(data.err(ln, "missing or bad metadata `case_id`")),
        }
    }
    Ok(())
}

/// The reference kinds: `generator` and `case_set`; the `case` of
/// `example`, otherwise a positive `size_max`; `steady` also needs
/// `nft_max >= 2`.
fn check_reference_meta(data: &RefData, ln: usize, kind: &str) -> Result<(), RefError> {
    require_meta(data, ln, &["generator", "case_set"])?;
    if kind == "example" {
        let case = data.metadata.get("case").map(|s| s.as_str()).unwrap_or("");
        if !REF_REFERENCE_CASES.contains(&case) {
            return Err(data.err(ln, "missing or bad metadata `case`"));
        }
    } else if data.int_meta("size_max")? < 1 {
        return Err(data.err(ln, "bad metadata `size_max`"));
    }
    if kind == "steady" && data.int_meta("nft_max")? < 2 {
        return Err(data.err(ln, "bad metadata `nft_max`"));
    }
    Ok(())
}

/// The `moving` kind: its metadata keys, the owning generator,
/// non-negative `tc`, `case_id`, and `tstp`, a positive `size`, and the
/// statistics sign.
fn check_moving_meta(data: &RefData, ln: usize) -> Result<(), RefError> {
    require_meta(data, ln, &REF_MOVING_META)?;
    let generator = data.metadata["generator"].as_str();
    if !REF_MOVING_GENERATORS.contains(&generator) {
        return Err(data.err(ln, "bad generator for moving data"));
    }
    if data.int_meta("tc")? < 0 || data.int_meta("size")? < 1 {
        return Err(data.err(ln, "bad metadata `tc`/`size`"));
    }
    if data.int_meta("case_id")? < 0 {
        return Err(data.err(ln, "bad metadata `case_id`"));
    }
    if data.int_meta("tstp")? < 0 {
        return Err(data.err(ln, "bad metadata `tstp`"));
    }
    let sign = data.metadata["sign"].as_str();
    if sign != "-1" && sign != "1" {
        return Err(data.err(ln, "bad metadata `sign`"));
    }
    Ok(())
}

/// The `series` kind: its metadata keys, a known case set with its
/// owning generator, a unique non-empty component list of known
/// components, and positive `nsteps`, `nk`, and `size`.
fn check_series_meta(data: &RefData, ln: usize) -> Result<(), RefError> {
    require_meta(data, ln, &REF_SERIES_META)?;
    let cs = data.metadata["case_set"].as_str();
    if series_class(cs, "").is_none() && !REF_SERIES_CASE_SETS.contains(&cs) {
        return Err(data.err(ln, &format!("unknown series case_set `{cs}`")));
    }
    if !REF_SERIES_CASE_SETS.contains(&cs) {
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
    if data.int_meta("nsteps")? < 1 || data.int_meta("nk")? < 1 || data.int_meta("size")? < 1 {
        return Err(data.err(ln, "bad metadata `nsteps`/`nk`/`size`"));
    }
    Ok(())
}

/// The `observable` and `selected_green` kinds: their metadata keys, a
/// known model with one of its approximations (the selected Green's
/// function keys exist for hubbard only), the profile, and positive
/// `nt` and `size`.
fn check_observable_meta(data: &RefData, ln: usize, kind: &str) -> Result<(), RefError> {
    require_meta(data, ln, &REF_OBSERVABLE_META)?;
    let model = data.metadata["model"].as_str();
    let approx = data.metadata["approximation"].as_str();
    let ok = approximations(model)
        .map(|a| a.contains(&approx))
        .unwrap_or(false);
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
    if data.int_meta("nt")? < 1 || data.int_meta("size")? < 1 {
        return Err(data.err(ln, "bad metadata `nt`/`size`"));
    }
    Ok(())
}

/// The `pseudo` kind: its metadata keys and the statistics sign.
fn check_pseudo_meta(data: &RefData, ln: usize) -> Result<(), RefError> {
    require_meta(data, ln, &REF_PSEUDO_META)?;
    let sig = data.metadata["sig"].as_str();
    if sig != "-1" && sig != "1" {
        return Err(data.err(ln, "bad metadata `sig`"));
    }
    Ok(())
}

// ------------------------------------------------------------------
// one data line
// ------------------------------------------------------------------

/// Parse one data line (`component i j a b re im`): seven columns, a
/// component known for the data kind, integer indices, finite values.
fn parse_row(s: &str, kind: &str, data: &RefData, ln: usize) -> Result<RefRow, RefError> {
    let f: Vec<&str> = s.split_whitespace().collect();
    if f.len() != 7 {
        return Err(data.err(ln, &format!("expected 7 columns, got {}", f.len())));
    }
    let comp = f[0].to_string();
    if !known_component(kind, &comp) {
        return Err(data.err(ln, &format!("unknown component `{comp}`")));
    }
    let parse_idx = |t: &str| -> Result<i64, RefError> {
        t.parse::<i64>()
            .map_err(|_| data.err(ln, "bad integer index"))
    };
    let i = parse_idx(f[1])?;
    let j = parse_idx(f[2])?;
    let a = parse_idx(f[3])?;
    let b = parse_idx(f[4])?;
    let re: f64 = f[5].parse().map_err(|_| data.err(ln, "bad number"))?;
    let im: f64 = f[6].parse().map_err(|_| data.err(ln, "bad number"))?;
    if !re.is_finite() || !im.is_finite() {
        return Err(data.err(ln, "non-finite value"));
    }
    Ok(RefRow {
        component: comp,
        i,
        j,
        a,
        b,
        value: Complex::new(re, im),
    })
}

// ------------------------------------------------------------------
// completeness of the independent storage domain
// ------------------------------------------------------------------

/// The completeness check of the data kind (no check for the other
/// kinds).
fn check_completeness(data: &RefData, kind: &str) -> Result<(), RefError> {
    match kind {
        "herm_matrix" => check_complete_herm_matrix(data),
        "pseudo" => check_complete_pseudo(data),
        "fourier" => check_complete_fourier(data),
        "moving" => check_complete_moving(data),
        "selected_green" => check_complete_selected(data),
        "series" => check_complete_series(data),
        "iteration" => check_complete_iteration(data),
        _ => Ok(()),
    }
}

/// `herm_matrix`: every Matsubara point, triangle cell, and mixed
/// point carries its `size1 * size2` block.
fn check_complete_herm_matrix(data: &RefData) -> Result<(), RefError> {
    let (nt, ntau) = (data.int_meta("nt")?, data.int_meta("ntau")?);
    let s = data.int_meta("size1")? * data.int_meta("size2")?;
    let ntri = (nt + 1) * (nt + 2) / 2;
    let nexp = s * ((ntau + 1) + 2 * ntri + (nt + 1) * (ntau + 1));
    if data.rows.len() as i64 != nexp {
        return Err(data.err(
            0,
            &format!("incomplete data: {} rows, expected {nexp}", data.rows.len()),
        ));
    }
    Ok(())
}

/// `pseudo`: every stored component (mat/gtr/tv/les) is complete once
/// any of them stores data.
fn check_complete_pseudo(data: &RefData) -> Result<(), RefError> {
    let (nt, ntau) = (data.int_meta("nt")?, data.int_meta("ntau")?);
    let s = data.int_meta("size1")? * data.int_meta("size2")?;
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
    Ok(())
}

/// `fourier`: the FFT inputs, forwards, and backwards hold the
/// columns `0..N` for every size `N`.
fn check_complete_fourier(data: &RefData) -> Result<(), RefError> {
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
    Ok(())
}

/// `moving`: both triangles of the window with all orbital indices.
fn check_complete_moving(data: &RefData) -> Result<(), RefError> {
    let (tc, s) = (data.int_meta("tc")?, data.int_meta("size")?);
    let nexp = 2 * (tc + 1).pow(2) * s.pow(2);
    if data.rows.len() as i64 != nexp {
        return Err(data.err(0, "incomplete moving window"));
    }
    Ok(())
}

/// `selected_green`: exactly the four fixed keys.
fn check_complete_selected(data: &RefData) -> Result<(), RefError> {
    if data.rows.len() != REF_SELECTED_GREEN_KEYS.len() {
        return Err(data.err(0, "incomplete selected_green data"));
    }
    Ok(())
}

/// `series`: the stored component set equals the metadata list.
fn check_complete_series(data: &RefData) -> Result<(), RefError> {
    let want: BTreeSet<&str> = data.metadata["components"].split(',').collect();
    let got: BTreeSet<&str> = data.rows.iter().map(|r| r.component.as_str()).collect();
    if want != got {
        return Err(data.err(0, "the component set does not match the metadata"));
    }
    Ok(())
}

/// `iteration`: every Matsubara point of every iteration of `g`, and
/// (cg only) every iteration of the five scalars.
fn check_complete_iteration(data: &RefData) -> Result<(), RefError> {
    let (niter, ntau) = (data.int_meta("niter")?, data.int_meta("ntau")?);
    let s = data.int_meta("size1")? * data.int_meta("size2")?;
    let cg = data.metadata["algorithm"] == "cg";
    let nexp = (if cg { niter } else { niter + 1 }) * (ntau + 1) * s;
    let gcount = data.rows.iter().filter(|r| r.component == "g").count() as i64;
    if gcount != nexp {
        return Err(data.err(0, "incomplete iteration data `g`"));
    }
    if cg {
        for c in REF_CG_SCALARS {
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
    Ok(())
}
