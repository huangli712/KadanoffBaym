// Project : KadanoffBaym
// File    : util/kbref/check.rs
// Purpose : the index-domain validation of one kbref-v1 row, one
//           validator per data kind
//

use super::error::KbrefError;
use super::read::{KbrefData, KbrefRow};
use super::schema::{
    KBREF_REFERENCE_CASES, KBREF_REFERENCE_KINDS, KBREF_SELECTED_GREEN_KEYS, KBREF_TOOL_KINDS,
    observable_class, series_class,
};

/// Dispatch one parsed row to the validator of its data kind; the
/// reference kinds and the tool kinds share one validator each.
pub(super) fn check_row(
    kind: &str,
    r: &KbrefRow,
    d: &KbrefData,
    ln: usize,
) -> Result<(), KbrefError> {
    match kind {
        "weights" => check_weights(r, d, ln),
        "iteration" => check_iteration(r, d, ln),
        "pseudo" => check_pseudo(r, d, ln),
        k if KBREF_REFERENCE_KINDS.contains(&k) => check_reference(r, d, ln),
        "moving" => check_moving(r, d, ln),
        "series" => check_series(r, d, ln),
        "observable" | "selected_green" => check_observable(r, d, ln),
        k if KBREF_TOOL_KINDS.contains(&k) => check_tool(r, d, ln),
        _ => check_herm_matrix(r, d, ln),
    }
}

// ------------------------------------------------------------------
// validators
// ------------------------------------------------------------------

/// The `weights` kind: the order `k` in 1:5, real values, and
/// the index domain of the five tables.
pub(super) fn check_weights(r: &KbrefRow, d: &KbrefData, ln: usize) -> Result<(), KbrefError> {
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
        k >= 2 && 1 <= r.j && r.j < k && 0 <= r.a && r.a <= k && 0 <= r.b && r.b <= k
    };
    if !ok {
        return e(&format!("index out of range for `{}`", r.component));
    }
    Ok(())
}

/// The `herm_matrix` kind (the fallback of the dispatch): the
/// orbital indices and the index domain of mat/ret/tv/les.
pub(super) fn check_herm_matrix(r: &KbrefRow, d: &KbrefData, ln: usize) -> Result<(), KbrefError> {
    let nt = d.int_meta("nt")?;
    let ntau = d.int_meta("ntau")?;
    let s1 = d.int_meta("size1")?;
    let s2 = d.int_meta("size2")?;
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

/// The tool kinds (`scalar`, `labels`, `function`, `element`): a
/// non-negative case id, the orbital indices, and the time index
/// domain of the component.
pub(super) fn check_tool(r: &KbrefRow, d: &KbrefData, ln: usize) -> Result<(), KbrefError> {
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
        let nt = d.int_meta("nt")?;
        let s1 = d.int_meta("size1")?;
        let s2 = d.int_meta("size2")?;
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

/// The `pseudo` kind: the orbital indices and the index domain of
/// mat/gtr/tv/les/density/les_start_source.
pub(super) fn check_pseudo(r: &KbrefRow, d: &KbrefData, ln: usize) -> Result<(), KbrefError> {
    let nt = d.int_meta("nt")?;
    let ntau = d.int_meta("ntau")?;
    let s1 = d.int_meta("size1")?;
    let s2 = d.int_meta("size2")?;
    let k = d.int_meta("k")?;
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

/// The `iteration` kind: the `algorithm` metadata, the iteration
/// range of `g` (fixpoint counts from 0), and the real cg scalars.
pub(super) fn check_iteration(r: &KbrefRow, d: &KbrefData, ln: usize) -> Result<(), KbrefError> {
    let alg = d.metadata.get("algorithm").cloned().unwrap_or_default();
    if alg != "fixpoint" && alg != "cg" {
        return Err(d.err(ln, "bad metadata `algorithm`"));
    }
    let niter = d.int_meta("niter")?;
    let ntau = d.int_meta("ntau")?;
    let s1 = d.int_meta("size1")?;
    let s2 = d.int_meta("size2")?;
    let imin = if alg == "fixpoint" { 0 } else { 1 };
    if r.component == "g" {
        if !(imin <= r.i
            && r.i <= niter
            && 0 <= r.j
            && r.j <= ntau
            && 0 <= r.a
            && r.a < s1
            && 0 <= r.b
            && r.b < s2)
        {
            return Err(d.err(ln, "index out of range for `g`"));
        }
    } else {
        if alg != "cg" {
            return Err(d.err(
                ln,
                &format!("scalar `{}` is only allowed for cg", r.component),
            ));
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

/// The reference kinds (`phase7_fourier`, `phase7_steady`,
/// `phase7_example`): the index domain of every component of the
/// three sets.
pub(super) fn check_reference(r: &KbrefRow, d: &KbrefData, ln: usize) -> Result<(), KbrefError> {
    let kind = &d.metadata["data_kind"];
    let (c, i, j, a, b, v) = (&r.component, r.i, r.j, r.a, r.b, r.value);
    let (cid, smax) = if kind == "phase7_example" {
        let cid = KBREF_REFERENCE_CASES
            .iter()
            .position(|x| *x == d.metadata["case"])
            .unwrap_or(usize::MAX) as i64;
        (cid, 1i64)
    } else {
        (-1i64, d.int_meta("size_max")?)
    };
    let orb = 0 <= a && a < smax && 0 <= b && b < smax;
    let ok = if kind == "phase7_fourier" {
        if c == "fft_input" || c == "fft_forward" || c == "fft_backward" {
            i >= 1 && 0 <= j && j < i && a == 0 && b == 0
        } else if c == "dftcorr" {
            i >= 0
                && (j == 1 || j == 3)
                && b == 0
                && 0 <= a
                && a <= if j == 1 { 3 } else { 5 }
                && (a >= 2 || v.im == 0.0)
        } else if c == "adaptive" {
            (0..=2).contains(&i)
                && j >= 0
                && (0..=2).contains(&a)
                && b == 0
                && (a != 0 || v.im == 0.0)
        } else if c == "matsubara_ft_result" {
            i >= 0 && (i % 10 == 0 || i % 10 == 1 || i % 10 == 3) && orb
        } else {
            i >= 0 && j >= 0 && orb
        }
    } else if kind == "phase7_steady" {
        let nft = d.int_meta("nft_max")?;
        if c == "ret_time" || c == "les_time" || c == "ret_freq" || c == "les_freq" {
            i >= 0 && 0 <= j && j < nft && orb
        } else if c == "bath_ret" || c == "bath_les" {
            (0..=3).contains(&i) && j >= 0 && a == 0 && b == 0
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
                    || (cid == 2 && j >= 0 && (0..=2).contains(&a) && b >= 0))
        }
    };
    if !ok {
        return Err(d.err(ln, &format!("index out of range for `{c}`")));
    }
    Ok(())
}

/// The `moving` kind: both time indices inside the window and the
/// orbital indices.
pub(super) fn check_moving(r: &KbrefRow, d: &KbrefData, ln: usize) -> Result<(), KbrefError> {
    let tc = d.int_meta("tc")?;
    let s = d.int_meta("size")?;
    if !(0 <= r.i
        && r.i <= tc
        && 0 <= r.j
        && r.j <= tc
        && 0 <= r.a
        && r.a < s
        && 0 <= r.b
        && r.b < s)
    {
        return Err(d.err(ln, &format!("index out of range for `{}`", r.component)));
    }
    Ok(())
}

/// The `series` kind: the component belongs to the case set and is
/// listed in the metadata; the index domain of its class (scalar,
/// `nk`-vector, matrix).
pub(super) fn check_series(r: &KbrefRow, d: &KbrefData, ln: usize) -> Result<(), KbrefError> {
    let cs = d.metadata["case_set"].as_str();
    let cls = series_class(cs, &r.component);
    if cls.is_none() {
        return Err(d.err(
            ln,
            &format!("component `{}` not allowed in this case_set", r.component),
        ));
    }
    if !d.metadata["components"]
        .split(',')
        .any(|x| x == r.component)
    {
        return Err(d.err(
            ln,
            &format!("component `{}` not listed in the metadata", r.component),
        ));
    }
    let nsteps = d.int_meta("nsteps")?;
    let nk = d.int_meta("nk")?;
    let s = d.int_meta("size")?;
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

/// The `observable` kind (the index domain of its class) and the
/// four fixed keys of `selected_green`.
pub(super) fn check_observable(r: &KbrefRow, d: &KbrefData, ln: usize) -> Result<(), KbrefError> {
    if d.metadata["data_kind"] == "selected_green" {
        let is_key = KBREF_SELECTED_GREEN_KEYS.iter().any(|(c, i, j, a, b)| {
            *c == r.component && *i == r.i && *j == r.j && *a == r.a && *b == r.b
        });
        if !is_key {
            return Err(d.err(ln, "bad selected_green key"));
        }
        return Ok(());
    }
    let model = d.metadata["model"].as_str();
    let cls = observable_class(model, &r.component);
    if cls.is_none() {
        return Err(d.err(
            ln,
            &format!("component `{}` not allowed for this model", r.component),
        ));
    }
    let nt = d.int_meta("nt")?;
    let s = d.int_meta("size")?;
    let mut ok = -1 <= r.i && r.i <= nt && r.j == -1;
    if cls == Some(0) {
        ok = ok && r.a == -1 && r.b == -1 && r.value.im == 0.0;
    } else {
        ok = ok && 0 <= r.a && r.a < s && 0 <= r.b && r.b < s;
    }
    if !ok {
        return Err(d.err(
            ln,
            &format!("index or value out of range for `{}`", r.component),
        ));
    }
    Ok(())
}
