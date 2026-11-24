// Project : KadanoffBaym
// File    : util/kbref/schema.rs
// Purpose : the kbref-v1 schema tables (data kinds, metadata keys,
//           components) and the index classes of the per-case sets
//

/// Format version.
pub(super) const KBREF_VERSION: &str = "1";

/// Column specification line.
pub(super) const KBREF_COLUMNS: &str = "component i j a b re im";

/// All known data kinds.
pub(super) const KBREF_KINDS: [&str; 15] = [
    "weights",
    "herm_matrix",
    "scalar",
    "function",
    "element",
    "labels",
    "iteration",
    "pseudo",
    "phase7_fourier",
    "phase7_steady",
    "phase7_example",
    "moving",
    "series",
    "observable",
    "selected_green",
];

/// Tool data kinds (require `generator` and `case_set` metadata).
pub(super) const KBREF_TOOL_KINDS: [&str; 5] =
    ["scalar", "function", "element", "labels", "iteration"];

/// Reference data kinds (NESSi comparison sets).
pub(super) const KBREF_REFERENCE_KINDS: [&str; 3] =
    ["phase7_fourier", "phase7_steady", "phase7_example"];

/// The `phase7_example` cases.
pub(super) const KBREF_REFERENCE_CASES: [&str; 3] = ["siam", "bethe", "bethe_prop"];

/// Metadata of the `moving` kind.
pub(super) const KBREF_MOVING_META: [&str; 8] = [
    "generator",
    "case_set",
    "case_id",
    "source",
    "tc",
    "size",
    "sign",
    "tstp",
];

/// Generators allowed for `moving`.
pub(super) const KBREF_MOVING_GENERATORS: [&str; 2] = ["p8_moving_ref", "p8_moving_solver_ref"];

/// Metadata of the `series` kind.
pub(super) const KBREF_SERIES_META: [&str; 7] = [
    "generator",
    "case_set",
    "components",
    "nsteps",
    "nk",
    "size",
    "source",
];

/// Metadata of the `pseudo` kind.
pub(super) const KBREF_PSEUDO_META: [&str; 8] = [
    "generator",
    "case",
    "sig",
    "nt",
    "ntau",
    "size1",
    "size2",
    "k",
];

/// CG iteration scalars.
pub const KBREF_CG_SCALARS: [&str; 5] = ["rsold", "c2", "alpha", "rsnew", "beta_cg"];

/// All `series` case sets.
pub const KBREF_SERIES_CASE_SETS: [&str; 4] =
    ["moving_solver", "trunc_bethe", "steady_bethe_trunc", "gw"];

/// All observable models.
pub const OBSERVABLE_MODELS: [&str; 5] = [
    "hubbard",
    "fkm",
    "holstein_bethe",
    "holstein_nambu",
    "holstein_impurity",
];
/// Metadata of the `observable` kinds.
pub const KBREF_OBSERVABLE_META: [&str; 15] = [
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
const KBREF_HOLSTEIN_SCALARS: [&str; 9] = [
    "Xph", "Pph", "XX", "Ekin", "Enx_MF", "Enx_corr", "Eph_cl", "Eph_corr", "Etot",
];

/// The four fixed keys of `selected_green` (hubbard only).
pub const KBREF_SELECTED_GREEN_KEYS: [(&str, i64, i64, i64, i64); 4] = [
    ("mat", 0, -1, 0, 0),
    ("ret", 3, 1, 0, 0),
    ("les", 1, 3, 0, 0),
    ("tv", 3, 2, 0, 0),
];

// ------------------------------------------------------------------
// lookups
// ------------------------------------------------------------------

/// Allowed components per data kind (None for `series` and `observable`,
/// which use per-case sets).
pub(super) fn components(kind: &str) -> Option<&'static [&'static str]> {
    match kind {
        "weights" => Some(&["greg", "bd", "pint", "pdiff", "rcorr"]),
        "herm_matrix" => Some(&["mat", "ret", "tv", "les"]),
        "scalar" => Some(&[
            "distance_ret",
            "distance_tv",
            "distance_les",
            "distance_total",
            "response",
        ]),
        "function" => Some(&["extrapolate", "interpolate"]),
        "element" => Some(&[
            "les_timediag",
            "density_matrix",
            "startup_less_adv",
            "startup_lmix_rmix",
        ]),
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
        "phase7_steady" => Some(&[
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

/// `observable` kind: index class (0 real scalar, 2 density matrix) of
/// a component for a model.
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
            } else if KBREF_HOLSTEIN_SCALARS.contains(&comp)
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

/// Approximations allowed for an observable model.
pub fn approximations(model: &str) -> Option<&'static [&'static str]> {
    match model {
        "hubbard" => Some(&["2B", "GW", "TPP"]),
        "fkm" => Some(&["none"]),
        "holstein_bethe" | "holstein_nambu" | "holstein_impurity" => Some(&["Migdal", "uMig"]),
        _ => None,
    }
}

/// `series` kind: index class of a component for a case_set.
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
        "steady_bethe_trunc" => match comp {
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
        "steady_bethe_trunc" => Some("p8_steady_bethe_trunc_ref"),
        "gw" => Some("p8_gw_ref"),
        _ => None,
    }
}

/// Components (unsorted) of a `series` case set.
pub fn series_components(cs: &str) -> Option<Vec<&'static str>> {
    let mut out = Vec::new();
    for c in [
        "kinetic",
        "density",
        "error",
        "ekin",
        "afield",
        "current_nessi",
        "eint",
        "dyson_ret",
        "dyson_less",
        "dyson0_ret",
        "dyson0_less",
        "bubble1_ret",
        "bubble1_less",
        "bubble2_ret",
        "bubble2_less",
        "vie2_ret",
        "vie2_less",
        "g_ret",
        "g_less",
        "sigma_ret",
        "sigma_less",
        "rho",
        "sigma_hf",
        "w_ret",
        "w_less",
        "sigma_gw_ret",
        "sigma_gw_less",
        "p_ret",
        "p_less",
    ] {
        if series_class(cs, c).is_some() {
            out.push(c);
        }
    }
    if out.is_empty() { None } else { Some(out) }
}

/// Components (unsorted) of a P5 observable model.
pub fn observable_components(model: &str) -> Option<Vec<&'static str>> {
    let mut out = Vec::new();
    for c in [
        "n1", "Ekin", "Epot", "Etot", "den0", "den1", "total", "rho", "Xph", "Pph", "XX", "Enx_MF",
        "Enx_corr", "Eph_cl", "Eph_corr", "Eph",
    ] {
        if observable_class(model, c).is_some() {
            out.push(c);
        }
    }
    if out.is_empty() { None } else { Some(out) }
}

/// Whether `comp` is a known component of the data kind `kind` (the
/// per-case sets decide for `series` and `observable`).
pub(super) fn known_component(kind: &str, comp: &str) -> bool {
    if kind == "series" {
        KBREF_SERIES_CASE_SETS
            .iter()
            .any(|cs| series_class(cs, comp).is_some())
    } else if kind == "observable" {
        OBSERVABLE_MODELS
            .iter()
            .any(|m| observable_class(m, comp).is_some())
    } else {
        components(kind).map(|c| c.contains(&comp)).unwrap_or(false)
    }
}
