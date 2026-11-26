// Project : KadanoffBaym
// File    : util/cases.rs
// Purpose : the runs of the example programs shared by the regression
//           (examples/validate.rs) and the timing driver (examples/bench.rs)
//

use std::path::Path;

use std::fs::read_to_string;

// ------------------------------------------------------------------
// types
// ------------------------------------------------------------------

/// One run of an example program.
#[derive(Clone, Copy, Debug)]
pub struct ExampleRun {
    /// The example binary.
    pub exe: &'static str,
    /// The model selector of a family binary; empty for the others.
    pub selector: &'static str,
    /// The configuration file, relative to the workspace root.
    pub toml: &'static str,
    /// The trailing command line arguments.
    pub args: &'static [&'static str],
}

// ------------------------------------------------------------------
// constants
// ------------------------------------------------------------------

const QUICK: [&str; 2] = ["--profile", "quick"];

/// The timed runs at the quick profile (the timing driver; the
/// regression checks them as well).
pub const TIMED_RUNS: [ExampleRun; 10] = [
    ExampleRun {
        exe: "hubbard",
        selector: "chain_2b",
        toml: "examples/hubbard/chain.toml",
        args: &QUICK,
    },
    ExampleRun {
        exe: "hubbard",
        selector: "chain_gw",
        toml: "examples/hubbard/chain.toml",
        args: &QUICK,
    },
    ExampleRun {
        exe: "hubbard",
        selector: "chain_tpp",
        toml: "examples/hubbard/chain.toml",
        args: &QUICK,
    },
    ExampleRun {
        exe: "falicov_kimball",
        selector: "",
        toml: "examples/falicov_kimball/falicov_kimball.toml",
        args: &QUICK,
    },
    ExampleRun {
        exe: "holstein",
        selector: "bethe_migdal",
        toml: "examples/holstein/bethe_migdal.toml",
        args: &QUICK,
    },
    ExampleRun {
        exe: "holstein",
        selector: "bethe_umig",
        toml: "examples/holstein/bethe_umig.toml",
        args: &QUICK,
    },
    ExampleRun {
        exe: "holstein",
        selector: "bethe_nambu_migdal",
        toml: "examples/holstein/bethe_nambu_migdal.toml",
        args: &QUICK,
    },
    ExampleRun {
        exe: "holstein",
        selector: "bethe_nambu_umig",
        toml: "examples/holstein/bethe_nambu_umig.toml",
        args: &QUICK,
    },
    ExampleRun {
        exe: "holstein",
        selector: "impurity_migdal",
        toml: "examples/holstein/impurity_migdal.toml",
        args: &QUICK,
    },
    ExampleRun {
        exe: "holstein",
        selector: "impurity_umig",
        toml: "examples/holstein/impurity_umig.toml",
        args: &QUICK,
    },
];

/// The further runs of the regression (no profile argument).
pub const UNTIMED_RUNS: [ExampleRun; 6] = [
    ExampleRun {
        exe: "hubbard",
        selector: "bethe_steady",
        toml: "examples/hubbard/bethe_steady.toml",
        args: &[],
    },
    ExampleRun {
        exe: "hubbard",
        selector: "bethe_prop",
        toml: "examples/hubbard/bethe_steady_propagation.toml",
        args: &[],
    },
    ExampleRun {
        exe: "hubbard",
        selector: "bethe_trunc",
        toml: "examples/hubbard/bethe_trunc.toml",
        args: &[],
    },
    ExampleRun {
        exe: "hubbard",
        selector: "bethe_steady_trunc",
        toml: "examples/hubbard/bethe_steady_truncated.toml",
        args: &[],
    },
    ExampleRun {
        exe: "gw",
        selector: "",
        toml: "examples/gw/gw.toml",
        args: &[],
    },
    ExampleRun {
        exe: "anderson",
        selector: "",
        toml: "examples/anderson/anderson.toml",
        args: &[],
    },
];

// ------------------------------------------------------------------
// functions
// ------------------------------------------------------------------

/// The run of the binary `exe` with the model selector `selector` (the
/// empty string for the single-model programs).
pub fn find_run(exe: &str, selector: &str) -> &'static ExampleRun {
    TIMED_RUNS
        .iter()
        .chain(UNTIMED_RUNS.iter())
        .find(|r| r.exe == exe && r.selector == selector)
        .unwrap_or_else(|| panic!("no example run of `{exe}` with the selector `{selector}`"))
}

/// The text of the configuration `path` with the output directory
/// `dir = "."` redirected to `dir`.
pub fn redirected_config(path: &Path, dir: &Path) -> Result<String, String> {
    let text = read_to_string(path).map_err(|e| e.to_string())?;
    let redirected = text.replace(
        "dir = \".\"",
        &format!("dir = {:?}", dir.display().to_string()),
    );
    Ok(redirected)
}
