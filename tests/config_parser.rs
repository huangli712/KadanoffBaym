//
// Port of julia/tutor/common/validate.jl (the common-layer self
// validation) plus the all-TOML fixture pass of rust-10 W1-T2.
//

#[path = "../examples/common/mod.rs"]
mod common;

use common::config::{
    contour_params, load_example_config, load_tvector, parse_example_args, solver_params,
};
use common::contour_loop::{iterate_bootstrap, iterate_matsubara, propagate_timesteps};
use common::kbref::check_order_gate;
use common::output::{fmt_17g, write_green_components, write_table, GreenRow};
use common::toml::parse_toml_file;
use kadanoffbaym::complex::c64::C64;
use kadanoffbaym::core::layout::MATSUBARA;

const P5EX_BASE: &str = r#"
[contour]
nt = 6
ntau = 20
h = 0.05
beta = 4.0
solve_order = 3

[solver]
mats_max_iter = 10
mats_max_error = 1e-8
bootstrap_max_iter = 10
bootstrap_max_error = 1e-8
corrector_steps = 2

[model]
U = 1.0
Nsites = 2

[output]
dir = "."
prefix = "ex"

[profile.quick]
contour = { nt = 4 }

[profile.release]
"#;

fn tmpdir(tag: &str) -> String {
    let d = std::env::temp_dir().join(format!("kb-r9-{}-{}-{}", tag, std::process::id(), std::time::SystemTime::now().elapsed().unwrap().subsec_nanos()));
    std::fs::create_dir_all(&d).unwrap();
    d.to_string_lossy().to_string()
}

fn write_file(dir: &str, name: &str, txt: &str) -> String {
    let p = format!("{dir}/{name}");
    std::fs::write(&p, txt).unwrap();
    p
}

fn load(p: &str, profile: &str) -> Result<common::config::ExampleConfig, String> {
    load_example_config(p, profile, &["U", "Nsites"], &[], &[], &[])
}

#[test]
fn config_schema_and_contour_mapping() {
    let dir = tmpdir("schema");
    let p = write_file(&dir, "ex.toml", P5EX_BASE);
    let cq = load(&p, "quick").unwrap();
    let cr = load(&p, "release").unwrap();
    let c = contour_params(&cq).unwrap();
    assert_eq!(cq.profile, "quick");
    assert_eq!(c.nt, 4);
    assert_eq!(c.ntau, 20);
    assert_eq!(c.k, 3);
    assert_eq!(c.h, 0.05);
    assert_eq!(c.beta, 4.0);
    assert_eq!(contour_params(&cr).unwrap().nt, 6);
    let s = solver_params(&cq).unwrap();
    assert_eq!(s.output_every, 1);
    assert_eq!(s.corrector_steps, 2);
    assert_eq!(
        cq.output["save_green_components"].as_bool(),
        Some(false)
    );
    // bad configurations
    let bad = |txt: &str| load(&write_file(&dir, "bad.toml", txt), "quick").is_err();
    assert!(bad(&P5EX_BASE.replace("solve_order = 3", "solve_order = 6")));
    assert!(bad(&P5EX_BASE.replace("solve_order = 3", "solve_order = 0")));
    assert!(bad(&P5EX_BASE.replace("U = 1.0", "U = 1.0\nV = 2.0")));
    assert!(bad(&P5EX_BASE.replace("U = 1.0\n", "")));
    assert!(bad(&P5EX_BASE.replace("U = 1.0", "U = nan")));
    assert!(bad(&P5EX_BASE.replace("h = 0.05", "h = inf")));
    assert!(bad(&format!("{P5EX_BASE}\n[extra]\nx = 1\n")));
    assert!(bad(&P5EX_BASE.replace("[profile.release]\n", "")));
    assert!(bad(&P5EX_BASE.replace("contour = { nt = 4 }", "contour = { nq = 4 }")));
    // nt < solve_order through the quick override
    assert!(bad(&P5EX_BASE.replace("contour = { nt = 4 }", "contour = { nt = 2 }")));
    assert!(bad(&P5EX_BASE.replace("prefix = \"ex\"", "prefix = \"ex.h5\"")));
    assert!(bad(&P5EX_BASE.replace("ntau = 20", "ntau = 20.5")));
    assert!(load(&p, "fast").is_err());
}

#[test]
fn config_time_dependent_parameters() {
    let dir = tmpdir("drive");
    let txt = format!(
        "{P5EX_BASE}\n[drive.J]\nmode = \"constant\"\nvalue = 0.5\n\n[drive.g]\nmode = \"file\"\npath = \"g.dat\"\n"
    );
    let p = write_file(&dir, "drive.toml", &txt);
    // nt = 4 (quick): nt + 2 = 6 values, the first is NESSi tstp = -1
    write_file(&dir, "g.dat", "# comment\n1.0 2.0\n3.0\n4.0 5.0 6.0\n");
    let cfg = load_example_config(&p, "quick", &["U", "Nsites"], &[], &["J", "g"], &["J", "g"]).unwrap();
    let j = load_tvector(&cfg, "J").unwrap();
    for i in MATSUBARA..=4 {
        assert_eq!(j.block(i)[0], C64::new(0.5, 0.0));
    }
    let g = load_tvector(&cfg, "g").unwrap();
    assert_eq!(g.block(MATSUBARA)[0], C64::new(1.0, 0.0));
    for i in 0..=4i64 {
        assert_eq!(g.block(i)[0], C64::new((i + 2) as f64, 0.0));
    }
    // one value short
    write_file(&dir, "g.dat", "1 2 3 4 5\n");
    assert!(load_tvector(&cfg, "g").is_err());
    // relative paths resolve from the TOML directory
    let sub = format!("{dir}/sub");
    std::fs::create_dir_all(&sub).unwrap();
    write_file(&sub, "g.dat", "1 2 3 4 5 6\n");
    let p2 = write_file(&sub, "drive.toml", &txt);
    let cfg2 = load_example_config(&p2, "quick", &["U", "Nsites"], &[], &["J", "g"], &["J", "g"]).unwrap();
    let g2 = load_tvector(&cfg2, "g").unwrap();
    assert_eq!(g2.block(4)[0], C64::new(6.0, 0.0));
    // unknown drive and bad mode
    assert!(load_example_config(&p, "quick", &["U", "Nsites"], &[], &["J"], &["J"]).is_err());
    assert!(load_example_config(
        &write_file(&dir, "bm.toml", &txt.replace("mode = \"constant\"", "mode = \"ramp\"")),
        "quick",
        &["U", "Nsites"],
        &[],
        &["J", "g"],
        &["J", "g"]
    )
    .is_err());
    // the sin2 pulse
    let txt2 = format!(
        "{P5EX_BASE}\n[drive.p]\nmode = \"sin2\"\namplitude = 2.0\nend = 0.1\n"
    );
    let p3 = write_file(&dir, "sin2.toml", &txt2);
    let cfg3 = load_example_config(&p3, "quick", &["U", "Nsites"], &[], &["p"], &["p"]).unwrap();
    let v = load_tvector(&cfg3, "p").unwrap();
    assert_eq!(v.block(MATSUBARA)[0], C64::ZERO);
    assert_eq!(v.block(0)[0], C64::ZERO); // t = 0
    // h = 0.05, tend = 0.1: only t = 0.05 is inside the pulse
    let want = 2.0 * (0.05 * std::f64::consts::PI / 0.1).sin() * (0.05 * std::f64::consts::PI / 0.1).sin();
    assert_eq!(v.block(1)[0], C64::new(want, 0.0));
}

#[test]
fn config_deterministic_text_output() {
    let dir = tmpdir("output");
    let a = format!("{dir}/a.dat");
    let b = format!("{dir}/b.dat");
    let rows = vec![
        vec![0.0, 0.1, 1.0 / 3.0],
        vec![0.05, -2.5e-17, std::f64::consts::PI],
    ];
    write_table(&a, &["t", "x", "y"], &rows).unwrap();
    write_table(&b, &["t", "x", "y"], &rows).unwrap();
    assert_eq!(std::fs::read(&a).unwrap(), std::fs::read(&b).unwrap());
    let lines: Vec<String> = std::fs::read_to_string(&a)
        .unwrap()
        .lines()
        .map(|s| s.to_string())
        .collect();
    assert_eq!(lines[0], "# 1:t 2:x 3:y");
    let parsed: Vec<f64> = lines[2].split_whitespace().map(|w| w.parse().unwrap()).collect();
    assert_eq!(parsed, rows[1]);
    // a failing row leaves no partial file
    let c = format!("{dir}/c.dat");
    assert!(write_table(&c, &["x", "y"], &[vec![1.0, 2.0], vec![f64::NAN, 0.0]]).is_err());
    assert!(!std::path::Path::new(&c).exists());
    assert!(write_table(&c, &["x"], &[vec![f64::NAN]]).is_err());
    assert!(!std::path::Path::new(&c).exists());
    // no temporary leftovers
    let names: Vec<String> = std::fs::read_dir(&dir)
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().to_string())
        .collect();
    assert!(!names.iter().any(|f| f.starts_with(".tmp_")));
    // Green's function components
    let g = format!("{dir}/g_green.dat");
    write_green_components(
        &g,
        &[
            GreenRow { component: "mat", i: 0, j: -1, a: 0, b: 0, value: C64::new(0.5, -0.25) },
            GreenRow { component: "les", i: 1, j: 3, a: 0, b: 0, value: C64::new(1e-20, 2.0) },
        ],
    )
    .unwrap();
    let gt = std::fs::read_to_string(&g).unwrap();
    let gl: Vec<&str> = gt.lines().collect();
    assert_eq!(gl[0], "# component i j a b re im");
    let f1: Vec<&str> = gl[1].split_whitespace().collect();
    assert_eq!(&f1[..5], &["mat", "0", "-1", "0", "0"]);
    let f2: Vec<&str> = gl[2].split_whitespace().collect();
    assert_eq!(f2[6].parse::<f64>().unwrap(), 2.0);
    // the %.17g emulation
    assert_eq!(fmt_17g(0.0), "0");
    assert_eq!(fmt_17g(1.0 / 3.0), "0.33333333333333331");
    assert_eq!(fmt_17g(100.0), "100");
    assert_eq!(fmt_17g(2.5e-17), "2.4999999999999999e-17");
    assert_eq!(fmt_17g(0.05), "0.050000000000000003");
    assert_eq!(fmt_17g(std::f64::consts::PI), "3.1415926535897931");
}

#[test]
fn config_three_stage_loop() {
    // Matsubara: converged after the error drops below tol
    let errs = [1.0, 0.1, 1e-9, 1e-12];
    let r = iterate_matsubara(|i| errs[i as usize], 3, 1e-8, |_| {}).unwrap();
    assert_eq!(r.0, 3);
    assert!(iterate_matsubara(|_| 1.0, 2, 1e-8, |_| {}).is_err());
    // min_iterations = 0 exits after the first iteration
    let mut n = 0;
    let r = iterate_bootstrap(
        |_| {
            n += 1;
            0.0
        },
        5,
        1e-8,
        0,
        |_| {},
    )
    .unwrap();
    assert_eq!(n, 1);
    assert_eq!(r.0, 1);
    // min_iterations = 3 does not exit earlier
    n = 0;
    let r = iterate_bootstrap(
        |_| {
            n += 1;
            0.0
        },
        5,
        1e-8,
        3,
        |_| {},
    )
    .unwrap();
    assert_eq!(n, 4);
    assert_eq!(r.0, 4);
    assert!(iterate_bootstrap(|_| 1.0, 2, 1e-8, 0, |_| {}).is_err());
    // propagation: predictor once, then the correctors
    let seq = std::cell::RefCell::new(Vec::new());
    propagate_timesteps(
        |n| seq.borrow_mut().push(format!("p{n}")),
        |n, i| seq.borrow_mut().push(format!("c{n}i{i}")),
        5,
        6,
        2,
        0.1,
        |_| {},
    );
    assert_eq!(
        *seq.borrow(),
        ["p5", "c5i0", "c5i1", "p6", "c6i0", "c6i1"]
    );
    // command line
    let a = parse_example_args(&[
        "x.toml".to_string(),
        "--profile".to_string(),
        "release".to_string(),
        "--validate-only".to_string(),
    ])
    .unwrap();
    assert_eq!(a.profile, "release");
    assert!(a.validate_only);
    assert_eq!(parse_example_args(&["x.toml".to_string()]).unwrap().profile, "quick");
    assert!(parse_example_args(&["x.toml".to_string(), "--bogus".to_string()]).is_err());
}

#[test]
fn config_order_gate_self_test() {
    let ns = [10, 20, 40, 80];
    let errs4: Vec<f64> = ns.iter().map(|n| 3.0 * (*n as f64).powi(-4)).collect();
    let (ok, observed, _) = check_order_gate(&ns, &errs4, 4.0, 1e-6, 1e-13, 0.3);
    assert!(ok);
    assert!((observed - 4.0).abs() < 1e-10);
    // plateau at the finest grid is tolerated if 3 points remain
    let mut errsp = errs4.clone();
    errsp[3] = 1e-15;
    assert!(check_order_gate(&ns, &errsp, 4.0, 1e-6, 1e-13, 0.3).0);
    // fewer than 4 grids
    assert!(!check_order_gate(&ns[..3], &errs4[..3], 4.0, 1e-6, 1e-13, 0.3).0);
    // fewer than 3 non-plateau points
    let bad = vec![errs4[0], errs4[1], 1e-15, 1e-16];
    assert!(!check_order_gate(&ns, &bad, 4.0, 1e-6, 1e-13, 0.3).0);
    // finest-grid cap exceeded
    assert!(!check_order_gate(&ns, &errs4, 4.0, errs4[3] / 2.0, 1e-13, 0.3).0);
    // non-increasing grids, bad errors
    let ni = [10, 20, 20, 40];
    assert!(!check_order_gate(&ni, &errs4, 4.0, 1e-6, 1e-13, 0.3).0);
    let nanerr = vec![errs4[0], errs4[1], errs4[2], f64::NAN];
    assert!(!check_order_gate(&ns, &nanerr, 4.0, 1e-6, 1e-13, 0.3).0);
    let zeroerr = vec![errs4[0], errs4[1], errs4[2], 0.0];
    assert!(!check_order_gate(&ns, &zeroerr, 4.0, 1e-6, 1e-13, 0.3).0);
    // reduced-order mutants must not pass
    let errs3: Vec<f64> = ns.iter().map(|n| 0.5 * (*n as f64).powi(-3)).collect();
    assert!(!check_order_gate(&ns, &errs3, 4.0, 1.0, 1e-13, 0.3).0);
    let ns5 = [8, 16, 32, 64, 128];
    let errs1: Vec<f64> = ns5
        .iter()
        .map(|n| 0.2 / *n as f64 + 1e-3 / (*n as f64).powi(2))
        .collect();
    assert!(!check_order_gate(&ns5, &errs1, 2.0, 1.0, 1e-13, 0.3).0);
}

/// Every actual tutor TOML parses; the Phase 5 configurations load with
/// both profiles and the quick/release overrides merge correctly.
#[test]
fn config_all_tutor_tomls() {
    let root = concat!(env!("CARGO_MANIFEST_DIR"), "/julia/tutor");
    let mut all = Vec::new();
    for entry in walk(std::path::Path::new(root)) {
        let f = entry.to_string_lossy().to_string();
        if f.ends_with(".toml") {
            all.push(f);
        }
    }
    all.sort();
    assert_eq!(all.len(), 17, "the tutor TOML fixtures");
    // the parser accepts every file
    for f in &all {
        parse_toml_file(f).unwrap_or_else(|e| panic!("{f}: {e}"));
    }
    // the Phase 5 examples load through the config layer (generic model
    // keys taken from the file itself)
    let p5: Vec<String> = all
        .iter()
        .filter(|f| {
            f.contains("hubbard_chain/")
                || f.contains("fkm_bethe/")
                || f.contains("holstein_bethe/")
                || f.contains("holstein_impurity/")
        })
        .cloned()
        .collect();
    assert_eq!(p5.len(), 8);
    for f in &p5 {
        let raw = parse_toml_file(f).unwrap();
        let model_keys: Vec<&str> = raw["model"]
            .as_table()
            .unwrap()
            .keys()
            .map(|k| k.as_str())
            .collect();
        let drive_keys: Vec<&str> = raw
            .get("drive")
            .and_then(|d| d.as_table())
            .map(|t| t.keys().map(|k| k.as_str()).collect())
            .unwrap_or_default();
        for profile in ["quick", "release"] {
            load_example_config(f, profile, &[], &model_keys, &drive_keys, &drive_keys)
                .unwrap_or_else(|e| panic!("{f} ({profile}): {e}"));
        }
    }
    // quick/release merge semantics of hubbard_chain
    let hc = format!("{root}/hubbard_chain/hubbard_chain.toml");
    let cq = load_example_config(&hc, "quick", &[], &["Nsites", "HoppingT", "HubbardU", "MuChem", "RampSite", "RampW0"], &[], &[]).unwrap();
    let cr = load_example_config(&hc, "release", &[], &["Nsites", "HoppingT", "HubbardU", "MuChem", "RampSite", "RampW0"], &[], &[]).unwrap();
    let q = contour_params(&cq).unwrap();
    let r = contour_params(&cr).unwrap();
    assert_eq!((q.nt, q.ntau, q.k), (10, 40, 3));
    assert_eq!((r.nt, r.ntau, r.k), (399, 400, 5));
}

fn walk(dir: &std::path::Path) -> Vec<std::path::PathBuf> {
    let mut out = Vec::new();
    if dir.is_dir() {
        for e in std::fs::read_dir(dir).unwrap() {
            let e = e.unwrap();
            if e.path().is_dir() {
                out.extend(walk(&e.path()));
            } else {
                out.push(e.path());
            }
        }
    }
    out
}
