//
// Usage:
//   cargo run --example hubbard_bethe -- ness2|prop|trunc|ness2_trunc <config.toml>
//     (trunc also accepts --start-only and --checkpoint)
//
// Hubbard model on the Bethe lattice: NESS2 steady state and its
// propagations (NESSi examples/programs/ness2_bethe*.cpp,
// hubbard_bethe_trunc.cpp).
//

#[path = "../common/mod.rs"]
mod common;
#[path = "../impl/ness_util.rs"]
mod ness_util;
#[path = "../impl/trunc_model.rs"]
mod trunc_model;
mod ness2;
mod ness2_prop;
mod trunc;
mod ness2_trunc;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (sel, rest) = match args.split_first() {
        Some((s, r)) => (s.as_str(), r.to_vec()),
        None => {
            eprintln!("usage: hubbard_bethe <ness2|prop|trunc|ness2_trunc> <config.toml> [flags]");
            std::process::exit(1);
        }
    };
    let cfg = rest.first().cloned().unwrap_or_default();
    let r: Result<(), String> = match sel {
        "ness2" => ness2::ness2_main(&cfg).map(|_| ()),
        "prop" => ness2_prop::prop_main(&cfg).map(|_| ()),
        "trunc" => {
            let start_only = rest.iter().any(|a| a == "--start-only");
            let checkpoint = rest.iter().any(|a| a == "--checkpoint");
            trunc::trunc_main(&cfg, start_only, checkpoint).map(|_| ())
        }
        "ness2_trunc" => ness2_trunc::ness2_trunc_main(&cfg).map(|_| ()),
        _ => {
            eprintln!("usage: hubbard_bethe <ness2|prop|trunc|ness2_trunc> <config.toml> [flags]");
            std::process::exit(1);
        }
    };
    if let Err(e) = r {
        eprintln!("{e}");
        std::process::exit(1);
    }
}
