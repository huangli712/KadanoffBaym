//
// Usage:
//   cargo run --example hubbard -- chain_2b|chain_gw|chain_tpp <config.toml>
//     [--profile quick|release] [--validate-only]
//   cargo run --example hubbard -- bethe_steady|bethe_prop|bethe_trunc|
//     bethe_steady_trunc <config.toml> [--start-only] [--checkpoint]
//
// The Hubbard model programs (NESSi examples/programs/hubbard_chain_*.cpp,
// ness2_bethe*.cpp, trunc_bethe*.cpp): the one-dimensional chain in the
// second Born / GW / particle-particle T-matrix approximations, and the
// Bethe lattice steady state with its propagations.
//

mod hubbard_chain_impl;
mod steady;
mod steady_prop;
mod steady_trunc;
mod trunc;
mod trunc_model;
#[path = "../../util/mod.rs"]
mod util;

fn usage() -> ! {
    eprintln!(
        "usage: hubbard <chain_2b|chain_gw|chain_tpp|bethe_steady|bethe_prop|bethe_trunc|bethe_steady_trunc> <config.toml> [flags]"
    );
    std::process::exit(1)
}

fn main() {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let r: Result<(), String> = match args.first().map(String::as_str) {
        Some("chain_2b") => {
            args.remove(0);
            hubbard_chain_impl::hubbard_main(&args, "2B").map(|_| ())
        }
        Some("chain_gw") => {
            args.remove(0);
            hubbard_chain_impl::hubbard_main(&args, "GW").map(|_| ())
        }
        Some("chain_tpp") => {
            args.remove(0);
            hubbard_chain_impl::hubbard_main(&args, "TPP").map(|_| ())
        }
        Some("bethe_steady") => {
            args.remove(0);
            steady::steady_main(&args.first().cloned().unwrap_or_default()).map(|_| ())
        }
        Some("bethe_prop") => {
            args.remove(0);
            steady_prop::prop_main(&args.first().cloned().unwrap_or_default()).map(|_| ())
        }
        Some("bethe_trunc") => {
            args.remove(0);
            let start_only = args.iter().any(|a| a == "--start-only");
            let checkpoint = args.iter().any(|a| a == "--checkpoint");
            trunc::trunc_main(
                &args.first().cloned().unwrap_or_default(),
                start_only,
                checkpoint,
            )
            .map(|_| ())
        }
        Some("bethe_steady_trunc") => {
            args.remove(0);
            steady_trunc::steady_trunc_main(&args.first().cloned().unwrap_or_default()).map(|_| ())
        }
        _ => usage(),
    };
    if let Err(e) = r {
        eprintln!("{e}");
        std::process::exit(1);
    }
}
