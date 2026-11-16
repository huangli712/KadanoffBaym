//
// Usage:
//   cargo run --example hubbard -- chain_2b|chain_gw|chain_tpp <config.toml>
//     [--profile quick|release] [--validate-only]
//   cargo run --example hubbard -- bethe_ness2|bethe_prop|bethe_trunc|
//     bethe_ness2_trunc <config.toml> [--start-only] [--checkpoint]
//
// The Hubbard model programs (NESSi examples/programs/hubbard_chain_*.cpp,
// ness2_bethe*.cpp, trunc_bethe*.cpp): the one-dimensional chain in the
// second Born / GW / particle-particle T-matrix approximations, and the
// Bethe lattice steady state with its propagations.
//

#[path = "../common/mod.rs"]
mod common;
mod hubbard_chain_impl;
mod ness2;
mod ness2_prop;
mod ness2_trunc;
mod trunc;
mod trunc_model;

fn usage() -> ! {
    eprintln!(
        "usage: hubbard <chain_2b|chain_gw|chain_tpp|bethe_ness2|bethe_prop|bethe_trunc|bethe_ness2_trunc> <config.toml> [flags]"
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
        Some("bethe_ness2") => {
            args.remove(0);
            ness2::ness2_main(&args.first().cloned().unwrap_or_default()).map(|_| ())
        }
        Some("bethe_prop") => {
            args.remove(0);
            ness2_prop::prop_main(&args.first().cloned().unwrap_or_default()).map(|_| ())
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
        Some("bethe_ness2_trunc") => {
            args.remove(0);
            ness2_trunc::ness2_trunc_main(&args.first().cloned().unwrap_or_default()).map(|_| ())
        }
        _ => usage(),
    };
    if let Err(e) = r {
        eprintln!("{e}");
        std::process::exit(1);
    }
}
