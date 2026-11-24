// Project : KadanoffBaym
// File    : examples/hubbard/main.rs
// Purpose : command-line entry of the Hubbard examples.
//
// Usage:
//   cargo run --example hubbard -- chain_2b|chain_gw|chain_tpp <config.toml> [--profile quick|release] [--validate-only]
//   cargo run --example hubbard -- bethe_steady|bethe_prop|bethe_trunc|bethe_steady_trunc <config.toml> [--start-only] [--checkpoint]
//

use std::env::args;
use std::process::exit;

#[path = "../../util/mod.rs"]
mod util;

mod bethe_trunc;
mod chain;
mod steady_propagation;
mod steady_state;
mod steady_truncated;
mod trunc;

fn usage() -> ! {
    eprintln!(
        "usage: hubbard <chain_2b|chain_gw|chain_tpp|bethe_steady|bethe_prop|bethe_trunc|bethe_steady_trunc> <config.toml> [flags]"
    );
    exit(1)
}

fn main() {
    let mut args: Vec<String> = args().skip(1).collect();
    let r: Result<(), String> = match args.first().map(String::as_str) {
        Some("chain_2b") => {
            args.remove(0);
            chain::run(&args, "2B")
        }
        Some("chain_gw") => {
            args.remove(0);
            chain::run(&args, "GW")
        }
        Some("chain_tpp") => {
            args.remove(0);
            chain::run(&args, "TPP")
        }
        Some("bethe_steady") => {
            args.remove(0);
            steady_state::run_steady_state(&args.first().cloned().unwrap_or_default()).map(|_| ())
        }
        Some("bethe_prop") => {
            args.remove(0);
            steady_propagation::run_propagation(&args.first().cloned().unwrap_or_default())
        }
        Some("bethe_trunc") => {
            args.remove(0);
            let start_only = args.iter().any(|a| a == "--start-only");
            let checkpoint = args.iter().any(|a| a == "--checkpoint");
            trunc::run_truncated(
                &args.first().cloned().unwrap_or_default(),
                start_only,
                checkpoint,
            )
        }
        Some("bethe_steady_trunc") => {
            args.remove(0);
            steady_truncated::run_steady_truncated(&args.first().cloned().unwrap_or_default())
        }
        _ => usage(),
    };
    if let Err(e) = r {
        eprintln!("{e}");
        exit(1);
    }
}
