// Project : KadanoffBaym
// File    : examples/holstein/main.rs
// Purpose : command-line entry of the Holstein examples.
//
// Usage:
//   cargo run --example holstein -- bethe_migdal|bethe_umig|bethe_nambu_migdal|bethe_nambu_umig|impurity_migdal|impurity_umig <config.toml> [--profile quick|release] [--validate-only]
//

use std::env::args;
use std::process::exit;
mod bethe;
mod energy;
mod impurity;
mod model;
mod nambu;
#[path = "../../util/mod.rs"]
mod util;

fn main() {
    let mut args: Vec<String> = args().skip(1).collect();
    let r = match args.first().map(String::as_str) {
        Some("bethe_migdal") => {
            args.remove(0);
            bethe::run_bethe_case(&args, "Migdal")
        }
        Some("bethe_umig") => {
            args.remove(0);
            bethe::run_bethe_case(&args, "uMig")
        }
        Some("bethe_nambu_migdal") => {
            args.remove(0);
            bethe::run_nambu_case(&args, "Migdal")
        }
        Some("bethe_nambu_umig") => {
            args.remove(0);
            bethe::run_nambu_case(&args, "uMig")
        }
        Some("impurity_migdal") => {
            args.remove(0);
            impurity::run(&args, "Migdal")
        }
        Some("impurity_umig") => {
            args.remove(0);
            impurity::run(&args, "uMig")
        }
        _ => {
            eprintln!(
                "usage: holstein <bethe_migdal|bethe_umig|bethe_nambu_migdal|bethe_nambu_umig|impurity_migdal|impurity_umig> <config.toml> [flags]"
            );
            exit(1);
        }
    };
    if let Err(e) = r {
        eprintln!("{e}");
        exit(1);
    }
}
