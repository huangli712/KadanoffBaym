//
// Usage:
//   cargo run --example holstein -- bethe_migdal|bethe_umig|bethe_nambu_migdal|
//     bethe_nambu_umig|impurity_migdal|impurity_umig <config.toml>
//     [--profile quick|release] [--validate-only]
//
// Holstein model programs (NESSi examples/programs/holstein_bethe_*.cpp,
// holstein_impurity_*.cpp): Migdal / uMig on the Bethe lattice (normal
// and Nambu form) and the single-bath impurity.
//

use std::env::args;
use std::process::exit;
mod holstein_bethe_impl;
mod holstein_energy;
mod holstein_impl;
mod holstein_impurity_impl;
mod holstein_nambu_impl;
#[path = "../../util/mod.rs"]
mod util;

fn main() {
    let mut args: Vec<String> = args().skip(1).collect();
    let r = match args.first().map(String::as_str) {
        Some("bethe_migdal") => {
            args.remove(0);
            holstein_bethe_impl::holstein_bethe_main(&args, "Migdal")
        }
        Some("bethe_umig") => {
            args.remove(0);
            holstein_bethe_impl::holstein_bethe_main(&args, "uMig")
        }
        Some("bethe_nambu_migdal") => {
            args.remove(0);
            holstein_bethe_impl::holstein_nambu_main(&args, "Migdal")
        }
        Some("bethe_nambu_umig") => {
            args.remove(0);
            holstein_bethe_impl::holstein_nambu_main(&args, "uMig")
        }
        Some("impurity_migdal") => {
            args.remove(0);
            holstein_impurity_impl::holstein_impurity_main(&args, "Migdal")
        }
        Some("impurity_umig") => {
            args.remove(0);
            holstein_impurity_impl::holstein_impurity_main(&args, "uMig")
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
