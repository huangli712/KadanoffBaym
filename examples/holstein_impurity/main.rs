//
// Usage:
//   cargo run --example holstein_impurity -- migdal|umig <config.toml>
//     [--profile quick|release] [--validate-only]
//
// Holstein impurity with a single bath (Migdal / uMig; NESSi
// examples/programs/holstein_impurity_*.cpp).
//

#[path = "../common/mod.rs"]
mod common;

#[path = "../impl/holstein_energy.rs"]
mod holstein_energy;

#[path = "../impl/holstein_impl.rs"]
mod holstein_impl;

#[path = "../impl/holstein_impurity_impl.rs"]
mod holstein_impurity_impl;

fn main() {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let code = match args.first().map(String::as_str) {
        Some("migdal") => {
            args.remove(0);
            "Migdal"
        }
        Some("umig") => {
            args.remove(0);
            "uMig"
        }
        _ => {
            eprintln!("usage: holstein_impurity <migdal|umig> <config.toml> [flags]");
            std::process::exit(1);
        }
    };
    if let Err(e) = holstein_impurity_impl::holstein_impurity_main(&args, code) {
        eprintln!("{e}");
        std::process::exit(1);
    }
}
