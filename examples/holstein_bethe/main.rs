//
// Usage:
//   cargo run --example holstein_bethe -- migdal|umig|nambu_migdal|nambu_umig <config.toml>
//     [--profile quick|release] [--validate-only]
//
// Holstein model on the Bethe lattice (Migdal / uMig, normal or Nambu
// form; NESSi examples/programs/holstein_bethe_*.cpp).
//

#[path = "../common/mod.rs"]
mod common;

#[path = "../impl/holstein_bethe_impl.rs"]
mod holstein_bethe_impl;

#[path = "../impl/holstein_energy.rs"]
mod holstein_energy;

#[path = "../impl/holstein_impl.rs"]
mod holstein_impl;

#[path = "../impl/holstein_nambu_impl.rs"]
mod holstein_nambu_impl;

fn main() {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let run = match args.first().map(String::as_str) {
        Some("migdal") => {
            args.remove(0);
            holstein_bethe_impl::holstein_bethe_main(&args, "Migdal")
        }
        Some("umig") => {
            args.remove(0);
            holstein_bethe_impl::holstein_bethe_main(&args, "uMig")
        }
        Some("nambu_migdal") => {
            args.remove(0);
            holstein_bethe_impl::holstein_nambu_main(&args, "Migdal")
        }
        Some("nambu_umig") => {
            args.remove(0);
            holstein_bethe_impl::holstein_nambu_main(&args, "uMig")
        }
        _ => {
            eprintln!("usage: holstein_bethe <migdal|umig|nambu_migdal|nambu_umig> <config.toml> [flags]");
            std::process::exit(1);
        }
    };
    if let Err(e) = run {
        eprintln!("{e}");
        std::process::exit(1);
    }
}
