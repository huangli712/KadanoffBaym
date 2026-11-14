//
// Usage:
//   cargo run --example holstein_bethe_nambu_umig -- julia/tutor/holstein_bethe/holstein_bethe_nambu_umig.toml [--profile quick|release] [--validate-only]
//
// Holstein model on the Bethe lattice (uMig, Nambu form).
//

#[path = "common/mod.rs"]
mod common;
#[path = "impl/holstein_impl.rs"]
mod holstein_impl;
#[path = "impl/holstein_nambu_impl.rs"]
mod holstein_nambu_impl;
#[path = "impl/holstein_energy.rs"]
mod holstein_energy;
#[path = "impl/holstein_bethe_impl.rs"]
mod holstein_bethe_impl;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let r = holstein_bethe_impl::holstein_nambu_main(&args, "uMig");
    if let Err(e) = r {
        eprintln!("{e}");
        std::process::exit(1);
    }
}
