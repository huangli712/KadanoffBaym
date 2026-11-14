//
// Usage:
//   cargo run --example fkm_bethe -- julia/tutor/fkm_bethe/fkm_bethe.toml [--profile quick|release] [--validate-only]
//
// Interaction quench of the Falicov-Kimball model on the Bethe lattice
// (NESSi examples/programs/fkm_bethe_quench.cpp).
//

#[path = "common/mod.rs"]
mod common;
#[path = "impl/fkm_bethe_impl.rs"]
mod fkm_bethe_impl;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if let Err(e) = fkm_bethe_impl::fkm_main(&args) {
        eprintln!("{e}");
        std::process::exit(1);
    }
}
