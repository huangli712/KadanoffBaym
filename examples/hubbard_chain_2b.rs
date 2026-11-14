//
// Usage:
//   cargo run --example hubbard_chain_2b -- julia/tutor/hubbard_chain/hubbard_chain.toml [--profile quick|release] [--validate-only]
//
// Hubbard chain in the second Born approximation (NESSi
// examples/programs/hubbard_chain_2b.cpp).
//

#[path = "common/mod.rs"]
mod common;
#[path = "impl/hubbard_chain_impl.rs"]
mod hubbard_chain_impl;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if let Err(e) = hubbard_chain_impl::hubbard_main(&args, "2B") {
        eprintln!("{e}");
        std::process::exit(1);
    }
}
