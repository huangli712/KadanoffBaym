//
// Usage:
//   cargo run --example hubbard_chain -- 2b|gw|tpp hubbard_chain.toml
//     [--profile quick|release] [--validate-only]
//
// Hubbard chain in the second Born / GW / particle-particle T-matrix
// approximation (NESSi examples/programs/hubbard_chain_{2b,gw,tpp}.cpp).
//

#[path = "../common/mod.rs"]
mod common;

#[path = "../impl/hubbard_chain_impl.rs"]
mod hubbard_chain_impl;

fn main() {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let code = match args.first().map(String::as_str) {
        Some("2b") => {
            args.remove(0);
            "2B"
        }
        Some("gw") => {
            args.remove(0);
            "GW"
        }
        Some("tpp") => {
            args.remove(0);
            "TPP"
        }
        _ => {
            eprintln!("usage: hubbard_chain <2b|gw|tpp> <config.toml> [flags]");
            std::process::exit(1);
        }
    };
    if let Err(e) = hubbard_chain_impl::hubbard_main(&args, code) {
        eprintln!("{e}");
        std::process::exit(1);
    }
}
