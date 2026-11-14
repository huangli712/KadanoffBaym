//
// Usage:
//   cargo run --example holstein_impurity_singlebath_migdal -- julia/tutor/holstein_impurity/holstein_impurity_migdal.toml [--profile quick|release] [--validate-only]
//
// Holstein impurity coupled to a single bath site (migdal self-energy),
// from NESSi examples/programs/Holstein_impurity_singlebath_migdal.cpp.
//

#[path = "common/mod.rs"]
mod common;
#[path = "impl/holstein_impl.rs"]
mod holstein_impl;
#[path = "impl/holstein_energy.rs"]
mod holstein_energy;
#[path = "impl/holstein_impurity_impl.rs"]
mod holstein_impurity_impl;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if let Err(e) =
        holstein_impurity_impl::holstein_impurity_main(&args, "Migdal")
    {
        eprintln!("{e}");
        std::process::exit(1);
    }
}
