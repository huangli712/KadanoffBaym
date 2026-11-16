//
// Usage:
//   cargo run --example two_level -- equilibrium|nonequilibrium <config.toml> <out.dat>
//
// The two-level embedding problem of NESSi examples/programs/
// test_equilibrium.cpp and test_nonequilibrium.cpp: a single level
// `eps1` coupled to a second level `eps2` (hopping `i lambda`), solved
// through the embedded 1 x 1 Dyson/VIE2 equations and compared with the
// (1,1) element of the exact 2 x 2 solution.
//

#[path = "../common/mod.rs"]
mod common;
mod equilibrium;
mod nonequilibrium;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (sel, rest) = match args.split_first() {
        Some((s, r)) if !r.is_empty() => (s.as_str(), r),
        _ => {
            eprintln!("usage: two_level <equilibrium|nonequilibrium> <config.toml> <out.dat>");
            std::process::exit(1);
        }
    };
    let r = match sel {
        "equilibrium" => equilibrium::test_equilibrium_main(&rest[0], &rest[1]).map(|_| ()),
        "nonequilibrium" => {
            nonequilibrium::test_nonequilibrium_main(&rest[0], &rest[1]).map(|_| ())
        }
        _ => {
            eprintln!("usage: two_level <equilibrium|nonequilibrium> <config.toml> <out.dat>");
            std::process::exit(1);
        }
    };
    if let Err(e) = r {
        eprintln!("{e}");
        std::process::exit(1);
    }
}
