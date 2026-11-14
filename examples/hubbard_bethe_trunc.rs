//
// Usage:
//   cargo run --example hubbard_bethe_trunc -- julia/tutor/hubbard_bethe/hubbard_bethe_trunc.toml [--start-only] [--checkpoint]
//
// Interaction quench of the Hubbard model on the Bethe lattice, second
// order perturbation theory: a full Kadanoff-Baym start followed by the
// truncated (moving window) propagation (the Rust port of
// julia/tutor/hubbard_bethe/hubbard_bethe_trunc.jl, from NESSi
// examples/programs/trunc_bethe_start.cpp and trunc_bethe.cpp).
//

#[path = "common/mod.rs"]
mod common;
#[path = "impl/ness_util.rs"]
mod ness_util;
#[path = "impl/trunc_model.rs"]
mod trunc_model;

use trunc_model::*;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        eprintln!("usage: hubbard_bethe_trunc <config.toml> [--start-only] [--checkpoint]");
        std::process::exit(1);
    }
    let start_only = args.iter().any(|a| a == "--start-only");
    let checkpoint = args.iter().any(|a| a == "--checkpoint");
    if let Err(e) = trunc_main(&args[0], start_only, checkpoint) {
        eprintln!("{e}");
        std::process::exit(1);
    }
}

/// The result of the truncated run.
#[allow(dead_code)]
pub struct TruncResult {
    pub steps: Vec<StepRecord>,
}

fn trunc_main(cfgfile: &str, start_only: bool, checkpoint: bool) -> Result<TruncResult, String> {
    use kadanoffbaym::observables::density_matrix_green;
    let p = read_trunc_params(cfgfile)?;
    let start = solve_bethe_start(&p)?;
    if start_only {
        let nt = start.g.nt as i64;
        let mut all_finite = true;
        for n in 0..=nt {
            let d = density_matrix_green(&start.g, n).get(0, 0).re;
            all_finite &= d.is_finite();
        }
        if !all_finite {
            return Err("non-finite density".to_string());
        }
        println!(
            "full start: density(t_max) = {:.10}",
            density_matrix_green(&start.g, nt).get(0, 0).re
        );
        return Ok(TruncResult { steps: Vec::new() });
    }

    let mut state = initialize_moving_state(&start, p.tc);
    if checkpoint {
        let cdir = format!("{}/{}_checkpoint", p.dir, p.prefix);
        write_checkpoint(&cdir, &state, p.h)?;
        state = read_checkpoint(&cdir, p.h)?;
    }

    // Densities of the window at the handoff, the KB steps tc+1..1
    #[allow(dead_code)]
    struct Row {
        tstp: i64,
        density: f64,
        densk: Vec<f64>,
        ekin: f64,
        err: f64,
    }
    let mut rows: Vec<Row> = (0..=p.tc)
        .rev()
        .map(|j| Row {
            tstp: state.g.tstp - j,
            density: moving_density(&state.g, j),
            densk: state.gk.iter().map(|g| moving_density(g, j)).collect(),
            ekin: f64::NAN,
            err: f64::NAN,
        })
        .collect();
    let steps = propagate_bethe_moving(&mut state, &p);
    for s in &steps {
        rows.push(Row {
            tstp: s.tstp,
            density: s.density,
            densk: s.densk.clone(),
            ekin: s.ekin,
            err: s.err,
        });
    }

    use std::io::Write;
    let mut f = std::fs::File::create(format!("{}/{}_density.dat", p.dir, p.prefix))
        .map_err(|e| e.to_string())?;
    writeln!(f, "# tstp  t  density  densk...").map_err(|e| e.to_string())?;
    for r in &rows {
        let mut line = format!("{} {} {}", r.tstp, crate::ness_util::fmt_e12((r.tstp - 1) as f64 * p.h), crate::ness_util::fmt_e12(r.density));
        for x in &r.densk {
            line.push(' ');
            line.push_str(&crate::ness_util::fmt_e12(*x));
        }
        writeln!(f, "{line}").map_err(|e| e.to_string())?;
    }
    let mut f = std::fs::File::create(format!("{}/{}_energy.dat", p.dir, p.prefix))
        .map_err(|e| e.to_string())?;
    writeln!(f, "# tstp  t  ekin_trunc  corrector_error").map_err(|e| e.to_string())?;
    for s in &steps {
        writeln!(
            f,
            "{} {} {} {}",
            s.tstp,
            crate::ness_util::fmt_e12((s.tstp - 1) as f64 * p.h),
            crate::ness_util::fmt_e12(s.ekin),
            crate::ness_util::fmt_e12(s.err)
        )
        .map_err(|e| e.to_string())?;
    }
    println!(
        "truncated propagation: density(t_max) = {:.10}",
        rows.last().unwrap().density
    );
    Ok(TruncResult { steps })
}
