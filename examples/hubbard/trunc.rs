// Project : KadanoffBaym
// File    : examples/hubbard/trunc.rs
// Purpose : full Kadanoff-Baym start and truncated propagation on the Bethe lattice.
//
// Usage:
//   cargo run --example hubbard -- bethe_trunc examples/hubbard/bethe_trunc.toml [--start-only] [--checkpoint]
//

use std::fs::File;
use std::io::Write;

use kadanoffbaym::observables::density;

use super::util::fmt_e12;

use super::bethe_trunc::{
    initialize_moving_state, load_params, moving_density, propagate_bethe_moving, read_checkpoint,
    solve_start, write_checkpoint,
};

pub(crate) fn run_truncated(
    cfgfile: &str,
    start_only: bool,
    checkpoint: bool,
) -> Result<(), String> {
    let p = load_params(cfgfile)?;
    let start = solve_start(&p)?;
    if start_only {
        let nt = start.g.nt as i64;
        let mut all_finite = true;
        for n in 0..=nt {
            let d = density(&start.g, n).re;
            all_finite &= d.is_finite();
        }
        if !all_finite {
            return Err("non-finite density".to_string());
        }
        println!(
            "full start: density(t_max) = {:.10}",
            density(&start.g, nt).re
        );
        return Ok(());
    }

    let mut state = initialize_moving_state(&start, p.tc);
    if checkpoint {
        let cdir = format!("{}/{}_checkpoint", p.dir, p.prefix);
        write_checkpoint(&cdir, &state, p.h)?;
        state = read_checkpoint(&cdir, p.h)?;
    }

    // Densities of the window at the handoff, the KB steps tc+1..1
    struct Row {
        tstp: i64,
        density: f64,
        densk: Vec<f64>,
    }
    let mut rows: Vec<Row> = (0..=p.tc)
        .rev()
        .map(|j| Row {
            tstp: state.g.tstp - j,
            density: moving_density(&state.g, j),
            densk: state.gk.iter().map(|g| moving_density(g, j)).collect(),
        })
        .collect();
    let steps = propagate_bethe_moving(&mut state, &p);
    for s in &steps {
        rows.push(Row {
            tstp: s.tstp,
            density: s.density,
            densk: s.densk.clone(),
        });
    }

    let mut f =
        File::create(format!("{}/{}_density.dat", p.dir, p.prefix)).map_err(|e| e.to_string())?;
    writeln!(f, "# tstp  t  density  densk...").map_err(|e| e.to_string())?;
    for r in &rows {
        let mut line = format!(
            "{} {} {}",
            r.tstp,
            fmt_e12((r.tstp - 1) as f64 * p.h),
            fmt_e12(r.density)
        );
        for x in &r.densk {
            line.push(' ');
            line.push_str(&fmt_e12(*x));
        }
        writeln!(f, "{line}").map_err(|e| e.to_string())?;
    }
    let mut f =
        File::create(format!("{}/{}_energy.dat", p.dir, p.prefix)).map_err(|e| e.to_string())?;
    writeln!(f, "# tstp  t  ekin_trunc  corrector_error").map_err(|e| e.to_string())?;
    for s in &steps {
        writeln!(
            f,
            "{} {} {} {}",
            s.tstp,
            fmt_e12((s.tstp - 1) as f64 * p.h),
            fmt_e12(s.ekin),
            fmt_e12(s.err)
        )
        .map_err(|e| e.to_string())?;
    }
    println!(
        "truncated propagation: density(t_max) = {:.10}",
        rows.last().unwrap().density
    );
    Ok(())
}
