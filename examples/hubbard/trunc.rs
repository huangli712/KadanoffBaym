//
// Usage:
//   cargo run --example hubbard -- bethe_trunc examples/hubbard/hubbard_bethe_trunc.toml [--start-only] [--checkpoint]
//
// Interaction quench of the Hubbard model on the Bethe lattice, second
// order perturbation theory: a full Kadanoff-Baym start followed by the
// truncated (moving window) propagation, from NESSi
// examples/programs/trunc_bethe_start.cpp and trunc_bethe.cpp.
//

use std::fs::File;
use std::io::Write;

use kadanoffbaym::observables::density;

use crate::util::fmt_e12;

use crate::trunc_model::*;

/// The result of the truncated run.
#[allow(dead_code)]
pub struct TruncResult {
    pub steps: Vec<StepRecord>,
}

pub fn trunc_main(
    cfgfile: &str,
    start_only: bool,
    checkpoint: bool,
) -> Result<TruncResult, String> {
    let p = read_trunc_params(cfgfile)?;
    let start = solve_bethe_start(&p)?;
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
    Ok(TruncResult { steps })
}
