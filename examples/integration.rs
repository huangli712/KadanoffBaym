//
// Usage:
//   cargo run --example integration -- julia/tutor/integration/integration.toml out.dat
//
// The integral of exp(i t) by the Gregory quadrature of order k, with a
// convergence-order gate (the Rust port of
// julia/tutor/integration/integration.jl, from NESSi
// examples/programs/integration.cpp).
//

#![allow(dead_code)]

#[path = "common/mod.rs"]
mod common;
#[path = "impl/ness_util.rs"]
mod ness_util;

use common::kbref::check_order_gate;
use kadanoffbaym::complex::Complex;
use kadanoffbaym::weights::Integrator;
use ness_util::{fmt_e12, tget_f, tget_i};
use std::io::Write;

fn cis(x: f64) -> Complex {
    Complex::from_polar(1.0, x)
}

/// The integral of `e^{it}` from 0 to `n h` by the Gregory quadrature.
fn gregory_integral(n: i64, h: f64, integ: &Integrator) -> Complex {
    let k = integ.k as i64;
    let f = |j: i64| cis(j as f64 * h);
    let mut s = Complex::ZERO;
    if n <= 2 * k + 2 {
        let n1 = if n < k { k } else { n };
        for j in 0..=n1 {
            s += integ.giw.at(n as usize, j as usize) * f(j);
        }
    } else {
        for j in 0..=k {
            s += integ.giw.omega[j as usize] * f(j);
        }
        for j in (k + 1)..=(n - k - 1) {
            s += f(j);
        }
        for j in (n - k)..=n {
            s += integ.giw.omega[(n - j) as usize] * f(j);
        }
    }
    s * Complex::new(h, 0.0)
}

/// The first order rectangle rule.
fn rectangle_integral(n: i64, h: f64) -> Complex {
    let mut s = Complex::ZERO;
    for j in 0..n {
        s += cis(j as f64 * h);
    }
    s * Complex::new(h, 0.0)
}

fn exact_integral(x: f64) -> Complex {
    (cis(x) - Complex::ONE) / Complex::new(0.0, 1.0)
}

/// Read the parameters, write the errors, and run the order gate
/// (blueprint `main`). Returns the observed orders of k = 1..5.
pub fn integration_main(cfgfile: &str, outfile: &str) -> Result<Vec<f64>, String> {
    let p = ness_util::inp_toml(cfgfile)?;
    let npts = tget_i(&p, "", "npts").or_else(|_| tget_i(&p, "grid", "npts"))?;
    let h = tget_f(&p, "", "h").or_else(|_| tget_f(&p, "grid", "h"))?;
    let k = tget_i(&p, "", "k").or_else(|_| tget_i(&p, "solver", "k"))?;
    if !((1..=5).contains(&k) && npts >= 0 && h > 0.0) {
        return Err(format!("{cfgfile}: invalid parameters"));
    }

    let integ = Integrator::new(k);
    let mut f = std::fs::File::create(outfile).map_err(|e| e.to_string())?;
    writeln!(f, "# x abs(error)").map_err(|e| e.to_string())?;
    for n in 0..=npts {
        let err = (gregory_integral(n, h, &integ) - exact_integral(n as f64 * h)).abs();
        writeln!(f, "{} {}", fmt_e12(n as f64 * h), format!("{err:.10e}"))
            .map_err(|e| e.to_string())?;
    }

    // the convergence order at x = X with N = X / h intervals
    let x = 2.5;
    let ns = [16, 32, 64, 128];
    let caps = [1e-6, 1e-8, 1e-10, 1e-11, 1e-12];
    let mut orders = Vec::new();
    for kk in 1..=5 {
        let ik = Integrator::new(kk);
        let errs: Vec<f64> = ns
            .iter()
            .map(|&n| (gregory_integral(n, x / n as f64, &ik) - exact_integral(x)).abs())
            .collect();
        let (ok, ord, msg) = check_order_gate(
            &ns,
            &errs,
            (kk + 2) as f64,
            caps[kk as usize - 1],
            1e-13,
            0.35,
        );
        if !ok {
            return Err(format!("integration: order gate of k = {kk}: {msg}"));
        }
        println!("    k = {kk}: observed order {ord:.2}, finest error {:.2e}", errs[3]);
        orders.push(ord);
        // the reduced-order mutant must fail the same gate
        let errm: Vec<f64> = ns
            .iter()
            .map(|&n| (rectangle_integral(n, x / n as f64) - exact_integral(x)).abs())
            .collect();
        let (ok, _, _) = check_order_gate(
            &ns,
            &errm,
            (kk + 2) as f64,
            caps[kk as usize - 1],
            1e-13,
            0.35,
        );
        if ok {
            return Err(format!("the rectangle rule passed the order gate of k = {kk}"));
        }
    }
    Ok(orders)
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() != 2 {
        eprintln!("usage: integration <config.toml> <output.dat>");
        std::process::exit(1);
    }
    if let Err(e) = integration_main(&args[0], &args[1]) {
        eprintln!("{e}");
        std::process::exit(1);
    }
}
