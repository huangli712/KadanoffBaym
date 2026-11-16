//
// Usage:
//   cargo run --release --example dyson_bench [-- --nt 500 --ktau 200 --dim 2]
//
// Microbenchmark of the Dyson real-time propagation (dyson_start +
// dyson_timestep over the full contour): a deterministic diagonal
// self-energy is applied to a free 2x2 propagator. Reports the wall
// time of the stepping loop only. Used by the performance tasks to
// measure allocation-free kernels; NOT part of the test suite.
//

use std::time::Instant;

use kadanoffbaym::complex::Complex;
use kadanoffbaym::dyson::{dyson_start, dyson_timestep};
use kadanoffbaym::equilibrium::from_ham;
use kadanoffbaym::linalg::DenseMat;
use kadanoffbaym::types::{Contour, ContourGreen, Sign, TimeFunction};

fn arg_or(name: &str, default: i64) -> i64 {
    let args: Vec<String> = std::env::args().collect();
    let mut val = default;
    let mut it = args.iter().skip(1);
    while let Some(a) = it.next() {
        if a == name
            && let Some(v) = it.next()
        {
            val = v.parse().unwrap_or(default);
        }
    }
    val
}

fn main() {
    let nt = arg_or("--nt", 500);
    let ntau = arg_or("--ktau", 200);
    let d = arg_or("--dim", 2) as usize;
    let k = 5i64;
    let (beta, h) = (10.0f64, 0.05f64);

    let c = Contour::new(nt, ntau, d as i64, d as i64, nt as f64 * h, beta);
    let mut g = ContourGreen::new(&c, Sign::Fermi, Complex::ZERO);

    // deterministic Hamiltonian and self-energy
    let mut h0 = DenseMat::zeros(d);
    for i in 0..d {
        h0.set(i, i, Complex::new(0.3 - 0.4 * i as f64, 0.0));
    }
    let mut htf = TimeFunction::new(nt as usize, d, d, Complex::ZERO);
    let mut sig = ContourGreen::new(&c, Sign::Fermi, Complex::ZERO);
    for t in -1..=nt {
        for i in 0..d {
            htf.set(t, i, i, Complex::new(0.02 * i as f64, 0.0));
        }
    }
    // constant diagonal self-energy on all components (ret/less/lmix/mat)
    for m in 0..=ntau as usize {
        for i in 0..d {
            sig.set_mat(
                m as i64,
                i,
                i,
                Complex::new(-0.2 * (0.1 + 0.05 * i as f64), 0.0),
            );
        }
    }
    for n in 0..=nt {
        for m in 0..=n {
            for i in 0..d {
                sig.set_ret(
                    n,
                    m,
                    i,
                    i,
                    Complex::new(0.15 * (0.1 + 0.05 * i as f64), 0.0),
                );
                sig.set_less(m, n, i, i, Complex::new(0.1 * (0.1 + 0.05 * i as f64), 0.0));
            }
        }
        for m in 0..=ntau {
            for i in 0..d {
                sig.set_lmix(
                    n,
                    m,
                    i,
                    i,
                    Complex::new(0.12 * (0.1 + 0.05 * i as f64), 0.0),
                );
            }
        }
    }

    // equilibrium start
    from_ham(&mut g, 0.0, &h0, beta, h);
    let t0 = Instant::now();
    dyson_start(&mut g, 0.0, &htf, &sig, beta, h, k);
    for n in (k + 1)..=nt {
        dyson_timestep(n, &mut g, 0.0, &htf, &sig, beta, h, k);
    }
    let dt = t0.elapsed().as_secs_f64();
    // sanity: the propagator stays finite
    let fin = g
        .ret
        .raw()
        .iter()
        .all(|z| z.re.is_finite() && z.im.is_finite());
    println!("dyson nt={nt} ntau={ntau} d={d} k={k}: {dt:.4} s (finite: {fin})");
}
