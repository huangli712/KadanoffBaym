// Project : KadanoffBaym
// File    : examples/bench_convolution.rs
// Purpose : benchmark of the full-contour Type-1 convolution C = A * B.
//
// Usage:
//   cargo run --release --example bench_convolution [-- --nt 400 --ktau 200 --dim 2 --k 5]
//

#[path = "../util/mod.rs"]
mod util;
use std::time::Instant;

use kadanoffbaym::weights::Integrator;

use kadanoffbaym::convolution::convolution;

use util::{arg_or_default, prand};

use util::prelude::*;

fn main() {
    let nt = arg_or_default("--nt", 400);
    let ntau = arg_or_default("--ktau", 200);
    let d = arg_or_default("--dim", 2) as usize;
    let k = arg_or_default("--k", 5);
    let (beta, h) = (10.0f64, 0.05f64);

    let c = Contour::new(nt, ntau, d as i64, d as i64, nt as f64 * h, beta);
    let mut a = ContourGreen::new(&c, Sign::Fermi, Complex::ZERO);
    let mut b = ContourGreen::new(&c, Sign::Fermi, Complex::ZERO);
    let mut g = ContourGreen::new(&c, Sign::Fermi, Complex::ZERO);
    let mut seed: u64 = 0x9E37_79B9_7F4A_7C15;

    // fill all four components of A and B with pseudo-random blocks
    for m in 0..=ntau as usize {
        for i in 0..d {
            for j in 0..d {
                let v = prand(&mut seed);
                a.set_mat(m as i64, i, j, v);
                b.set_mat(m as i64, i, j, v);
            }
        }
    }
    for n in 0..=nt {
        for m in 0..=n {
            for i in 0..d {
                for j in 0..d {
                    let v = prand(&mut seed);
                    a.set_ret(n, m, i, j, v);
                    b.set_ret(n, m, i, j, v);
                    let w = prand(&mut seed);
                    a.set_less(m, n, i, j, w);
                    b.set_less(m, n, i, j, w);
                }
            }
        }
        for m in 0..=ntau {
            for i in 0..d {
                for j in 0..d {
                    let v = prand(&mut seed);
                    a.set_lmix(n, m, i, j, v);
                    b.set_lmix(n, m, i, j, v);
                }
            }
        }
    }

    let integ = Integrator::new(k);
    // warm-up run, then the timed run
    convolution(&mut g, &a, &a, None, &b, &b, &integ, beta, h);
    let t0 = Instant::now();
    convolution(&mut g, &a, &a, None, &b, &b, &integ, beta, h);
    let dt = t0.elapsed().as_secs_f64();

    let fin = g
        .mat
        .raw()
        .iter()
        .all(|z| z.re.is_finite() && z.im.is_finite());
    println!("kb conv nt={nt} ntau={ntau} d={d} k={k}: {dt:.4} s (finite: {fin})");
}
