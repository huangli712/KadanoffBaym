//
// Usage:
//   cargo run --release --example conv_bench [-- --nt 400 --ktau 200 --dim 2 --k 5]
//
// Kernel benchmark of the full-contour Type-1 convolution
// C = A * B (counterpart of .nessi/bench/kernel_conv_bench.cpp):
// deterministic pseudo-random data on all four components of A and B,
// Acc = A and Bcc = B (the common DMFT convention). One warm-up run,
// the second run is timed.
//

#[path = "../util/mod.rs"]
mod util;
use std::time::Instant;

use kadanoffbaym::complex::Complex;
use kadanoffbaym::convolution::convolution;
use kadanoffbaym::types::{Contour, ContourGreen, Sign};
use kadanoffbaym::weights::Integrator;
fn main() {
    let nt = util::arg_or("--nt", 400);
    let ntau = util::arg_or("--ktau", 200);
    let d = util::arg_or("--dim", 2) as usize;
    let k = util::arg_or("--k", 5);
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
                let v = util::prand(&mut seed);
                a.set_mat(m as i64, i, j, v);
                b.set_mat(m as i64, i, j, v);
            }
        }
    }
    for n in 0..=nt {
        for m in 0..=n {
            for i in 0..d {
                for j in 0..d {
                    let v = util::prand(&mut seed);
                    a.set_ret(n, m, i, j, v);
                    b.set_ret(n, m, i, j, v);
                    let w = util::prand(&mut seed);
                    a.set_less(m, n, i, j, w);
                    b.set_less(m, n, i, j, w);
                }
            }
        }
        for m in 0..=ntau {
            for i in 0..d {
                for j in 0..d {
                    let v = util::prand(&mut seed);
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
