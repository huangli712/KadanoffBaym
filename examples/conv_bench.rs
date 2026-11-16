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

use std::time::Instant;

use kadanoffbaym::complex::Complex;
use kadanoffbaym::convolution::convolution;
use kadanoffbaym::types::{Contour, ContourGreen, Sign};
use kadanoffbaym::weights::Integrator;

fn arg_or(name: &str, default: i64) -> i64 {
    let args: Vec<String> = std::env::args().collect();
    let mut val = default;
    let mut it = args.iter().skip(1);
    while let Some(a) = it.next() {
        if a == name {
            if let Some(v) = it.next() {
                val = v.parse().unwrap_or(default);
            }
        }
    }
    val
}

/// deterministic pseudo-random complex in (-0.5, 0.5); the same LCG
/// as the C++ side (values need not agree between the two programs,
/// only determinism matters)
fn prand(state: &mut u64) -> Complex {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    let re = ((*state >> 33) % 1000) as f64 / 1000.0 - 0.5;
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    let im = ((*state >> 33) % 1000) as f64 / 1000.0 - 0.5;
    Complex::new(re, im)
}

fn main() {
    let nt = arg_or("--nt", 400);
    let ntau = arg_or("--ktau", 200);
    let d = arg_or("--dim", 2) as usize;
    let k = arg_or("--k", 5);
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
    for n in 0..=nt as i64 {
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
        for m in 0..=ntau as i64 {
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
