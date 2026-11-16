//
// Usage:
//   cargo run --release --example vie2_bench [-- --nt 500 --ktau 200 --dim 2 --k 5]
//
// Kernel benchmark of the VIE2 propagation [1+F] G = Q (counterpart
// of .nessi/bench/kernel_vie2_bench.cpp): vie2_mat (fixpoint) +
// vie2_start + the vie2_timestep loop over deterministic
// pseudo-random F (scaled small for convergence) and Q.
//

use std::time::Instant;

use kadanoffbaym::complex::Complex;
use kadanoffbaym::dyson::MatsubaraMethod;
use kadanoffbaym::types::{Contour, ContourGreen, Sign};
use kadanoffbaym::vie2::{vie2_mat, vie2_start, vie2_timestep};
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
    let nt = arg_or("--nt", 500);
    let ntau = arg_or("--ktau", 200);
    let d = arg_or("--dim", 2) as usize;
    let k = arg_or("--k", 5);
    let (beta, h) = (10.0f64, 0.05f64);

    let c = Contour::new(nt, ntau, d as i64, d as i64, nt as f64 * h, beta);
    let mut g = ContourGreen::new(&c, Sign::Fermi, Complex::ZERO);
    let mut f = ContourGreen::new(&c, Sign::Fermi, Complex::ZERO);
    let mut q = ContourGreen::new(&c, Sign::Fermi, Complex::ZERO);
    let mut seed: u64 = 0xD1B5_4A32_D192_ED03;

    // small pseudo-random kernel (|F| < 0.05), full-size source Q
    for m in 0..=ntau as usize {
        for i in 0..d {
            for j in 0..d {
                let v = 0.05 * prand(&mut seed);
                f.set_mat(m as i64, i, j, v);
                let w = prand(&mut seed);
                q.set_mat(m as i64, i, j, w);
            }
        }
    }
    for n in 0..=nt as i64 {
        for m in 0..=n {
            for i in 0..d {
                for j in 0..d {
                    let v = 0.05 * prand(&mut seed);
                    f.set_ret(n, m, i, j, v);
                    f.set_less(m, n, i, j, v);
                    let w = prand(&mut seed);
                    q.set_ret(n, m, i, j, w);
                    q.set_less(m, n, i, j, w);
                }
            }
        }
        for m in 0..=ntau as i64 {
            for i in 0..d {
                for j in 0..d {
                    let v = 0.05 * prand(&mut seed);
                    f.set_lmix(n, m, i, j, v);
                    let w = prand(&mut seed);
                    q.set_lmix(n, m, i, j, w);
                }
            }
        }
    }

    let integ = Integrator::new(k);
    let t0 = Instant::now();
    vie2_mat(
        &mut g,
        &f,
        &f,
        &q,
        beta,
        &integ,
        MatsubaraMethod::Fixpoint,
        100,
        100,
        1e-8,
    );
    vie2_start(&mut g, &f, &f, &q, k, beta, h);
    for n in (k + 1)..=nt {
        vie2_timestep(
            n,
            &mut g,
            &f,
            &f,
            &q,
            &integ,
            beta,
            h,
            MatsubaraMethod::Fixpoint,
        );
    }
    let dt = t0.elapsed().as_secs_f64();

    let fin = g
        .mat
        .raw()
        .iter()
        .all(|z| z.re.is_finite() && z.im.is_finite());
    println!("kb vie2 nt={nt} ntau={ntau} d={d} k={k}: {dt:.4} s (finite: {fin})");
}
