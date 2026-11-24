// Project : KadanoffBaym
// File    : tests/convolution/reference.rs
// Purpose : point-wise comparison of the Type-1/Type-2 convolutions
//           with the reference data
//

use std::path::{Path, PathBuf};

use kadanoffbaym::linalg::DenseMat;
use kadanoffbaym::weights::Integrator;

use kadanoffbaym::types::MATSUBARA;

use kadanoffbaym::convolution::{convolution, convolution_ab};
use kadanoffbaym::linalg::eigh;

use super::util::RefData;

use super::util::read_ref;

use super::util::prelude::*;

// ------------------------------------------------------------------
// constants
// ------------------------------------------------------------------

const CONV_BETA: f64 = 4.0;

const CONV_H: f64 = 0.05;

const CONV_MU: f64 = 0.1;

const CONV_NT: i64 = 30;

const CONV_NTAU: i64 = 60;

// ------------------------------------------------------------------
// helpers
// ------------------------------------------------------------------

/// Free fermionic Green's function for H(t) = H_0 + a(t) I (closed
/// form, nothing read back from the reference output).
fn conv_green(which: usize, size: usize) -> (Contour, ContourGreen) {
    let c = Contour::new(
        CONV_NT,
        CONV_NTAU,
        size as i64,
        size as i64,
        CONV_NT as f64 * CONV_H,
        CONV_BETA,
    );
    let mut g = ContourGreen::new(&c, Sign::Fermi, Complex::ZERO);
    let h = conv_hamiltonian(which);
    // epsilon = H0[0:size,0:size] - mu I
    let mut eps = DenseMat::zeros(size);
    for i in 0..size {
        for j in 0..size {
            eps.set(i, j, h.get(i, j));
        }
        eps.set(i, i, eps.get(i, i) - Complex::new(CONV_MU, 0.0));
    }
    let (vals, v) = eigh(&eps);
    // Matsubara component
    for q in 0..=CONV_NTAU {
        let tau = q as f64 * CONV_BETA / CONV_NTAU as f64;
        let gv: Vec<Complex> = vals
            .iter()
            .map(|&l| {
                -Complex::new((-l * tau).exp(), 0.0)
                    / Complex::new(1.0 + (-CONV_BETA * l).exp(), 0.0)
            })
            .collect();
        let m = diagonal_transform(&v, &vals, &gv);
        for i in 0..size {
            for j in 0..size {
                g.set_mat(q, i, j, m.get(i, j));
            }
        }
    }
    // Evolution and density
    let mut us = Vec::new();
    for i in 0..=CONV_NT {
        let t = i as f64 * CONV_H;
        let gv: Vec<Complex> = vals
            .iter()
            .map(|&l| Complex::from_polar(1.0, -l * t))
            .collect();
        let mut u = diagonal_transform(&v, &vals, &gv);
        let ph = Complex::from_polar(1.0, -phase_factor(which, t));
        for q in 0..(size * size) {
            u.raw_mut()[q] = u.raw()[q] * ph;
        }
        us.push(u);
    }
    let rhov: Vec<Complex> = vals
        .iter()
        .map(|&l| Complex::new(1.0 / (1.0 + (CONV_BETA * l).exp()), 0.0))
        .collect();
    let rho = diagonal_transform(&v, &vals, &rhov);
    for i in 0..=CONV_NT {
        for j in 0..=i {
            let m = us[i as usize].mat_mul(&us[j as usize].adjoint());
            for a in 0..size {
                for b in 0..size {
                    g.set_ret(i, j, a, b, Complex::new(0.0, -1.0) * m.get(a, b));
                }
            }
        }
        for j in i..=CONV_NT {
            let m = us[i as usize]
                .mat_mul(&rho)
                .mat_mul(&us[j as usize].adjoint());
            for a in 0..size {
                for b in 0..size {
                    g.set_less(i, j, a, b, Complex::new(0.0, 1.0) * m.get(a, b));
                }
            }
        }
        for q in 0..=CONV_NTAU {
            let tau = q as f64 * CONV_BETA / CONV_NTAU as f64;
            let gv: Vec<Complex> = vals
                .iter()
                .map(|&l| {
                    Complex::new((l * tau).exp(), 0.0)
                        / Complex::new(1.0 + (CONV_BETA * l).exp(), 0.0)
                })
                .collect();
            let m = diagonal_transform(&v, &vals, &gv);
            let mm = us[i as usize].mat_mul(&m);
            for a in 0..size {
                for b in 0..size {
                    g.set_lmix(i, q, a, b, Complex::new(0.0, 1.0) * mm.get(a, b));
                }
            }
        }
    }
    (c, g)
}

fn conv_hamiltonian(which: usize) -> DenseMat {
    let mut m = DenseMat::zeros(2);
    if which == 0 {
        m.set(0, 0, Complex::new(-0.5, 0.0));
        m.set(0, 1, Complex::new(0.2, 0.1));
        m.set(1, 0, Complex::new(0.2, -0.1));
        m.set(1, 1, Complex::new(0.4, 0.0));
    } else {
        m.set(0, 0, Complex::new(0.3, 0.0));
        m.set(0, 1, Complex::new(0.25, -0.05));
        m.set(1, 0, Complex::new(0.25, 0.05));
        m.set(1, 1, Complex::new(-0.2, 0.0));
    }
    m
}

fn conv_middle(t: f64) -> DenseMat {
    let mut m = DenseMat::zeros(2);
    m.set(0, 0, Complex::new(1.0 + 0.2 * t, 0.0));
    m.set(0, 1, Complex::new(0.1, 0.3 * t));
    m.set(1, 0, Complex::new(-0.2, 0.05));
    m.set(1, 1, Complex::new(0.8, 0.0));
    m
}

fn conv_middle_function(size: usize) -> TimeFunction {
    let mut f = TimeFunction::new(CONV_NT as usize, size, size, Complex::ZERO);
    let m0 = conv_middle(0.0);
    for a in 0..size {
        for b in 0..size {
            f.set(MATSUBARA, a, b, m0.get(a, b));
        }
    }
    for i in 0..=CONV_NT {
        let m = conv_middle(i as f64 * CONV_H);
        for a in 0..size {
            for b in 0..size {
                f.set(i, a, b, m.get(a, b));
            }
        }
    }
    f
}

fn data_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("data")
        .join("test")
}

/// `V diag(g) V^dagger`
fn diagonal_transform(v: &DenseMat, vals: &[f64], g: &[Complex]) -> DenseMat {
    let n = vals.len();
    let mut d = DenseMat::zeros(n);
    for (i, gi) in g.iter().enumerate() {
        d.set(i, i, *gi);
    }
    v.mat_mul(&d).mat_mul(&v.adjoint())
}

fn phase_factor(which: usize, t: f64) -> f64 {
    if which == 0 {
        0.3 * (1.0 - t.cos())
    } else {
        0.1 * (1.0 - (2.0 * t).cos())
    }
}

/// Per-component max errors of `g` against the ref rows (mat, ret,
/// tv, les) with combined absolute/relative tolerance weights.
fn reference_green(g: &ContourGreen, data: &RefData, atol: f64, rtol: f64) -> [f64; 4] {
    let mut errs = [0.0f64; 4];
    for r in &data.rows {
        let (got, slot) = match r.component.as_str() {
            // mat rows carry the tau index in `i` (j is -1)
            "mat" => (g.get_mat(r.i, r.a as usize, r.b as usize), 0),
            "ret" => (g.get_ret(r.i, r.j, r.a as usize, r.b as usize), 1),
            "tv" => (g.get_lmix(r.i, r.j, r.a as usize, r.b as usize), 2),
            "les" => (g.get_less(r.i, r.j, r.a as usize, r.b as usize), 3),
            other => panic!("unexpected component {other}"),
        };
        let d = (got - r.value).abs();
        let scale = atol + rtol * r.value.abs();
        errs[slot] = errs[slot].max(d / scale);
    }
    errs
}

fn run_case(typ: i64, size: usize, k: i64) {
    let name = format!("convolution_type{typ}_size{size}_k{k}");
    let path = data_dir().join(format!("{name}.dat"));
    let reference = read_ref(path.to_str().unwrap()).expect("read ref");
    assert_eq!(reference.metadata["case"], name);
    assert_eq!(reference.metadata["solve_order"], k.to_string());
    assert_eq!(reference.metadata["size1"], size.to_string());
    let (c, a) = conv_green(0, size);
    let (_, b) = conv_green(1, size);
    let mut cv = ContourGreen::new(&c, Sign::Fermi, Complex::ZERO);
    let integ = Integrator::new(k);
    if typ == 1 {
        convolution_ab(&mut cv, &a, &b, &integ, CONV_BETA, CONV_H);
    } else {
        let f = conv_middle_function(size);
        convolution(&mut cv, &a, &a, Some(&f), &b, &b, &integ, CONV_BETA, CONV_H);
    }
    let errs = reference_green(&cv, &reference, 1e-10, 1e-10);
    for (slot, e) in errs.iter().enumerate() {
        assert!(*e < 1.0, "{name}: component {slot} relative error {e:.3e}");
    }
    println!(
        "  {name:32} max errs: {:.2e} {:.2e} {:.2e} {:.2e}",
        errs[0], errs[1], errs[2], errs[3]
    );
}

// ------------------------------------------------------------------
// tests
// ------------------------------------------------------------------

#[test]
fn convolution_reference_all() {
    for size in [2usize, 1] {
        for typ in [1i64, 2] {
            for k in 1..=5i64 {
                run_case(typ, size, k);
            }
        }
    }
}
