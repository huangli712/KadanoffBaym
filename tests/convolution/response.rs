// Project : KadanoffBaym
// File    : tests/convolution/response.rs
// Purpose : point-wise comparison of the response convolution with the
//           reference data
//

use kadanoffbaym::weights::Integrator;

use kadanoffbaym::convolution::response_convolution;

use super::reference::{conv_green_grid, conv_middle, data_dir};

use super::util::prelude::*;

use super::util::read_ref;

// ------------------------------------------------------------------
// helpers
// ------------------------------------------------------------------

/// The contour function of the reference generator: `middle(t)` with
/// `t = -0.5` at the Matsubara slot and `t = i * h` at time step `i`.
fn response_middle(nt: i64, size: usize, h: f64) -> TimeFunction {
    let mut f = TimeFunction::new(nt as usize, size, size, Complex::ZERO);
    let m = conv_middle(-0.5);
    for a in 0..size {
        for b in 0..size {
            f.set(MATSUBARA, a, b, m.get(a, b));
        }
    }
    for i in 0..=nt {
        let m = conv_middle(i as f64 * h);
        for a in 0..size {
            for b in 0..size {
                f.set(i, a, b, m.get(a, b));
            }
        }
    }
    f
}

// ------------------------------------------------------------------
// tests
// ------------------------------------------------------------------

/// `response_convolution` against the NESSi reference values: the free
/// Green's function `A` as the response function, the contour function
/// of the generator as the perturbation, sizes 1 and 2, orders 1 to 5,
/// at the Matsubara axis and at the time steps `tstp >= k`.
#[test]
fn convolution_response_reference() {
    let path = data_dir().join("response_values.dat");
    let reference = read_ref(path.to_str().unwrap()).expect("read ref");
    assert_eq!(reference.metadata["generator"], "response");
    assert_eq!(reference.metadata["case_set"], "response_values");
    // the grid is taken from the header of the reference data
    let header = |key: &str| reference.metadata[key].clone();
    let nt: i64 = header("nt").parse().unwrap();
    let ntau: i64 = header("ntau").parse().unwrap();
    let beta: f64 = header("beta").parse().unwrap();
    let h: f64 = header("h").parse().unwrap();
    assert_eq!(
        header("orbitals"),
        "0:(0,0,0,0),1:(0,1,1,0),2:(1,0,0,1),3:(1,1,1,1)"
    );
    // the inputs of the generator: W is the free Green's function A
    let free = [
        conv_green_grid(0, 1, nt, ntau, beta, h).1,
        conv_green_grid(0, 2, nt, ntau, beta, h).1,
    ];
    let middle = [response_middle(nt, 1, h), response_middle(nt, 2, h)];
    let orbs = [
        (0usize, 0usize, 0usize, 0usize),
        (0, 1, 1, 0),
        (1, 0, 0, 1),
        (1, 1, 1, 1),
    ];
    let mut worst = 0.0f64;
    let mut orders = std::collections::BTreeSet::new();
    for r in &reference.rows {
        assert_eq!(r.component, "response");
        // id = (((size - 1) * 5 + (k - 1)) * 4 + orb) * 16 + (tstp + 1)
        let group = r.i / 16;
        let tstp = r.i % 16 - 1;
        let orb = (group % 4) as usize;
        let k = (group / 4) % 5 + 1;
        let size = (group / 20 + 1) as usize;
        if size == 1 {
            assert_eq!(orb, 0);
        }
        let (a1, a2, b1, b2) = orbs[orb];
        assert!(tstp == MATSUBARA || tstp >= k);
        let integ = Integrator::new(k);
        let got = response_convolution(
            tstp,
            &free[size - 1],
            a1,
            a2,
            &middle[size - 1],
            b1,
            b2,
            &integ,
            beta,
            h,
        );
        let e = (got - r.value).abs() / (1e-10 + 1e-10 * r.value.abs());
        assert!(
            e < 1.0,
            "response: size {size}, k {k}, orb {orb}, tstp {tstp}: got {got}, reference {}",
            r.value
        );
        worst = worst.max(e);
        orders.insert(k);
    }
    assert_eq!(orders.into_iter().collect::<Vec<i64>>(), [1, 2, 3, 4, 5]);
    println!(
        "  response reference: {} values, worst relative error {worst:.2e}",
        reference.rows.len()
    );
}
