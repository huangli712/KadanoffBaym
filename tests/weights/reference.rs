// Project : KadanoffBaym
// File    : tests/weights/reference.rs
// Purpose : point-wise comparison of the integration weights with the
//           reference data
//

use std::collections::{BTreeMap, BTreeSet};

use kadanoffbaym::weights::Integrator;

use super::util::RefRow;

use super::util::read_ref;

// ------------------------------------------------------------------
// helpers
// ------------------------------------------------------------------

fn data_path(name: &str) -> String {
    format!("{}/data/test/{name}", env!("CARGO_MANIFEST_DIR"))
}

fn is_close(v: f64, w: f64, atol: f64, rtol: f64) -> bool {
    (v - w).abs() <= atol + rtol * w.abs()
}

/// Weight value of a ref row .
fn weight_value(row: &RefRow) -> f64 {
    let integ = Integrator::new(row.i);
    match row.component.as_str() {
        "greg" => integ.giw.at(row.j as usize, row.a as usize),
        "bd" => integ.bdw.at(row.j as usize),
        "pint" => integ.xiw.at(row.j as usize, row.a as usize, row.b as usize),
        "pdiff" => integ.pdw.at(row.j as usize, row.a as usize),
        "rcorr" => integ.bcw.at(row.j as usize, row.a as usize, row.b as usize),
        c => panic!("ref: unknown component {c}"),
    }
}

// ------------------------------------------------------------------
// tests
// ------------------------------------------------------------------

#[test]
fn metadata() {
    let ref_data = read_ref(&data_path("weights_k1_k5.dat")).expect("read ref");
    assert_eq!(ref_data.metadata["data_kind"], "weights");
    let comps: BTreeSet<&str> = ref_data.rows.iter().map(|r| r.component.as_str()).collect();
    assert_eq!(
        comps,
        ["bd", "greg", "pdiff", "pint", "rcorr"]
            .into_iter()
            .collect::<BTreeSet<_>>()
    );
    // rcorr exists only for k = 2:5 and m = 1:k-1
    for r in ref_data.rows.iter().filter(|r| r.component == "rcorr") {
        assert!(
            r.i >= 2 && 1 <= r.j && r.j < r.i,
            "rcorr k={} m={}",
            r.i,
            r.j
        );
    }
    let ks: BTreeSet<i64> = ref_data
        .rows
        .iter()
        .filter(|r| r.component != "rcorr")
        .map(|r| r.i)
        .collect();
    assert_eq!(ks, [1, 2, 3, 4, 5].into_iter().collect::<BTreeSet<_>>());
}

#[test]
fn point_wise_comparison() {
    let ref_data = read_ref(&data_path("weights_k1_k5.dat")).expect("read ref");
    let (atol, rtol) = (2e-12, 2e-12);
    let mut errs: BTreeMap<String, f64> = BTreeMap::new();
    let mut nfail = 0usize;
    for row in &ref_data.rows {
        let v = weight_value(row);
        let w = row.value.re;
        let e = errs.entry(row.component.clone()).or_insert(0.0);
        *e = e.max((v - w).abs());
        if !is_close(v, w, atol, rtol) {
            nfail += 1;
            if nfail <= 10 {
                eprintln!(
                    "reference weight mismatch {} k={} ({},{},{}) want={} got={}",
                    row.component, row.i, row.j, row.a, row.b, w, v
                );
            }
        }
    }
    assert_eq!(nfail, 0, "{nfail} weight entries mismatch");
    let max_err = errs.values().cloned().fold(0.0f64, f64::max);
    for (c, e) in &errs {
        eprintln!("  max |KB - reference| {c:<6} : {e:.3e}");
    }
    assert!(max_err <= 2e-12, "max err {max_err}");
}

#[test]
fn k1_boundary_convolution_weights() {
    // k = 1: rcorr is not allocated in reference, so W is empty.
    let integ = Integrator::new(1);
    assert_eq!(integ.bcw.w.len(), 0);
}
