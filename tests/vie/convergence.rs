// Project : KadanoffBaym
// File    : tests/vie/convergence.rs
// Purpose : convergence-order gate of the real-time VIE2 solver
//

use super::*;

#[test]
fn vie2_convergence_order() {
    for k in 1..=5i64 {
        let (ns, caps): (&[i64], [f64; 5]) = if is_release() {
            (&[20, 40, 80, 160], [5e-7, 1e-8, 2e-10, 1e-11, 1e-12])
        } else {
            (&[8, 16, 32, 64], [5e-6, 2e-7, 1e-8, 1e-9, 1e-10])
        };
        let mut errs3 = [Vec::new(), Vec::new(), Vec::new()];
        for &n in ns {
            let e = vt_errors(k, n, Sign::Fermi);
            errs3[0].push(e.0);
            errs3[1].push(e.1);
            errs3[2].push(e.2);
        }
        for errs in &errs3 {
            let gate = check_order_gate(
                ns,
                errs,
                (k + 2) as f64,
                caps[(k - 1) as usize],
                1e-13,
                0.35,
            );
            assert!(gate.ok, "k = {k}: {:?} (errs {errs:?})", gate.msg);
        }
        // reduced-order mutant at k = 2
        if k == 2 {
            let mut m3 = [Vec::new(), Vec::new(), Vec::new()];
            for &n in ns {
                let e = vt_errors(1, n, Sign::Fermi);
                m3[0].push(e.0);
                m3[1].push(e.1);
                m3[2].push(e.2);
            }
            for errs in &m3 {
                let gate = check_order_gate(ns, errs, 4.0, caps[1], 1e-13, 0.35);
                assert!(!gate.ok, "mutant must be rejected");
            }
        }
    }
}
