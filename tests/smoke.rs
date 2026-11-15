//! Smoke test of the R0 crate skeleton.

use kadanoffbaym::{F64, I64, VERSION};

#[test]
fn skeleton_exposes_aliases() {
    let x: F64 = 1.5;
    let n: I64 = 3;
    assert_eq!(x + 1.0, 2.5);
    assert_eq!(n + 1, 4);
    assert!(!VERSION.is_empty());
}
