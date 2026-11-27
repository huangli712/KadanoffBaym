// Project : KadanoffBaym
// File    : src/convolution/check.rs
// Purpose : parameter checks of the convolution entry points, of the
//           pseudo-particle and the bubble families: the shape relation
//           of the objects and of their partner containers, the aliasing,
//           the orbital indices, the mask

use std::ptr;

use crate::types::{ContourGreen, MovingGreen, PseudoGreen};
use crate::types::{MatComp, TimeFunction};

use crate::types::MATSUBARA;

// ---- standard convolution ----

/// Shape contract of `C = A * f * B` (of `C = A * B` without `f`): the
/// columns of `A` match the rows of `f` (of `B` without `f`), the columns
/// of `f` match the rows of `B`, and `C` has the rows of `A` and the
/// columns of `B`.
pub(super) fn check_conv_shapes(
    c: (usize, usize),
    a: (usize, usize),
    f: Option<(usize, usize)>,
    b: (usize, usize),
) {
    let (s1c, s2c) = c;
    let (s1a, s2a) = a;
    let (s1b, s2b) = b;
    let inner = match f {
        Some((s1f, s2f)) => s2a == s1f && s2f == s1b,
        None => s2a == s1b,
    };
    assert!(
        inner && s1c == s1a && s2c == s2b,
        "convolution: inconsistent shapes: A {}x{}, f {:?}, B {}x{}, C {}x{}",
        s1a,
        s2a,
        f,
        s1b,
        s2b,
        s1c,
        s2c
    );
}

/// The partner objects hold the transposed blocks: `acc` has `a.size2`
/// rows and `a.size1` columns, `bcc` has `b.size2` rows and `b.size1`
/// columns.
pub(super) fn check_partner(name: &str, p: (usize, usize), of: (usize, usize)) {
    assert!(
        p.0 == of.1 && p.1 == of.0,
        "convolution: {} {}x{} does not match its partner {}x{} ({}x{} blocks needed)",
        name,
        p.0,
        p.1,
        of.0,
        of.1,
        of.1,
        of.0
    );
}

/// [`check_conv_shapes`] and the partner checks, for the usual argument
/// list of the convolution entry points.
pub(super) fn check_conv_args(
    c: (usize, usize),
    a: &ContourGreen,
    acc: &ContourGreen,
    f: Option<&TimeFunction>,
    b: &ContourGreen,
    bcc: &ContourGreen,
) {
    check_conv_shapes(
        c,
        (a.size1, a.size2),
        f.map(|cf| (cf.size1, cf.size2)),
        (b.size1, b.size2),
    );
    check_partner("Acc", (acc.size1, acc.size2), (a.size1, a.size2));
    check_partner("Bcc", (bcc.size1, bcc.size2), (b.size1, b.size2));
}

/// Shape contract of the Matsubara convolution `C = A * B`: one common
/// Matsubara grid and the block shapes of a matrix product.
pub(super) fn check_conv_mat_shapes(a: &MatComp, b: &MatComp, c: &MatComp) {
    assert!(
        a.ntau == b.ntau
            && b.ntau == c.ntau
            && a.size2 == b.size1
            && c.size1 == a.size1
            && c.size2 == b.size2,
        "conv_mat: inconsistent shapes: A {}x{}, B {}x{}, C {}x{} (ntau {} / {} / {})",
        a.size1,
        a.size2,
        b.size1,
        b.size2,
        c.size1,
        c.size2,
        a.ntau,
        b.ntau,
        c.ntau
    );
}

// ---- pseudo-particle convolution ----

/// Shape contract of the pseudo-particle convolution `C = A * f * B`:
/// the objects are square of one size, and so is `f`.
pub(super) fn check_pseudo_shapes(
    c: (usize, usize),
    a: &PseudoGreen,
    f: Option<&TimeFunction>,
    b: &PseudoGreen,
) {
    let d = a.size1;
    let ok = a.size2 == d
        && b.size1 == d
        && b.size2 == d
        && c.0 == d
        && c.1 == d
        && f.is_none_or(|cf| cf.size1 == d && cf.size2 == d);
    assert!(
        ok,
        "pseudo convolution: all shapes must be one square: A {}x{}, f {:?}, B {}x{}, C {}x{}",
        a.size1,
        a.size2,
        f.map(|cf| (cf.size1, cf.size2)),
        b.size1,
        b.size2,
        c.0,
        c.1
    );
}

/// Check the optional mask.
pub(super) fn check_pseudo_mask(mask: Option<&[bool]>, n: usize) {
    if let Some(mk) = mask {
        assert_eq!(mk.len(), n, "the mask must have {n} entries");
    }
}

// ---- bubble diagrams ----

/// Every orbital index must lie inside the block of its object.
pub(super) fn check_bubble_orbital(indices: &[(usize, usize, &str)]) {
    for &(x, dim, name) in indices {
        assert!(
            x < dim,
            "bubble: orbital index {name} = {x} is out of bounds (the block has {dim} entries)"
        );
    }
}

/// Parameter check of `bubble1`/`bubble2`: no aliasing of `C` with the
/// inputs, a common imaginary time grid, the partner objects with the
/// transposed shape, and the statistics `sign(C) = sign(A) sign(B)`.
pub(super) fn check_bubble(
    tstp: i64,
    c: &ContourGreen,
    a: &ContourGreen,
    acc: &ContourGreen,
    b: &ContourGreen,
    bcc: &ContourGreen,
) {
    assert!(!ptr::eq(c, a) && !ptr::eq(c, acc) && !ptr::eq(c, b) && !ptr::eq(c, bcc));
    assert_eq!(a.ntau, c.ntau, "imaginary time grids are not compatible");
    assert!(tstp == MATSUBARA || (0..=c.nt as i64).contains(&tstp));
    // The partner objects hold the transposed blocks. Passing an object
    // as its own partner (the `*_herm` shortcuts) needs square blocks.
    assert!(
        acc.size1 == a.size2 && acc.size2 == a.size1,
        "bubble: Acc must have the transposed shape {}x{} of A, got {}x{}",
        a.size2,
        a.size1,
        acc.size1,
        acc.size2
    );
    assert!(
        bcc.size1 == b.size2 && bcc.size2 == b.size1,
        "bubble: Bcc must have the transposed shape {}x{} of B, got {}x{}",
        b.size2,
        b.size1,
        bcc.size1,
        bcc.size2
    );
    assert!(acc.sign == a.sign);
    assert!(bcc.sign == b.sign);
    assert_eq!(c.sign.as_i64(), a.sign.as_i64() * b.sign.as_i64());
}

/// Parameter check of the moving bubbles: every object initialized on one
/// common window at the leading step `tstp`, partner sizes, statistics,
/// and orbital indices.
#[allow(clippy::too_many_arguments)]
pub(super) fn check_moving_bubble(
    tstp: i64,
    objs: &[&MovingGreen],
    idx: (usize, usize, usize, usize, usize, usize),
) {
    for x in objs {
        assert!(x.is_initialized(), "the moving object is not initialized");
        assert!(x.tc == objs[0].tc, "the cut-offs differ");
        assert!(
            x.tstp == tstp,
            "tstp = {tstp} is not the leading time step {}",
            x.tstp
        );
    }
    let c = objs[0];
    let (a, acc, b, bcc) = (objs[1], objs[2], objs[3], objs[4]);
    assert!(
        acc.ret.size1 == a.ret.size1 && acc.ret.size2 == a.ret.size2,
        "A and Acc have different sizes"
    );
    assert!(
        bcc.ret.size1 == b.ret.size1 && bcc.ret.size2 == b.ret.size2,
        "B and Bcc have different sizes"
    );
    assert!(
        c.sign.as_i64() == a.sign.as_i64() * b.sign.as_i64(),
        "the statistics of C must be sign(A) * sign(B)"
    );
    let (c1, c2, a1, a2, b1, b2) = idx;
    for (x, i) in [(c, c1), (c, c2), (a, a1), (a, a2), (b, b1), (b, b2)] {
        assert!(i < x.ret.size1, "orbital index {i} out of range");
    }
}
