// Project : KadanoffBaym
// File    : tests/convolution/nonsquare.rs
// Purpose : regression tests of the non-square convolution: the
//           reference inputs padded with zero rows / columns reproduce
//           the reference data, and every public entry point agrees
//           with the top-left block of the embedded square problem
//

use std::path::Path;

use kadanoffbaym::weights::Integrator;

use kadanoffbaym::convolution::{
    ConvWorkspace, FOp, conv_lmix_mat, convolution, convolution_timestep, incr_convolution,
    incr_convolution_mat, incr_convolution_step, incr_less_adv_buf, incr_lmix_rmix_buf,
};

use super::reference::{
    CONV_BETA, CONV_H, CONV_NT, CONV_NTAU, conv_green, conv_middle_function, reference_green,
};

use super::util::prelude::*;

use super::util::{RefData, prand, read_ref};

// ------------------------------------------------------------------
// constants
// ------------------------------------------------------------------

/// Grid of the embedding identity.
const NS_NT: i64 = 6;

const NS_NTAU: i64 = 4;

const NS_BETA: f64 = 2.0;

const NS_H: f64 = 0.1;

// ------------------------------------------------------------------
// helpers: the reference inputs padded into non-square objects
// ------------------------------------------------------------------

/// The `d x d` object `g` placed in the top-left block of a zero
/// `(rows x cols)` one on the same contour.
fn embed(g: &ContourGreen, rows: usize, cols: usize) -> ContourGreen {
    let d = g.size1;
    assert_eq!(g.size2, d);
    let contour = Contour::new(
        CONV_NT,
        CONV_NTAU,
        rows as i64,
        cols as i64,
        CONV_NT as f64 * CONV_H,
        CONV_BETA,
    );
    let mut o = ContourGreen::new(&contour, g.sign, Complex::ZERO);
    let place = |src: &[Complex], dst: &mut [Complex]| {
        for a in 0..d {
            for b in 0..d {
                dst[a * cols + b] = src[a * d + b];
            }
        }
    };
    for m in 0..=(CONV_NTAU as usize) {
        place(g.mat.block(m), o.mat.block_mut(m));
    }
    for p in 0..=CONV_NT {
        for q in 0..=p {
            place(
                g.ret.block(g.ret.idx(p, q)),
                o.ret.block_mut(o.ret.idx(p, q)),
            );
            place(
                g.less.block(g.less.idx(q, p)),
                o.less.block_mut(o.less.idx(q, p)),
            );
        }
        for tau in 0..=CONV_NTAU {
            place(
                g.lmix.block(g.lmix.idx(p, tau)),
                o.lmix.block_mut(o.lmix.idx(p, tau)),
            );
        }
    }
    o
}

/// The `d x d` middle function placed in the top-left block of a
/// `(rows x cols)` one. The other entries get `filler` and must not
/// influence the result.
fn embed_middle(f: &TimeFunction, rows: usize, cols: usize, filler: Complex) -> TimeFunction {
    let d = f.size1;
    let mut o = TimeFunction::new(CONV_NT as usize, rows, cols, filler);
    for t in MATSUBARA..=CONV_NT {
        let (src, dst) = (f.block(t), o.block_mut(t));
        for a in 0..d {
            for b in 0..d {
                dst[a * cols + b] = src[a * d + b];
            }
        }
    }
    o
}

/// Relative errors of the reference rows of one time-step object: the
/// rows of time step `t`, at `MATSUBARA` the Matsubara rows.
fn reference_step(s: &GreenTimeStep, t: i64, data: &RefData, atol: f64, rtol: f64) -> [f64; 4] {
    let mut errs = [0.0f64; 4];
    for r in &data.rows {
        let (got, slot) = match r.component.as_str() {
            "mat" => {
                if t != MATSUBARA {
                    continue;
                }
                (s.get_mat(r.i, r.a as usize, r.b as usize), 0)
            }
            "ret" => {
                if t != r.i {
                    continue;
                }
                (s.get_ret(r.i, r.j, r.a as usize, r.b as usize), 1)
            }
            "tv" => {
                if t != r.i {
                    continue;
                }
                (s.get_lmix(r.i, r.j, r.a as usize, r.b as usize), 2)
            }
            "les" => {
                if t != r.j {
                    continue;
                }
                (s.get_less(r.i, r.j, r.a as usize, r.b as usize), 3)
            }
            other => panic!("unexpected component {other}"),
        };
        let d = (got - r.value).abs();
        errs[slot] = errs[slot].max(d / (atol + rtol * r.value.abs()));
    }
    errs
}

// ------------------------------------------------------------------
// helpers: the embedded-square identity
// ------------------------------------------------------------------

fn contour(rows: usize, cols: usize) -> Contour {
    Contour::new(
        NS_NT,
        NS_NTAU,
        rows as i64,
        cols as i64,
        NS_NT as f64 * NS_H,
        NS_BETA,
    )
}

/// Pseudo-random source of an object with `(rows x cols)` blocks: every
/// block of every component and of both time triangles, so that the
/// adjoint partner objects are built from independent data.
struct Source {
    rows: usize,
    cols: usize,
    mat: Vec<Vec<Complex>>,
    ret: Vec<Vec<Complex>>,
    less: Vec<Vec<Complex>>,
    lmix: Vec<Vec<Complex>>,
}

fn source(rows: usize, cols: usize, state: &mut u64) -> Source {
    let (n, ntau) = ((NS_NT + 1) as usize, (NS_NTAU + 1) as usize);
    let blocks = |count: usize, state: &mut u64| -> Vec<Vec<Complex>> {
        (0..count)
            .map(|_| (0..rows * cols).map(|_| prand(state)).collect())
            .collect()
    };
    Source {
        rows,
        cols,
        mat: blocks(ntau, state),
        ret: blocks(n * n, state),
        less: blocks(n * n, state),
        lmix: blocks(n * ntau, state),
    }
}

/// Pseudo-random middle function with `(rows x cols)` blocks: one block
/// for the Matsubara slot and one for every time step.
struct Middle {
    rows: usize,
    cols: usize,
    blocks: Vec<Vec<Complex>>,
}

fn middle(rows: usize, cols: usize, state: &mut u64) -> Middle {
    Middle {
        rows,
        cols,
        blocks: (0..(NS_NT + 2) as usize)
            .map(|_| (0..rows * cols).map(|_| prand(state)).collect())
            .collect(),
    }
}

/// the block `(p, q)` of `n` blocks per row
fn at(blocks: &[Vec<Complex>], n: usize, p: usize, q: usize) -> &[Complex] {
    &blocks[p * n + q]
}

/// The `(rows x cols)` source block placed in the top-left corner of a
/// zero `(s1 x s2)` one.
fn place(block: &[Complex], rows: usize, cols: usize, s1: usize, s2: usize) -> Vec<Complex> {
    let mut out = vec![Complex::ZERO; s1 * s2];
    for i in 0..rows.min(s1) {
        for j in 0..cols.min(s2) {
            out[i * s2 + j] = block[i * cols + j];
        }
    }
    out
}

/// the adjoint of an `(s1 x s2)` block
fn adjoint(block: &[Complex], s1: usize, s2: usize) -> Vec<Complex> {
    let mut out = vec![Complex::ZERO; s1 * s2];
    for i in 0..s1 {
        for j in 0..s2 {
            out[j * s1 + i] = block[i * s2 + j].conj();
        }
    }
    out
}

/// The object of shape `(s1 x s2)` carrying the source (in its top-left
/// block) and its adjoint partner object of shape `(s2 x s1)`: the
/// partner holds the adjoints of the other time triangles of the source.
fn pair(src: &Source, s1: usize, s2: usize) -> (ContourGreen, ContourGreen) {
    let (rows, cols) = (src.rows, src.cols);
    let n = (NS_NT + 1) as usize;
    let mut g = ContourGreen::new(&contour(s1, s2), Sign::Fermi, Complex::ZERO);
    let mut gc = ContourGreen::new(&contour(s2, s1), Sign::Fermi, Complex::ZERO);
    for tau in 0..=NS_NTAU {
        let blk = at(&src.mat, 1, 0, tau as usize);
        g.mat
            .block_mut(tau as usize)
            .copy_from_slice(&place(blk, rows, cols, s1, s2));
    }
    for p in 0..=NS_NT {
        for tau in 0..=NS_NTAU {
            let blk = at(&src.lmix, (NS_NTAU + 1) as usize, p as usize, tau as usize);
            let placed = place(blk, rows, cols, s1, s2);
            g.lmix
                .block_mut(g.lmix.idx(p, tau))
                .copy_from_slice(&placed);
            gc.lmix
                .block_mut(gc.lmix.idx(p, tau))
                .copy_from_slice(&adjoint(&placed, s1, s2));
        }
        for q in 0..=NS_NT {
            if p >= q {
                let blk = at(&src.ret, n, p as usize, q as usize);
                g.ret
                    .block_mut(g.ret.idx(p, q))
                    .copy_from_slice(&place(blk, rows, cols, s1, s2));
            }
            if p <= q {
                let blk = at(&src.less, n, p as usize, q as usize);
                g.less
                    .block_mut(g.less.idx(p, q))
                    .copy_from_slice(&place(blk, rows, cols, s1, s2));
            }
            if q <= p {
                // the partner blocks: adjoints of the other triangles
                let blk = at(&src.ret, n, q as usize, p as usize);
                gc.ret.block_mut(gc.ret.idx(p, q)).copy_from_slice(&adjoint(
                    &place(blk, rows, cols, s1, s2),
                    s1,
                    s2,
                ));
                let blk = at(&src.less, n, p as usize, q as usize);
                gc.less
                    .block_mut(gc.less.idx(q, p))
                    .copy_from_slice(&adjoint(&place(blk, rows, cols, s1, s2), s1, s2));
            }
        }
    }
    (g, gc)
}

/// the middle function source placed in a `(s1 x s2)` container
fn place_middle(src: &Middle, s1: usize, s2: usize) -> TimeFunction {
    let mut f = TimeFunction::new(NS_NT as usize, s1, s2, Complex::ZERO);
    for t in MATSUBARA..=NS_NT {
        f.block_mut(t).copy_from_slice(&place(
            &src.blocks[(t + 1) as usize],
            src.rows,
            src.cols,
            s1,
            s2,
        ));
    }
    f
}

type Block = (&'static str, i64, i64, Vec<Complex>);

type Blocks = Vec<Block>;

/// one entry point of the convolution, as a list of output blocks
type Entry = (&'static str, Box<dyn Fn(&Case) -> Blocks>);

/// every stored block of a full object, in a fixed order
fn blocks(g: &ContourGreen) -> Blocks {
    let mut out = Vec::new();
    for tau in 0..=g.ntau as i64 {
        out.push(("mat", tau, 0, g.mat.block(tau as usize).to_vec()));
    }
    for t in 0..=g.nt as i64 {
        for tau in 0..=g.ntau as i64 {
            out.push(("lmix", t, tau, g.lmix.block(g.lmix.idx(t, tau)).to_vec()));
        }
        for j in 0..=t {
            out.push(("ret", t, j, g.ret.block(g.ret.idx(t, j)).to_vec()));
            out.push(("less", j, t, g.less.block(g.less.idx(j, t)).to_vec()));
        }
    }
    out
}

/// the blocks of one time-step row (the Matsubara blocks at `MATSUBARA`)
fn step_blocks(s: &GreenTimeStep, t: i64) -> Blocks {
    if t == MATSUBARA {
        return (0..=s.ntau as i64)
            .map(|tau| ("mat", tau, 0, s.mat.block(tau as usize).to_vec()))
            .collect();
    }
    let mut out = Vec::new();
    for tau in 0..=s.ntau as i64 {
        out.push(("lmix", t, tau, s.lmix.block(s.lmix.idx(0, tau)).to_vec()));
    }
    for j in 0..=t {
        out.push(("ret", t, j, s.ret.block(s.ret.idx(t, j)).to_vec()));
        out.push(("less", j, t, s.less.block(s.less.idx(j, t)).to_vec()));
    }
    out
}

/// the largest elementwise deviation of the non-square blocks from the
/// top-left `(r x c)` part of the `(n x n)` square ones, relative to the
/// scale of the block
fn deviation(square: &[Block], non_square: &[Block], n: usize, r: usize, c: usize) -> f64 {
    assert_eq!(square.len(), non_square.len(), "the block counts differ");
    let mut worst = 0.0f64;
    for ((na, ta, ja, xb), (nb, tb, jb, yb)) in square.iter().zip(non_square) {
        assert_eq!((na, ta, ja), (nb, tb, jb), "the block order differs");
        assert_eq!(xb.len() % (n * n), 0);
        assert_eq!(yb.len() % (r * c), 0);
        assert_eq!(
            xb.len() / (n * n),
            yb.len() / (r * c),
            "the block counts differ"
        );
        for (x, y) in xb.chunks(n * n).zip(yb.chunks(r * c)) {
            let scale = x
                .iter()
                .chain(y)
                .fold(0.0f64, |m, z| m.max(z.re.abs()).max(z.im.abs()))
                .max(1.0);
            for i in 0..r {
                for j in 0..c {
                    worst = worst.max((y[i * c + j] - x[i * n + j]).abs() / scale);
                }
            }
        }
    }
    worst
}

/// the operands of one problem: `C = A * f * B`
struct Case {
    contour: Contour,
    a: ContourGreen,
    acc: ContourGreen,
    f: Option<TimeFunction>,
    b: ContourGreen,
    bcc: ContourGreen,
    integ: Integrator,
}

/// every public entry point of the non-square convolution, as a list of
/// output blocks
fn entries() -> Vec<Entry> {
    vec![
        (
            "convolution",
            Box::new(|cs: &Case| {
                let mut c = ContourGreen::new(&cs.contour, Sign::Fermi, Complex::ZERO);
                convolution(
                    &mut c,
                    &cs.a,
                    &cs.acc,
                    cs.f.as_ref(),
                    &cs.b,
                    &cs.bcc,
                    &cs.integ,
                    NS_BETA,
                    NS_H,
                );
                blocks(&c)
            }),
        ),
        (
            "convolution_timestep",
            Box::new(|cs: &Case| {
                let mut c = ContourGreen::new(&cs.contour, Sign::Fermi, Complex::ZERO);
                for t in 0..=NS_NT {
                    convolution_timestep(
                        t,
                        &mut c,
                        &cs.a,
                        &cs.acc,
                        cs.f.as_ref(),
                        &cs.b,
                        &cs.bcc,
                        &cs.integ,
                        NS_BETA,
                        NS_H,
                        None,
                    );
                }
                blocks(&c)
            }),
        ),
        (
            "incr_convolution",
            Box::new(|cs: &Case| {
                let mut c = ContourGreen::new(&cs.contour, Sign::Fermi, Complex::ZERO);
                for t in MATSUBARA..=NS_NT {
                    incr_convolution(
                        t,
                        &mut c,
                        &cs.a,
                        &cs.acc,
                        cs.f.as_ref(),
                        &cs.b,
                        &cs.bcc,
                        &cs.integ,
                        NS_BETA,
                        NS_H,
                        Complex::ONE,
                    );
                }
                blocks(&c)
            }),
        ),
        (
            "incr_convolution_step",
            Box::new(|cs: &Case| {
                let mut out = Vec::new();
                for t in MATSUBARA..=NS_NT {
                    let mut s =
                        GreenTimeStep::new(&cs.contour, t.max(0), Sign::Fermi, Complex::ZERO);
                    incr_convolution_step(
                        t,
                        &mut s,
                        &cs.a,
                        &cs.acc,
                        cs.f.as_ref(),
                        &cs.b,
                        &cs.bcc,
                        &cs.integ,
                        NS_BETA,
                        NS_H,
                        Complex::ONE,
                    );
                    out.extend(step_blocks(&s, t));
                }
                out
            }),
        ),
        (
            "incr_convolution_mat",
            Box::new(|cs: &Case| {
                let mut c = ContourGreen::new(&cs.contour, Sign::Fermi, Complex::ZERO);
                incr_convolution_mat(
                    &mut c,
                    &cs.a,
                    cs.f.as_ref(),
                    &cs.b,
                    &cs.integ,
                    NS_BETA,
                    Complex::ONE,
                );
                blocks(&c)
            }),
        ),
        (
            "conv_lmix_mat",
            Box::new(|cs: &Case| {
                // the kernel without a middle function
                if cs.f.is_some() {
                    return Vec::new();
                }
                let mut c = ContourGreen::new(&cs.contour, Sign::Fermi, Complex::ZERO);
                for t in 0..=NS_NT {
                    conv_lmix_mat(
                        t,
                        &mut c.lmix,
                        &cs.a.lmix,
                        &cs.b.mat,
                        &cs.integ,
                        NS_BETA,
                        Sign::Fermi,
                    );
                }
                blocks(&c)
            }),
        ),
        (
            "incr_lmix_rmix_buf",
            Box::new(|cs: &Case| {
                let mut out = Vec::new();
                let f0 = cs.f.as_ref().map(|f| f.block(MATSUBARA));
                for t in 0..=NS_NT {
                    let mut buf =
                        vec![vec![Complex::ZERO; cs.a.size1 * cs.bcc.size1]; (t + 1) as usize];
                    incr_lmix_rmix_buf(
                        &mut buf,
                        t,
                        &cs.a,
                        f0,
                        &cs.bcc,
                        &cs.integ,
                        Sign::Fermi,
                        NS_BETA,
                        Complex::ONE,
                    );
                    for (m, blk) in buf.iter().enumerate() {
                        out.push(("lmix", t, m as i64, blk.clone()));
                    }
                }
                out
            }),
        ),
        (
            "incr_less_adv_buf",
            Box::new(|cs: &Case| {
                let mut out = Vec::new();
                let fop = match cs.f.as_ref() {
                    Some(f) => FOp::Cf(f),
                    None => FOp::None,
                };
                for diagcc in [true, false] {
                    let mut ws = ConvWorkspace::new(cs.a.size1, cs.a.size2, cs.b.size1, cs.b.size2);
                    for t in 0..=NS_NT {
                        let mut buf =
                            vec![vec![Complex::ZERO; cs.a.size1 * cs.bcc.size1]; (t + 1) as usize];
                        incr_less_adv_buf(
                            &mut buf,
                            t,
                            &cs.a,
                            &cs.acc,
                            &fop,
                            &cs.b,
                            &cs.bcc,
                            &cs.integ,
                            NS_BETA,
                            NS_H,
                            Complex::ONE,
                            &mut ws,
                            diagcc,
                        );
                        for (m, blk) in buf.iter().enumerate() {
                            out.push(("less", m as i64, t, blk.clone()));
                        }
                    }
                }
                out
            }),
        ),
    ]
}

/// the embedded-square identity of every entry point for one problem
/// `A (r x m) * f (m x nf) * B (nf x c)`
fn check(
    sa: &Source,
    sb: &Source,
    sf: Option<&Middle>,
    shape: (usize, usize, usize, usize),
    checked: &mut usize,
) {
    let (r, m, nf, c) = shape;
    let n = r.max(m).max(nf).max(c);
    let build = |embed: bool, k: i64| -> Case {
        // the embedded objects are `n x n`, the others have their own shapes
        let (ar, am, bfr, bfc) = if embed { (n, n, n, n) } else { (r, m, nf, c) };
        let (a, acc) = pair(sa, ar, am);
        let (b, bcc) = pair(sb, bfr, bfc);
        let f = sf.map(|src| {
            let (s1, s2) = if embed { (n, n) } else { (m, nf) };
            place_middle(src, s1, s2)
        });
        Case {
            contour: contour(a.size1, b.size2),
            a,
            acc,
            f,
            b,
            bcc,
            integ: Integrator::new(k),
        }
    };
    for k in [2i64, 4] {
        let (sq, ns) = (build(true, k), build(false, k));
        for (name, op) in entries() {
            let dev = deviation(&op(&sq), &op(&ns), n, r, c);
            assert!(
                dev < 1e-12,
                "A {r}x{m}, f {m}x{nf}, B {nf}x{c}, k {k}, {name}: deviation {dev:.2e}"
            );
            *checked += 1;
        }
    }
}

// ------------------------------------------------------------------
// tests
// ------------------------------------------------------------------

/// The NESSi reference inputs padded with zero rows / columns into
/// non-square objects: the padded problem has exactly the reference
/// solution in its top-left block.
#[test]
fn convolution_reference_nonsquare_padded() {
    let (a, b, middle) = (
        conv_green(0, 2).1,
        conv_green(1, 2).1,
        conv_middle_function(2),
    );
    // (rows of A, columns of A = rows of f, columns of f = rows of B,
    // columns of B)
    let shapes = [
        (2usize, 3usize, 3usize, 2usize),
        (3, 2, 2, 3),
        (4, 3, 3, 5),
        (2, 3, 5, 2),
        (5, 2, 4, 3),
        (2, 2, 3, 2),
    ];
    let filler = Complex::new(7.0, -3.0);
    let data = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("data")
        .join("test");
    for &(ra, inner, fc, cb) in &shapes {
        // the partner of a padded object is the padded partner
        let (pa, pb) = (embed(&a, ra, inner), embed(&b, fc, cb));
        let (pacc, pbcc) = (embed(&a, inner, ra), embed(&b, cb, fc));
        let contour = Contour::new(
            CONV_NT,
            CONV_NTAU,
            ra as i64,
            cb as i64,
            CONV_NT as f64 * CONV_H,
            CONV_BETA,
        );
        for typ in [1i64, 2] {
            // without `f` the inner sizes must agree
            if typ == 1 && inner != fc {
                continue;
            }
            for k in 1..=5i64 {
                let name = format!("convolution_type{typ}_size2_k{k}");
                let path = data.join(format!("{name}.dat"));
                let reference = read_ref(path.to_str().unwrap()).expect("read ref");
                assert_eq!(reference.metadata["case"], name);
                let integ = Integrator::new(k);
                let f = (typ == 2).then(|| embed_middle(&middle, inner, fc, filler));
                // the full driver, at every order
                let mut c = ContourGreen::new(&contour, Sign::Fermi, Complex::ZERO);
                convolution(
                    &mut c,
                    &pa,
                    &pacc,
                    f.as_ref(),
                    &pb,
                    &pbcc,
                    &integ,
                    CONV_BETA,
                    CONV_H,
                );
                let full = reference_green(&c, &reference, 1e-10, 1e-10);
                // the time-step driver and the incremental step kernels (they
                // share the step kernels), at two orders
                let (mut step, mut incr) = ([0.0f64; 4], [0.0f64; 4]);
                if k == 2 || k == 5 {
                    let mut c = ContourGreen::new(&contour, Sign::Fermi, Complex::ZERO);
                    for t in 0..=CONV_NT {
                        convolution_timestep(
                            t,
                            &mut c,
                            &pa,
                            &pacc,
                            f.as_ref(),
                            &pb,
                            &pbcc,
                            &integ,
                            CONV_BETA,
                            CONV_H,
                            None,
                        );
                    }
                    // the Matsubara component is left alone
                    step = reference_green(&c, &reference, 1e-10, 1e-10);
                    step[0] = 0.0;
                    for t in MATSUBARA..=CONV_NT {
                        let mut s =
                            GreenTimeStep::new(&contour, t.max(0), Sign::Fermi, Complex::ZERO);
                        incr_convolution_step(
                            t,
                            &mut s,
                            &pa,
                            &pacc,
                            f.as_ref(),
                            &pb,
                            &pbcc,
                            &integ,
                            CONV_BETA,
                            CONV_H,
                            Complex::ONE,
                        );
                        let e = reference_step(&s, t, &reference, 1e-10, 1e-10);
                        for q in 0..4 {
                            incr[q] = incr[q].max(e[q]);
                        }
                    }
                }
                for (driver, errs) in [("full", full), ("time step", step), ("incremental", incr)] {
                    for (slot, e) in errs.iter().enumerate() {
                        assert!(
                            *e < 1.0,
                            "{name} padded to A {ra}x{inner}, f {inner}x{fc}, B {fc}x{cb}, {driver} driver: component {slot} relative error {e:.3e}"
                        );
                    }
                }
            }
        }
    }
}

/// Every public entry point of the non-square convolution agrees with
/// the top-left block of the convolution of the embedded squares, for
/// every shape with block sizes in `1..=3`.
#[test]
fn convolution_nonsquare_matches_embedded_squares() {
    let mut state = 0x5eed_1234_abcd_0001u64;
    let sizes = [1usize, 2, 3];
    let mut checked = 0usize;
    // a square middle function: `A (r x m) * f (m x m) * B (m x c)`,
    // with and without `f`
    for &r in &sizes {
        for &m in &sizes {
            for &c in &sizes {
                if r == m && m == c {
                    continue;
                }
                for with_f in [false, true] {
                    let (sa, sb) = (source(r, m, &mut state), source(m, c, &mut state));
                    let sf = with_f.then(|| middle(m, m, &mut state));
                    check(&sa, &sb, sf.as_ref(), (r, m, m, c), &mut checked);
                }
            }
        }
    }
    // a non-square middle function: `A (r x m) * f (m x nf) * B (nf x c)`
    for &r in &sizes {
        for &m in &sizes {
            for &nf in &sizes {
                for &c in &sizes {
                    if nf == m {
                        continue;
                    }
                    let (sa, sb, sf) = (
                        source(r, m, &mut state),
                        source(nf, c, &mut state),
                        middle(m, nf, &mut state),
                    );
                    check(&sa, &sb, Some(&sf), (r, m, nf, c), &mut checked);
                }
            }
        }
    }
    assert!(checked > 0);
}
