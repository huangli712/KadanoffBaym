// Project : KadanoffBaym
// File    : src/types/green.rs
// Purpose : component containers, ContourGreen, GreenTimeStep, TimeFunction
//
// Blueprint: julia/src/types/{green,function,mat,ret,less,lmix}.jl.
// Julia components store their own arrays (G* owned by ℱ, g* owned by
// 𝒻); the same layout is used here. Indices are 0-based (D6):
// tstp = 0..nt, tau = 0..ntau, Matsubara via tstp = MATSUBARA (-1).

use super::contour::{Contour, Sign};
use super::layout::{Layout, MATSUBARA};
use crate::complex::Complex;


/// Matsubara component: `ntau + 1` blocks, `G^M(tau_n)`.
#[derive(Clone, Debug, PartialEq)]
pub struct MatComp {
    pub ntau: usize,
    pub size1: usize,
    pub size2: usize,
    block: usize,
    data: Vec<Complex>,
}


        impl MatComp {
            /// Number of stored blocks.
            #[inline]
            pub fn nblocks(&self) -> usize {
                self.data.len() / self.block
            }

            /// Borrow the whole raw storage (blocks in row-major order).
            #[inline]
            pub fn raw(&self) -> &[Complex] {
                &self.data
            }

            /// Mutably borrow the whole raw storage.
            #[inline]
            pub fn raw_mut(&mut self) -> &mut [Complex] {
                &mut self.data
            }

            /// Borrow block `k` as a row-major `(size1 x size2)` slice.
            #[inline]
            pub fn block(&self, k: usize) -> &[Complex] {
                let b = self.block;
                &self.data[k * b..(k + 1) * b]
            }

            /// Mutably borrow block `k`.
            #[inline]
            pub fn block_mut(&mut self, k: usize) -> &mut [Complex] {
                let b = self.block;
                &mut self.data[k * b..(k + 1) * b]
            }

            /// Element access of block `k` at `(a, b_orb)` (0-based).
            #[inline]
            pub fn get(&self, k: usize, a: usize, b_orb: usize) -> Complex {
                self.data[k * self.block + a * self.size2 + b_orb]
            }

            /// Element write of block `k` at `(a, b_orb)`.
            #[inline]
            pub fn set(&mut self, k: usize, a: usize, b_orb: usize, v: Complex) {
                let i = k * self.block + a * self.size2 + b_orb;
                self.data[i] = v;
            }
        }
    

impl MatComp {
    pub fn new(ntau: usize, size1: usize, size2: usize, v: Complex) -> MatComp {
        let block = size1 * size2;
        MatComp {
            ntau,
            size1,
            size2,
            block,
            data: vec![v; (ntau + 1) * block],
        }
    }

    /// Validate a tau index and return it (0-based).
    #[inline]
    pub fn check_tau(&self, n: i64) -> usize {
        assert!(
            (0..=self.ntau as i64).contains(&n),
            "mat: tau index {n} out of range 0..={}",
            self.ntau
        );
        n as usize
    }
}

/// Retarded component: lower triangle, row `tstp` holds columns
/// `j = 0..=tstp`, block index `tri_off(tstp) + j`.
#[derive(Clone, Debug, PartialEq)]
pub struct RetComp {
    pub nt: usize,
    pub size1: usize,
    pub size2: usize,
    block: usize,
    data: Vec<Complex>,
}


        impl RetComp {
            /// Number of stored blocks.
            #[inline]
            pub fn nblocks(&self) -> usize {
                self.data.len() / self.block
            }

            /// Borrow the whole raw storage (blocks in row-major order).
            #[inline]
            pub fn raw(&self) -> &[Complex] {
                &self.data
            }

            /// Mutably borrow the whole raw storage.
            #[inline]
            pub fn raw_mut(&mut self) -> &mut [Complex] {
                &mut self.data
            }

            /// Borrow block `k` as a row-major `(size1 x size2)` slice.
            #[inline]
            pub fn block(&self, k: usize) -> &[Complex] {
                let b = self.block;
                &self.data[k * b..(k + 1) * b]
            }

            /// Mutably borrow block `k`.
            #[inline]
            pub fn block_mut(&mut self, k: usize) -> &mut [Complex] {
                let b = self.block;
                &mut self.data[k * b..(k + 1) * b]
            }

            /// Element access of block `k` at `(a, b_orb)` (0-based).
            #[inline]
            pub fn get(&self, k: usize, a: usize, b_orb: usize) -> Complex {
                self.data[k * self.block + a * self.size2 + b_orb]
            }

            /// Element write of block `k` at `(a, b_orb)`.
            #[inline]
            pub fn set(&mut self, k: usize, a: usize, b_orb: usize, v: Complex) {
                let i = k * self.block + a * self.size2 + b_orb;
                self.data[i] = v;
            }
        }
    

impl RetComp {
    /// Full triangle for `nt` (rows 0..=nt).
    pub fn new(nt: usize, size1: usize, size2: usize, v: Complex) -> RetComp {
        let block = size1 * size2;
        RetComp {
            nt,
            size1,
            size2,
            block,
            data: vec![v; Layout::tri_off(nt + 1) * block],
        }
    }

    /// Validate `(tstp, j)` with `0 <= j <= tstp <= nt` and return the
    /// block index.
    #[inline]
    pub fn idx(&self, tstp: i64, j: i64) -> usize {
        assert!(
            (0..=self.nt as i64).contains(&tstp),
            "ret: tstp {tstp} out of range 0..={}",
            self.nt
        );
        assert!(
            (0..=tstp).contains(&j),
            "ret: column {j} out of range 0..={tstp}"
        );
        Layout::tri_off(tstp as usize) + j as usize
    }

    /// Borrow the whole row `tstp` (blocks `j = 0..=tstp`).
    #[inline]
    pub fn row(&self, tstp: i64) -> &[Complex] {
        let k = Layout::tri_off(tstp as usize);
        let n = (tstp as usize + 1) * self.block;
        &self.data[k * self.block..k * self.block + n]
    }

    /// Mutably borrow the whole row `tstp`.
    #[inline]
    pub fn row_mut(&mut self, tstp: i64) -> &mut [Complex] {
        let k = Layout::tri_off(tstp as usize);
        let n = (tstp as usize + 1) * self.block;
        &mut self.data[k * self.block..k * self.block + n]
    }
}

/// Lesser component: mirror triangle, row `tstp` (the larger time)
/// holds columns `i = 0..=tstp`, block index `tri_off(tstp) + i`.
#[derive(Clone, Debug, PartialEq)]
pub struct LessComp {
    pub nt: usize,
    pub size1: usize,
    pub size2: usize,
    block: usize,
    data: Vec<Complex>,
}


        impl LessComp {
            /// Number of stored blocks.
            #[inline]
            pub fn nblocks(&self) -> usize {
                self.data.len() / self.block
            }

            /// Borrow the whole raw storage (blocks in row-major order).
            #[inline]
            pub fn raw(&self) -> &[Complex] {
                &self.data
            }

            /// Mutably borrow the whole raw storage.
            #[inline]
            pub fn raw_mut(&mut self) -> &mut [Complex] {
                &mut self.data
            }

            /// Borrow block `k` as a row-major `(size1 x size2)` slice.
            #[inline]
            pub fn block(&self, k: usize) -> &[Complex] {
                let b = self.block;
                &self.data[k * b..(k + 1) * b]
            }

            /// Mutably borrow block `k`.
            #[inline]
            pub fn block_mut(&mut self, k: usize) -> &mut [Complex] {
                let b = self.block;
                &mut self.data[k * b..(k + 1) * b]
            }

            /// Element access of block `k` at `(a, b_orb)` (0-based).
            #[inline]
            pub fn get(&self, k: usize, a: usize, b_orb: usize) -> Complex {
                self.data[k * self.block + a * self.size2 + b_orb]
            }

            /// Element write of block `k` at `(a, b_orb)`.
            #[inline]
            pub fn set(&mut self, k: usize, a: usize, b_orb: usize, v: Complex) {
                let i = k * self.block + a * self.size2 + b_orb;
                self.data[i] = v;
            }
        }
    

impl LessComp {
    pub fn new(nt: usize, size1: usize, size2: usize, v: Complex) -> LessComp {
        let block = size1 * size2;
        LessComp {
            nt,
            size1,
            size2,
            block,
            data: vec![v; Layout::tri_off(nt + 1) * block],
        }
    }

    /// Validate `(i, tstp)` with `0 <= i <= tstp <= nt` and return the
    /// block index.
    #[inline]
    pub fn idx(&self, i: i64, tstp: i64) -> usize {
        assert!(
            (0..=self.nt as i64).contains(&tstp),
            "less: tstp {tstp} out of range 0..={}",
            self.nt
        );
        assert!(
            (0..=tstp).contains(&i),
            "less: column {i} out of range 0..={tstp}"
        );
        Layout::tri_off(tstp as usize) + i as usize
    }

    /// Borrow the whole row `tstp` (blocks `i = 0..=tstp`).
    #[inline]
    pub fn row(&self, tstp: i64) -> &[Complex] {
        let k = Layout::tri_off(tstp as usize);
        let n = (tstp as usize + 1) * self.block;
        &self.data[k * self.block..k * self.block + n]
    }

    /// Mutably borrow the whole row `tstp`.
    #[inline]
    pub fn row_mut(&mut self, tstp: i64) -> &mut [Complex] {
        let k = Layout::tri_off(tstp as usize);
        let n = (tstp as usize + 1) * self.block;
        &mut self.data[k * self.block..k * self.block + n]
    }
}

/// Mixed component: flat `(nt + 1) x (ntau + 1)` blocks,
/// `G^mix(tstp, tau_n)`.
#[derive(Clone, Debug, PartialEq)]
pub struct LmixComp {
    pub nt: usize,
    pub ntau: usize,
    pub size1: usize,
    pub size2: usize,
    block: usize,
    data: Vec<Complex>,
}


        impl LmixComp {
            /// Number of stored blocks.
            #[inline]
            pub fn nblocks(&self) -> usize {
                self.data.len() / self.block
            }

            /// Borrow the whole raw storage (blocks in row-major order).
            #[inline]
            pub fn raw(&self) -> &[Complex] {
                &self.data
            }

            /// Mutably borrow the whole raw storage.
            #[inline]
            pub fn raw_mut(&mut self) -> &mut [Complex] {
                &mut self.data
            }

            /// Borrow block `k` as a row-major `(size1 x size2)` slice.
            #[inline]
            pub fn block(&self, k: usize) -> &[Complex] {
                let b = self.block;
                &self.data[k * b..(k + 1) * b]
            }

            /// Mutably borrow block `k`.
            #[inline]
            pub fn block_mut(&mut self, k: usize) -> &mut [Complex] {
                let b = self.block;
                &mut self.data[k * b..(k + 1) * b]
            }

            /// Element access of block `k` at `(a, b_orb)` (0-based).
            #[inline]
            pub fn get(&self, k: usize, a: usize, b_orb: usize) -> Complex {
                self.data[k * self.block + a * self.size2 + b_orb]
            }

            /// Element write of block `k` at `(a, b_orb)`.
            #[inline]
            pub fn set(&mut self, k: usize, a: usize, b_orb: usize, v: Complex) {
                let i = k * self.block + a * self.size2 + b_orb;
                self.data[i] = v;
            }
        }
    

impl LmixComp {
    pub fn new(nt: usize, ntau: usize, size1: usize, size2: usize, v: Complex) -> LmixComp {
        let block = size1 * size2;
        LmixComp {
            nt,
            ntau,
            size1,
            size2,
            block,
            data: vec![v; (nt + 1) * (ntau + 1) * block],
        }
    }

    /// Validate `(tstp, n)` and return the block index.
    #[inline]
    pub fn idx(&self, tstp: i64, n: i64) -> usize {
        assert!(
            (0..=self.nt as i64).contains(&tstp),
            "lmix: tstp {tstp} out of range 0..={}",
            self.nt
        );
        assert!(
            (0..=self.ntau as i64).contains(&n),
            "lmix: tau index {n} out of range 0..={}",
            self.ntau
        );
        tstp as usize * (self.ntau + 1) + n as usize
    }
}

/// Full contour-ordered Green's function (blueprint `ℱ`): four owned
/// components on an L-shaped contour.
#[derive(Clone, Debug, PartialEq)]
pub struct ContourGreen {
    pub sign: Sign,
    pub nt: usize,
    pub ntau: usize,
    pub size1: usize,
    pub size2: usize,
    pub mat: MatComp,
    pub ret: RetComp,
    pub lmix: LmixComp,
    pub less: LessComp,
}

impl ContourGreen {
    /// Standard constructor (blueprint `ℱ(C, v, sign)`): every element
    /// initialized to `v`.
    pub fn new(c: &Contour, sign: Sign, v: Complex) -> ContourGreen {
        let (nt, ntau) = (c.nt as usize, c.ntau as usize);
        let (s1, s2) = (c.size1 as usize, c.size2 as usize);
        ContourGreen {
            sign,
            nt,
            ntau,
            size1: s1,
            size2: s2,
            mat: MatComp::new(ntau, s1, s2, v),
            ret: RetComp::new(nt, s1, s2, v),
            lmix: LmixComp::new(nt, ntau, s1, s2, v),
            less: LessComp::new(nt, s1, s2, v),
        }
    }

    /// Zero-initialized constructor with default sign (FERMI).
    pub fn zeros(c: &Contour) -> ContourGreen {
        ContourGreen::new(c, Sign::Fermi, Complex::ZERO)
    }

    /// The underlying `Layout`.
    #[inline]
    pub fn layout(&self) -> Layout {
        Layout::new(self.nt, self.ntau, self.size1, self.size2)
    }

    /// Total number of stored f64 words (memory gate check).
    pub fn nwords(&self) -> usize {
        (self.mat.raw().len()
            + self.ret.raw().len()
            + self.lmix.raw().len()
            + self.less.raw().len())
            * 2
    }
}

/// Single-timestep Green's function (blueprint `𝒻`): mat (`ntau + 1`),
/// ret and less triangles up to `tstp`, and the lmix row of `tstp`.
#[derive(Clone, Debug, PartialEq)]
pub struct GreenTimeStep {
    pub sign: Sign,
    pub tstp: usize,
    pub ntau: usize,
    pub size1: usize,
    pub size2: usize,
    pub mat: MatComp,
    pub ret: RetComp,
    pub lmix: LmixComp,
    pub less: LessComp,
}

impl GreenTimeStep {
    /// Standard constructor (blueprint `𝒻(C, tstp, v, sign)`).
    pub fn new(c: &Contour, tstp: i64, sign: Sign, v: Complex) -> GreenTimeStep {
        assert!(
            (0..=c.nt).contains(&tstp),
            "GreenTimeStep: tstp {tstp} out of range 0..={}",
            c.nt
        );
        let (tstp, ntau) = (tstp as usize, c.ntau as usize);
        let (s1, s2) = (c.size1 as usize, c.size2 as usize);
        GreenTimeStep {
            sign,
            tstp,
            ntau,
            size1: s1,
            size2: s2,
            mat: MatComp::new(ntau, s1, s2, v),
            ret: RetComp::new(tstp, s1, s2, v),
            lmix: LmixComp::new(0, ntau, s1, s2, v), // single row
            less: LessComp::new(tstp, s1, s2, v),
        }
    }
}

/// Single-time function (blueprint `Cf`): one matrix block per real-time
/// step `0..=nt` plus one Matsubara slot.
///
/// Storage: `data[0..=nt]` are the time rows, `data[nt + 1]` is the
/// Matsubara value (Julia `cf[0]` <-> `get(MATSUBARA)`).
#[derive(Clone, Debug, PartialEq)]
pub struct TimeFunction {
    pub nt: usize,
    pub size1: usize,
    pub size2: usize,
    block: usize,
    data: Vec<Complex>,
}

impl TimeFunction {
    /// Constructor (blueprint `Cf(ntime, ndim1, ndim2, v)` with the -1
    /// translation): `nt >= 0`, all elements set to `v`.
    pub fn new(nt: usize, size1: usize, size2: usize, v: Complex) -> TimeFunction {
        let block = size1 * size2;
        TimeFunction {
            nt,
            size1,
            size2,
            block,
            data: vec![v; (nt + 2) * block],
        }
    }

    /// Zero-initialized constructor.
    pub fn zeros(nt: usize, size1: usize, size2: usize) -> TimeFunction {
        TimeFunction::new(nt, size1, size2, Complex::ZERO)
    }

    /// Block index for timestep `tstp` (`0..=nt` or `MATSUBARA`).
    #[inline]
    pub fn idx(&self, tstp: i64) -> usize {
        if tstp == MATSUBARA {
            self.nt + 1
        } else {
            assert!(
                (0..=self.nt as i64).contains(&tstp),
                "TimeFunction: tstp {tstp} out of range 0..={}",
                self.nt
            );
            tstp as usize
        }
    }

    /// Element access at `(tstp, a, b)` (0-based; `MATSUBARA` for the
    /// Matsubara slot).
    #[inline]
    pub fn get(&self, tstp: i64, a: usize, b: usize) -> Complex {
        self.data[self.idx(tstp) * self.block + a * self.size2 + b]
    }

    /// Element write at `(tstp, a, b)`.
    #[inline]
    pub fn set(&mut self, tstp: i64, a: usize, b: usize, v: Complex) {
        let i = self.idx(tstp) * self.block + a * self.size2 + b;
        self.data[i] = v;
    }

    /// Borrow the block of `tstp` as a row-major slice.
    #[inline]
    pub fn block(&self, tstp: i64) -> &[Complex] {
        let k = self.idx(tstp);
        &self.data[k * self.block..(k + 1) * self.block]
    }

    /// Mutably borrow the block of `tstp`.
    #[inline]
    pub fn block_mut(&mut self, tstp: i64) -> &mut [Complex] {
        let k = self.idx(tstp);
        &mut self.data[k * self.block..(k + 1) * self.block]
    }

    /// Borrow the whole raw storage (time rows then Matsubara slot).
    #[inline]
    pub fn raw(&self) -> &[Complex] {
        &self.data
    }

    /// Mutably borrow the whole raw storage.
    #[inline]
    pub fn raw_mut(&mut self) -> &mut [Complex] {
        &mut self.data
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn small_contour() -> Contour {
        Contour::new(3, 4, 2, 2, 1.0, 2.0)
    }

    #[test]
    fn component_sizes() {
        let c = small_contour();
        let g = ContourGreen::new(&c, Sign::Fermi, Complex::ZERO);
        assert_eq!(g.mat.nblocks(), 5);
        assert_eq!(g.ret.nblocks(), 10); // 1+2+3+4
        assert_eq!(g.less.nblocks(), 10);
        assert_eq!(g.lmix.nblocks(), 4 * 5);
        // Each block is 2x2 = 4 elements.
        assert_eq!(g.ret.raw().len(), 40);
        // mat 20 + ret 40 + lmix 80 + less 40 = 180 elements = 360 words
        assert_eq!(g.nwords(), 360);
    }

    #[test]
    fn triangle_index_matches_offsets() {
        let r = RetComp::new(4, 1, 1, Complex::ZERO);
        assert_eq!(r.idx(0, 0), 0);
        assert_eq!(r.idx(1, 0), 1);
        assert_eq!(r.idx(1, 1), 2);
        assert_eq!(r.idx(4, 4), 14);
        let l = LessComp::new(4, 1, 1, Complex::ZERO);
        assert_eq!(l.idx(1, 2), 4);
        assert_eq!(l.idx(0, 4), 10);
    }

    #[test]
    fn row_slices_span_the_row() {
        let mut r = RetComp::new(3, 1, 2, Complex::ZERO);
        assert_eq!(r.row(0).len(), 2);
        assert_eq!(r.row(3).len(), 8);
        r.row_mut(2)[0] = Complex::ONE;
        assert_eq!(r.get(r.idx(2, 0), 0, 0), Complex::ONE);
    }

    #[test]
    fn timestep_object_sizes() {
        let c = small_contour();
        let f = GreenTimeStep::new(&c, 2, Sign::Bose, Complex::ZERO);
        assert_eq!(f.tstp, 2);
        assert_eq!(f.mat.nblocks(), 5);
        assert_eq!(f.ret.nblocks(), 6); // 1+2+3
        assert_eq!(f.less.nblocks(), 6);
        assert_eq!(f.lmix.nblocks(), 5); // single row
    }

    #[test]
    fn time_function_matsubara_slot() {
        let mut cf = TimeFunction::new(3, 1, 1, Complex::ZERO);
        assert_eq!(cf.block(0).len(), 1);
        cf.set(MATSUBARA, 0, 0, Complex::new(9.0, 0.0));
        cf.set(2, 0, 0, Complex::new(2.0, 0.0));
        assert_eq!(cf.get(MATSUBARA, 0, 0), Complex::new(9.0, 0.0));
        assert_eq!(cf.get(2, 0, 0), Complex::new(2.0, 0.0));
        assert_eq!(cf.block(MATSUBARA)[0], Complex::new(9.0, 0.0));
    }

    #[test]
    #[should_panic(expected = "out of range")]
    fn ret_rejects_upper_triangle() {
        let r = RetComp::new(2, 1, 1, Complex::ZERO);
        let _ = r.idx(0, 1);
    }

    #[test]
    #[should_panic(expected = "out of range")]
    fn lmix_rejects_bad_tau() {
        let l = LmixComp::new(2, 3, 1, 1, Complex::ZERO);
        let _ = l.idx(0, 4);
    }
}

// Two-index raw block accessors (stored domain only). MatComp uses its
// single-index `block(m)`.
impl RetComp {
    /// Borrow the stored block `ret(i, j)` (`j <= i`).
    #[inline]
    pub fn raw_block_at(&self, i: i64, j: i64) -> &[Complex] {
        self.block(self.idx(i, j) as usize)
    }

    /// Mutably borrow the stored block `ret(i, j)` (`j <= i`).
    #[inline]
    pub fn raw_block_at_mut(&mut self, i: i64, j: i64) -> &mut [Complex] {
        let k = self.idx(i, j) as usize;
        self.block_mut(k)
    }
}

impl LessComp {
    /// Borrow the stored block `less(i, j)` (`i <= j`).
    #[inline]
    pub fn raw_block_at(&self, i: i64, j: i64) -> &[Complex] {
        self.block(self.idx(i, j) as usize)
    }

    /// Mutably borrow the stored block `less(i, j)` (`i <= j`).
    #[inline]
    pub fn raw_block_at_mut(&mut self, i: i64, j: i64) -> &mut [Complex] {
        let k = self.idx(i, j) as usize;
        self.block_mut(k)
    }
}

impl LmixComp {
    /// Borrow the stored block `lmix(i, j)`.
    #[inline]
    pub fn raw_block_at(&self, i: i64, j: i64) -> &[Complex] {
        self.block(self.idx(i, j) as usize)
    }

    /// Mutably borrow the stored block `lmix(i, j)`.
    #[inline]
    pub fn raw_block_at_mut(&mut self, i: i64, j: i64) -> &mut [Complex] {
        let k = self.idx(i, j) as usize;
        self.block_mut(k)
    }
}
