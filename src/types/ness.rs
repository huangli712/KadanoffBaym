// Project : KadanoffBaym
// File    : src/types/ness.rs
// Purpose : the steady-state (NESS2) types (blueprint
//           julia/src/types/ness.jl, properties/ness.jl and the NESS2
//           parts of indexing/{getindex,layout}.jl)
//
// The steady-state functions depend on the relative time only. They are
// stored on a periodic grid of `Nft` points. The mode `m` (0-based,
// NESSi convention, used as the storage index here) corresponds to the
// time `t_m = m h` for `m < Nft/2` and `t_m = (m - Nft) h` otherwise,
// and to the frequency `omega_m` with `dw = pi/((Nft/2) h)`.
//
// The storage layout of `NessArray` is `(size1, size2, Nft)` with the
// mode running slowest, so that the orbital matrix of every mode is
// contiguous and the Fourier transforms run along the last dimension.

use crate::complex::Complex;

/// Domain of the NESS2 data (blueprint `FFTDomain`).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum FftDomain {
    Time,
    Freq,
}

/// Grid of the steady-state functions (blueprint `NessGrid`). It is an
/// immutable value and not a contour function.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct NessGrid {
    /// Number of the grid points, an even number >= 4.
    pub nft: i64,
    /// Time step.
    pub h: f64,
    /// Frequency step, `pi / ((Nft/2) h)`.
    pub dw: f64,
}

impl NessGrid {
    /// Constructor with the frequency step `pi / ((Nft/2) h)`
    /// (blueprint `NessGrid(Nft, h)`).
    pub fn new(nft: i64, h: f64) -> NessGrid {
        assert!(h > 0.0, "h must be positive, got {h}");
        NessGrid::new_full(nft, h, std::f64::consts::PI / ((nft / 2) as f64 * h))
    }

    /// Constructor with all three fields (blueprint
    /// `NessGrid(Nft, h, dw)`).
    pub fn new_full(nft: i64, h: f64, dw: f64) -> NessGrid {
        assert!(
            nft >= 4 && nft % 2 == 0,
            "Nft must be an even number >= 4, got {nft}"
        );
        assert!(h > 0.0, "h must be positive, got {h}");
        assert!(dw > 0.0, "dw must be positive, got {dw}");
        NessGrid { nft, h, dw }
    }

    /// A new grid with the time step `h`, the frequency step is updated
    /// accordingly (blueprint `set_h`).
    pub fn set_h(&self, h: f64) -> NessGrid {
        assert!(h > 0.0, "h must be positive, got {h}");
        NessGrid::new_full(self.nft, h, std::f64::consts::PI / ((self.nft / 2) as f64 * h))
    }

    /// A new grid with the frequency step `dw`, the time step is
    /// updated accordingly (blueprint `set_dw`).
    pub fn set_dw(&self, dw: f64) -> NessGrid {
        assert!(dw > 0.0, "dw must be positive, got {dw}");
        NessGrid::new_full(
            self.nft,
            std::f64::consts::PI / ((self.nft / 2) as f64 * dw),
            dw,
        )
    }

    /// Relative time of the signed mode `m` (0-based, periodic;
    /// blueprint `_time_mode`).
    pub(crate) fn time_mode(&self, m: i64) -> f64 {
        let n = m.rem_euclid(self.nft);
        if n < self.nft / 2 {
            n as f64 * self.h
        } else {
            (n - self.nft) as f64 * self.h
        }
    }

    /// Frequency of the signed mode `m` (0-based, periodic; blueprint
    /// `_freq_mode`).
    pub(crate) fn freq_mode(&self, m: i64) -> f64 {
        let n = m.rem_euclid(self.nft);
        if n < self.nft / 2 {
            n as f64 * self.dw
        } else {
            (n - self.nft) as f64 * self.dw
        }
    }

    /// Relative time of the storage mode `m` (0-based, `0 <= m < Nft`).
    /// The second half of the grid represents the negative times
    /// (blueprint `time_at`).
    pub fn time_at(&self, m: i64) -> f64 {
        assert!(
            (0..self.nft).contains(&m),
            "time_at: mode {m} out of 0..{}",
            self.nft
        );
        self.time_mode(m)
    }

    /// Frequency of the storage mode `m` (0-based). The second half of
    /// the grid represents the negative frequencies (blueprint
    /// `freq_at`).
    pub fn freq_at(&self, m: i64) -> f64 {
        assert!(
            (0..self.nft).contains(&m),
            "freq_at: mode {m} out of 0..{}",
            self.nft
        );
        self.freq_mode(m)
    }
}

/// Time and frequency data of a steady-state matrix function on the
/// NESS2 grid (blueprint `NessArray`). A plain numerical storage, not a
/// contour function.
#[derive(Clone)]
pub struct NessArray {
    pub nft: i64,
    pub size1: usize,
    pub size2: usize,
    /// Time data, the matrix of the mode `m` is `time[m*block..]`.
    pub time: Vec<Complex>,
    /// Frequency data, the matrix of the mode `m` is `freq[m*block..]`.
    pub freq: Vec<Complex>,
}

impl NessArray {
    /// Constructor with zero time and frequency data (blueprint
    /// `NessArray(Nft, size1)`).
    pub fn new(nft: i64, size1: usize) -> NessArray {
        assert!(nft >= 1, "Nft must be positive, got {nft}");
        assert!(size1 >= 1, "size1 must be positive, got {size1}");
        let block = size1 * size1;
        let total = nft as usize * block;
        NessArray {
            nft,
            size1,
            size2: size1,
            time: vec![Complex::ZERO; total],
            freq: vec![Complex::ZERO; total],
        }
    }

    fn block_len(&self) -> usize {
        self.size1 * self.size2
    }

    /// The matrix of the mode `m` (0-based) in the domain `dom`
    /// (blueprint `A[n, dom]` with the 1-based `n`).
    pub fn block(&self, dom: FftDomain, m: i64) -> &[Complex] {
        assert!(
            (0..self.nft).contains(&m),
            "mode {m} out of 0..{}",
            self.nft
        );
        let data = if dom == FftDomain::Time {
            &self.time
        } else {
            &self.freq
        };
        let b = self.block_len();
        &data[m as usize * b..(m as usize + 1) * b]
    }

    /// The mutable matrix of the mode `m` in the domain `dom`.
    pub fn block_mut(&mut self, dom: FftDomain, m: i64) -> &mut [Complex] {
        assert!(
            (0..self.nft).contains(&m),
            "mode {m} out of 0..{}",
            self.nft
        );
        let b = self.block_len();
        let data = if dom == FftDomain::Time {
            &mut self.time
        } else {
            &mut self.freq
        };
        &mut data[m as usize * b..(m as usize + 1) * b]
    }

    /// Zero one domain of the data (blueprint `zeros!(A, dom)`).
    pub fn zeros_domain(&mut self, dom: FftDomain) {
        let data = if dom == FftDomain::Time {
            &mut self.time
        } else {
            &mut self.freq
        };
        data.fill(Complex::ZERO);
    }
}

/// Hermitian steady-state Green's function on the NESS2 grid
/// (blueprint `NessHermMat`). Only the retarded and the lesser
/// components are stored, both in the time and in the frequency
/// domain. The lesser component at negative times is not reconstructed
/// automatically, the algorithms keep the relation
/// `G^<(-t) = -G^<(t)^dagger` explicitly.
#[derive(Clone)]
pub struct NessHermMat {
    pub grid: NessGrid,
    pub ret: NessArray,
    pub les: NessArray,
}

impl NessHermMat {
    /// Constructor with zero values (blueprint
    /// `NessHermMat(grid, size1)`).
    pub fn new(grid: NessGrid, size1: usize) -> NessHermMat {
        let ret = NessArray::new(grid.nft, size1);
        let les = NessArray::new(grid.nft, size1);
        NessHermMat { grid, ret, les }
    }

    /// Same grid and matrix sizes (blueprint `iscompatible`).
    pub fn is_compatible(&self, other: &NessHermMat) -> bool {
        self.grid == other.grid
            && self.ret.size1 == other.ret.size1
            && self.ret.size2 == other.ret.size2
            && self.ret.nft == other.ret.nft
    }

    /// A copy of the retarded matrix at the mode `m` (blueprint
    /// `get_ret`).
    pub fn get_ret(&self, m: i64, dom: FftDomain) -> Vec<Complex> {
        self.ret.block(dom, m).to_vec()
    }

    /// A copy of the lesser matrix at the mode `m` (blueprint
    /// `get_les`).
    pub fn get_les(&self, m: i64, dom: FftDomain) -> Vec<Complex> {
        self.les.block(dom, m).to_vec()
    }

    /// Copy the matrix `M` to the retarded component at the mode `m`
    /// (blueprint `set_ret!`).
    pub fn set_ret(&mut self, m: i64, dom: FftDomain, mm: &[Complex]) {
        self.ret.block_mut(dom, m).copy_from_slice(mm);
    }

    /// Copy the matrix `M` to the lesser component at the mode `m`
    /// (blueprint `set_les!`).
    pub fn set_les(&mut self, m: i64, dom: FftDomain, mm: &[Complex]) {
        self.les.block_mut(dom, m).copy_from_slice(mm);
    }

    /// Zero one domain of both components (blueprint `zeros!(G, dom)`).
    pub fn zeros_domain(&mut self, dom: FftDomain) {
        self.ret.zeros_domain(dom);
        self.les.zeros_domain(dom);
    }
}

// ------------------------------------------------------------------
// algebra of the NESS2 data (blueprint julia/src/traits/{incr,smul,
// memset,lmul}.jl, blocks/element.jl, properties/distance.jl)
// ------------------------------------------------------------------

impl NessArray {
    /// Add `alpha * B` to `self` in the domain `dom` (blueprint `incr!`
    /// on `NessArray`).
    pub fn incr(&mut self, b: &NessArray, alpha: Complex, dom: FftDomain) {
        assert!(
            self.time.len() == b.time.len(),
            "A and B must have the same size"
        );
        let (x, y) = if dom == FftDomain::Time {
            (&mut self.time, &b.time)
        } else {
            (&mut self.freq, &b.freq)
        };
        for q in 0..x.len() {
            x[q] += alpha * y[q];
        }
    }

    /// Multiply the data in the domain `dom` by the scalar `alpha`
    /// (blueprint `smul!` on `NessArray`).
    pub fn smul(&mut self, alpha: Complex, dom: FftDomain) {
        let x = if dom == FftDomain::Time {
            &mut self.time
        } else {
            &mut self.freq
        };
        for z in x.iter_mut() {
            *z *= alpha;
        }
    }

    /// Fill the data in the domain `dom` with the value `x` (blueprint
    /// `memset!` on `NessArray`).
    pub fn memset(&mut self, dom: FftDomain, x: Complex) {
        let data = if dom == FftDomain::Time {
            &mut self.time
        } else {
            &mut self.freq
        };
        data.fill(x);
    }

    /// Euclidean (Frobenius) distance to `b` in the domain `dom`
    /// (blueprint `distance`).
    pub fn distance(&self, b: &NessArray, dom: FftDomain) -> f64 {
        assert!(
            self.time.len() == b.time.len(),
            "A and B must have the same size"
        );
        let (x, y) = if dom == FftDomain::Time {
            (&self.time, &b.time)
        } else {
            (&self.freq, &b.freq)
        };
        let mut s = 0.0;
        for q in 0..x.len() {
            let dz = x[q] - y[q];
            s += dz.re * dz.re + dz.im * dz.im;
        }
        s.sqrt()
    }

    /// `self[i1, i2] <- B[k1, k2]` in every mode of the domain `dom`
    /// (blueprint `set_matrixelement!`).
    pub fn set_matrixelement(
        &mut self,
        i1: usize,
        i2: usize,
        b: &NessArray,
        k1: usize,
        k2: usize,
        dom: FftDomain,
    ) {
        assert!(
            self.size1 == b.size1 && self.size2 == b.size2 && self.nft == b.nft,
            "A and B must have the same size"
        );
        let d = self.size1;
        for m in 0..self.nft {
            let v = b.block(dom, m)[k1 * d + k2];
            self.block_mut(dom, m)[i1 * d + i2] = v;
        }
    }

    /// Left multiplication of every matrix of the domain `dom` by `M`
    /// (or `M^dagger` with `hermconj`; blueprint `left_multiply!` on
    /// `NessArray`).
    pub fn left_multiply(&mut self, m: &[Complex], dom: FftDomain, hermconj: bool) {
        let d = self.size1;
        assert!(m.len() == d * d, "the matrix must have the fitting size");
        for mode in 0..self.nft {
            let blk = self.block(dom, mode).to_vec();
            let out = self.block_mut(dom, mode);
            for i in 0..d {
                for j in 0..d {
                    let mut acc = Complex::ZERO;
                    for k in 0..d {
                        let mk = if hermconj {
                            m[k * d + i].conj()
                        } else {
                            m[i * d + k]
                        };
                        acc += mk * blk[k * d + j];
                    }
                    out[i * d + j] = acc;
                }
            }
        }
    }

    /// Right multiplication of every matrix of the domain `dom` by `M`
    /// (or `M^dagger` with `hermconj`; blueprint `right_multiply!` on
    /// `NessArray`).
    pub fn right_multiply(&mut self, m: &[Complex], dom: FftDomain, hermconj: bool) {
        let d = self.size1;
        assert!(m.len() == d * d, "the matrix must have the fitting size");
        for mode in 0..self.nft {
            let blk = self.block(dom, mode).to_vec();
            let out = self.block_mut(dom, mode);
            for i in 0..d {
                for j in 0..d {
                    let mut acc = Complex::ZERO;
                    for k in 0..d {
                        let mk = if hermconj {
                            m[j * d + k].conj()
                        } else {
                            m[k * d + j]
                        };
                        acc += blk[i * d + k] * mk;
                    }
                    out[i * d + j] = acc;
                }
            }
        }
    }
}

impl NessHermMat {
    /// Add `alpha * B` to `self` in the domain `dom` (blueprint `incr!`
    /// on `NessHermMat`).
    pub fn incr(&mut self, b: &NessHermMat, alpha: Complex, dom: FftDomain) {
        self.ret.incr(&b.ret, alpha, dom);
        self.les.incr(&b.les, alpha, dom);
    }

    /// Multiply the data in the domain `dom` by the scalar `alpha`
    /// (blueprint `smul!` on `NessHermMat`).
    pub fn smul(&mut self, alpha: Complex, dom: FftDomain) {
        self.ret.smul(alpha, dom);
        self.les.smul(alpha, dom);
    }

    /// Fill the data in the domain `dom` with the value `x` (blueprint
    /// `memset!` on `NessHermMat`).
    pub fn memset(&mut self, dom: FftDomain, x: Complex) {
        self.ret.memset(dom, x);
        self.les.memset(dom, x);
    }

    /// Euclidean distance `sqrt(d_R^2 + d_<^2)` to `b` in the domain
    /// `dom` (blueprint `distance`).
    pub fn distance(&self, b: &NessHermMat, dom: FftDomain) -> f64 {
        let x = self.ret.distance(&b.ret, dom);
        let y = self.les.distance(&b.les, dom);
        (x * x + y * y).sqrt()
    }

    /// `self[i1, i2] <- B[k1, k2]` in every mode of the domain `dom`
    /// for both components (blueprint `set_matrixelement!`).
    pub fn set_matrixelement(
        &mut self,
        i1: usize,
        i2: usize,
        b: &NessHermMat,
        k1: usize,
        k2: usize,
        dom: FftDomain,
    ) {
        self.ret.set_matrixelement(i1, i2, &b.ret, k1, k2, dom);
        self.les.set_matrixelement(i1, i2, &b.les, k1, k2, dom);
    }

    /// Left multiplication of both components (blueprint
    /// `left_multiply!` on `NessHermMat`).
    pub fn left_multiply(&mut self, m: &[Complex], dom: FftDomain, hermconj: bool) {
        self.ret.left_multiply(m, dom, hermconj);
        self.les.left_multiply(m, dom, hermconj);
    }

    /// Right multiplication of both components (blueprint
    /// `right_multiply!` on `NessHermMat`).
    pub fn right_multiply(&mut self, m: &[Complex], dom: FftDomain, hermconj: bool) {
        self.ret.right_multiply(m, dom, hermconj);
        self.les.right_multiply(m, dom, hermconj);
    }
}
