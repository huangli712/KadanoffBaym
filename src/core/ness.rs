// Project : KadanoffBaym
// File    : src/core/ness.rs
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

use crate::complex::c64::C64;

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
}

/// Relative time of the signed mode `m` (0-based, periodic; blueprint
/// `_time_mode`).
pub fn time_mode(g: &NessGrid, m: i64) -> f64 {
    let n = m.rem_euclid(g.nft);
    if n < g.nft / 2 {
        n as f64 * g.h
    } else {
        (n - g.nft) as f64 * g.h
    }
}

/// Frequency of the signed mode `m` (0-based, periodic; blueprint
/// `_freq_mode`).
pub fn freq_mode(g: &NessGrid, m: i64) -> f64 {
    let n = m.rem_euclid(g.nft);
    if n < g.nft / 2 {
        n as f64 * g.dw
    } else {
        (n - g.nft) as f64 * g.dw
    }
}

/// Relative time of the storage mode `m` (0-based, `0 <= m < Nft`).
/// The second half of the grid represents the negative times
/// (blueprint `time_at`).
pub fn time_at(g: &NessGrid, m: i64) -> f64 {
    assert!((0..g.nft).contains(&m), "time_at: mode {m} out of 0..{}", g.nft);
    time_mode(g, m)
}

/// Frequency of the storage mode `m` (0-based). The second half of the
/// grid represents the negative frequencies (blueprint `freq_at`).
pub fn freq_at(g: &NessGrid, m: i64) -> f64 {
    assert!((0..g.nft).contains(&m), "freq_at: mode {m} out of 0..{}", g.nft);
    freq_mode(g, m)
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
    pub time: Vec<C64>,
    /// Frequency data, the matrix of the mode `m` is `freq[m*block..]`.
    pub freq: Vec<C64>,
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
            time: vec![C64::ZERO; total],
            freq: vec![C64::ZERO; total],
        }
    }

    fn block_len(&self) -> usize {
        self.size1 * self.size2
    }

    /// The matrix of the mode `m` (0-based) in the domain `dom`
    /// (blueprint `A[n, dom]` with the 1-based `n`).
    pub fn block(&self, dom: FftDomain, m: i64) -> &[C64] {
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
    pub fn block_mut(&mut self, dom: FftDomain, m: i64) -> &mut [C64] {
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
        data.fill(C64::ZERO);
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

    /// A copy of the retarded matrix at the mode `m` (blueprint
    /// `get_ret`).
    pub fn get_ret(&self, m: i64, dom: FftDomain) -> Vec<C64> {
        self.ret.block(dom, m).to_vec()
    }

    /// A copy of the lesser matrix at the mode `m` (blueprint
    /// `get_les`).
    pub fn get_les(&self, m: i64, dom: FftDomain) -> Vec<C64> {
        self.les.block(dom, m).to_vec()
    }

    /// Copy the matrix `M` to the retarded component at the mode `m`
    /// (blueprint `set_ret!`).
    pub fn set_ret(&mut self, m: i64, dom: FftDomain, mm: &[C64]) {
        self.ret.block_mut(dom, m).copy_from_slice(mm);
    }

    /// Copy the matrix `M` to the lesser component at the mode `m`
    /// (blueprint `set_les!`).
    pub fn set_les(&mut self, m: i64, dom: FftDomain, mm: &[C64]) {
        self.les.block_mut(dom, m).copy_from_slice(mm);
    }

    /// Zero one domain of both components (blueprint `zeros!(G, dom)`).
    pub fn zeros_domain(&mut self, dom: FftDomain) {
        self.ret.zeros_domain(dom);
        self.les.zeros_domain(dom);
    }

    /// Same grid and matrix sizes (blueprint `iscompatible`).
    pub fn is_compatible(&self, other: &NessHermMat) -> bool {
        self.grid == other.grid
            && self.ret.size1 == other.ret.size1
            && self.ret.size2 == other.ret.size2
            && self.ret.nft == other.ret.nft
    }
}

// ------------------------------------------------------------------
// algebra of the NESS2 data (blueprint julia/src/traits/{incr,smul,
// memset,lmul}.jl, blocks/element.jl, properties/distance.jl)
// ------------------------------------------------------------------

/// Add `alpha * B` to `A` in the domain `dom` (blueprint `incr!` on
/// `NessArray`).
pub fn incr_ness(a: &mut NessArray, b: &NessArray, alpha: C64, dom: FftDomain) {
    assert!(
        a.time.len() == b.time.len(),
        "A and B must have the same size"
    );
    let (x, y) = if dom == FftDomain::Time {
        (&mut a.time, &b.time)
    } else {
        (&mut a.freq, &b.freq)
    };
    for q in 0..x.len() {
        x[q] += alpha * y[q];
    }
}

/// Multiply the data of `A` in the domain `dom` by the scalar `alpha`
/// (blueprint `smul!` on `NessArray`).
pub fn smul_ness(a: &mut NessArray, alpha: C64, dom: FftDomain) {
    let x = if dom == FftDomain::Time {
        &mut a.time
    } else {
        &mut a.freq
    };
    for z in x.iter_mut() {
        *z *= alpha;
    }
}

/// Fill the data of `A` in the domain `dom` with the value `x`
/// (blueprint `memset!` on `NessArray`).
pub fn memset_ness(a: &mut NessArray, dom: FftDomain, x: C64) {
    let data = if dom == FftDomain::Time {
        &mut a.time
    } else {
        &mut a.freq
    };
    data.fill(x);
}

/// Euclidean (Frobenius) distance between the data of two `NessArray`
/// structs in the domain `dom` (blueprint `distance`).
pub fn distance_ness_array(a: &NessArray, b: &NessArray, dom: FftDomain) -> f64 {
    assert!(
        a.time.len() == b.time.len(),
        "A and B must have the same size"
    );
    let (x, y) = if dom == FftDomain::Time {
        (&a.time, &b.time)
    } else {
        (&a.freq, &b.freq)
    };
    let mut s = 0.0;
    for q in 0..x.len() {
        let dz = x[q] - y[q];
        s += dz.re * dz.re + dz.im * dz.im;
    }
    s.sqrt()
}

/// `A[i1, i2] <- B[k1, k2]` in every mode of the domain `dom`
/// (blueprint `set_matrixelement!`).
pub fn set_matrixelement(
    a: &mut NessArray,
    i1: usize,
    i2: usize,
    b: &NessArray,
    k1: usize,
    k2: usize,
    dom: FftDomain,
) {
    assert!(
        a.size1 == b.size1 && a.size2 == b.size2 && a.nft == b.nft,
        "A and B must have the same size"
    );
    let d = a.size1;
    for m in 0..a.nft {
        let v = b.block(dom, m)[k1 * d + k2];
        a.block_mut(dom, m)[i1 * d + i2] = v;
    }
}

/// Left multiplication of every matrix of the domain `dom` by `M` (or
/// `M^dagger` with `hermconj`; blueprint `left_multiply!` on
/// `NessArray`).
pub fn left_multiply_ness(a: &mut NessArray, m: &[C64], dom: FftDomain, hermconj: bool) {
    let d = a.size1;
    assert!(m.len() == d * d, "the matrix must have the fitting size");
    for mode in 0..a.nft {
        let blk = a.block(dom, mode).to_vec();
        let out = a.block_mut(dom, mode);
        for i in 0..d {
            for j in 0..d {
                let mut acc = C64::ZERO;
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
pub fn right_multiply_ness(a: &mut NessArray, m: &[C64], dom: FftDomain, hermconj: bool) {
    let d = a.size1;
    assert!(m.len() == d * d, "the matrix must have the fitting size");
    for mode in 0..a.nft {
        let blk = a.block(dom, mode).to_vec();
        let out = a.block_mut(dom, mode);
        for i in 0..d {
            for j in 0..d {
                let mut acc = C64::ZERO;
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

/// Add `alpha * B` to `A` in the domain `dom` (blueprint `incr!` on
/// `NessHermMat`).
pub fn incr_ness_mat(a: &mut NessHermMat, b: &NessHermMat, alpha: C64, dom: FftDomain) {
    incr_ness(&mut a.ret, &b.ret, alpha, dom);
    incr_ness(&mut a.les, &b.les, alpha, dom);
}

/// Multiply the data of `G` in the domain `dom` by the scalar `alpha`
/// (blueprint `smul!` on `NessHermMat`).
pub fn smul_ness_mat(g: &mut NessHermMat, alpha: C64, dom: FftDomain) {
    smul_ness(&mut g.ret, alpha, dom);
    smul_ness(&mut g.les, alpha, dom);
}

/// Fill the data of `G` in the domain `dom` with the value `x`
/// (blueprint `memset!` on `NessHermMat`).
pub fn memset_ness_mat(g: &mut NessHermMat, dom: FftDomain, x: C64) {
    memset_ness(&mut g.ret, dom, x);
    memset_ness(&mut g.les, dom, x);
}

/// Euclidean distance `sqrt(d_R^2 + d_<^2)` between two `NessHermMat`
/// structs in the domain `dom` (blueprint `distance`).
pub fn distance_ness(a: &NessHermMat, b: &NessHermMat, dom: FftDomain) -> f64 {
    let x = distance_ness_array(&a.ret, &b.ret, dom);
    let y = distance_ness_array(&a.les, &b.les, dom);
    (x * x + y * y).sqrt()
}

/// `G[i1, i2] <- B[k1, k2]` in every mode of the domain `dom` for
/// both components (blueprint `set_matrixelement!`).
pub fn set_matrixelement_mat(
    g: &mut NessHermMat,
    i1: usize,
    i2: usize,
    b: &NessHermMat,
    k1: usize,
    k2: usize,
    dom: FftDomain,
) {
    set_matrixelement(&mut g.ret, i1, i2, &b.ret, k1, k2, dom);
    set_matrixelement(&mut g.les, i1, i2, &b.les, k1, k2, dom);
}

/// Left multiplication of both components (blueprint `left_multiply!`
/// on `NessHermMat`).
pub fn left_multiply_ness_mat(g: &mut NessHermMat, m: &[C64], dom: FftDomain, hermconj: bool) {
    left_multiply_ness(&mut g.ret, m, dom, hermconj);
    left_multiply_ness(&mut g.les, m, dom, hermconj);
}

/// Right multiplication of both components (blueprint
/// `right_multiply!` on `NessHermMat`).
pub fn right_multiply_ness_mat(g: &mut NessHermMat, m: &[C64], dom: FftDomain, hermconj: bool) {
    right_multiply_ness(&mut g.ret, m, dom, hermconj);
    right_multiply_ness(&mut g.les, m, dom, hermconj);
}
