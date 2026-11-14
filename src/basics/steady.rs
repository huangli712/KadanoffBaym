// Project : KadanoffBaym
// File    : src/basics/steady.rs
// Purpose : operations on the steady-state (NESS2) containers
//           (blueprint julia/src/traits/{incr,smul,memset,lmul}.jl,
//           blocks/element.jl, properties/distance.jl, and the NESS2
//           parts of indexing/layout.jl)
//
// Split from types/steady.rs: the type definitions and constructors
// live there, the operation impl blocks live here.

use crate::Complex;
use crate::types::steady::{FftDomain, SteadyFunction, SteadyGrid, SteadyGreen};

impl SteadyGrid {
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

impl SteadyFunction {
    /// Add `alpha * B` to `self` in the domain `dom` (blueprint `incr!`
    /// on `NessArray`).
    pub fn incr(&mut self, b: &SteadyFunction, alpha: Complex, dom: FftDomain) {
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
    pub fn distance(&self, b: &SteadyFunction, dom: FftDomain) -> f64 {
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
        b: &SteadyFunction,
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
    /// `SteadyFunction`).
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
    /// `SteadyFunction`).
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

impl SteadyGreen {
    /// Add `alpha * B` to `self` in the domain `dom` (blueprint `incr!`
    /// on `NessHermMat`).
    pub fn incr(&mut self, b: &SteadyGreen, alpha: Complex, dom: FftDomain) {
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
    pub fn distance(&self, b: &SteadyGreen, dom: FftDomain) -> f64 {
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
        b: &SteadyGreen,
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
