// Project : KadanoffBaym
// File    : src/basics/steady.rs
// Purpose : operations on the steady-state containers

use crate::complex::Complex;
use crate::types::{FftDomain, SteadyFunction, SteadyGreen, SteadyGrid};

impl SteadyGrid {
    /// Relative time of the storage mode `m` (0-based, `0 <= m < Nft`).
    /// The second half of the grid represents the negative times.
    pub fn time_at(&self, m: i64) -> f64 {
        assert!(
            (0..self.nft).contains(&m),
            "time_at: mode {m} out of 0..{}",
            self.nft
        );
        self.time_mode(m)
    }

    /// Frequency of the storage mode `m` (0-based). The second half of
    /// the grid represents the negative frequencies.
    pub fn freq_at(&self, m: i64) -> f64 {
        assert!(
            (0..self.nft).contains(&m),
            "freq_at: mode {m} out of 0..{}",
            self.nft
        );
        self.freq_mode(m)
    }

    /// Relative time of the signed mode `m` (0-based, periodic).
    pub(crate) fn time_mode(&self, m: i64) -> f64 {
        let n = m.rem_euclid(self.nft);
        if n < self.nft / 2 {
            n as f64 * self.h
        } else {
            (n - self.nft) as f64 * self.h
        }
    }

    /// Frequency of the signed mode `m` (0-based, periodic).
    pub(crate) fn freq_mode(&self, m: i64) -> f64 {
        let n = m.rem_euclid(self.nft);
        if n < self.nft / 2 {
            n as f64 * self.dw
        } else {
            (n - self.nft) as f64 * self.dw
        }
    }
}

impl SteadyFunction {
    /// Fill the data in the domain `dom` with the value `x`.
    pub fn memset(&mut self, dom: FftDomain, x: Complex) {
        let data = if dom == FftDomain::Time {
            &mut self.time
        } else {
            &mut self.freq
        };
        data.fill(x);
    }

    /// Add `alpha * B` to `self` in the domain `dom`.
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

    /// Multiply the data in the domain `dom` by the scalar `alpha`.
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

    /// `self[i1, i2] <- b[k1, k2]` in every mode of the domain `dom`.
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
    /// (or `M^dagger` with `hermconj`).
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
    /// (or `M^dagger` with `hermconj`).
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

    /// Euclidean (Frobenius) distance to `b` in the domain `dom`.
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
}

impl SteadyGreen {
    /// Fill the data in the domain `dom` with the value `x`.
    pub fn memset(&mut self, dom: FftDomain, x: Complex) {
        self.ret.memset(dom, x);
        self.les.memset(dom, x);
    }

    /// Add `alpha * B` to `self` in the domain `dom`.
    pub fn incr(&mut self, b: &SteadyGreen, alpha: Complex, dom: FftDomain) {
        self.ret.incr(&b.ret, alpha, dom);
        self.les.incr(&b.les, alpha, dom);
    }

    /// Multiply the data in the domain `dom` by the scalar `alpha`.
    pub fn smul(&mut self, alpha: Complex, dom: FftDomain) {
        self.ret.smul(alpha, dom);
        self.les.smul(alpha, dom);
    }

    /// `self[i1, i2] <- b[k1, k2]` in every mode of the domain `dom`
    /// for both components.
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

    /// Left multiplication of both components.
    pub fn left_multiply(&mut self, m: &[Complex], dom: FftDomain, hermconj: bool) {
        self.ret.left_multiply(m, dom, hermconj);
        self.les.left_multiply(m, dom, hermconj);
    }

    /// Right multiplication of both components.
    pub fn right_multiply(&mut self, m: &[Complex], dom: FftDomain, hermconj: bool) {
        self.ret.right_multiply(m, dom, hermconj);
        self.les.right_multiply(m, dom, hermconj);
    }

    /// Frobenius distance `sqrt(d_R^2 + d_<^2)` to `b` in the domain `dom`.
    pub fn distance(&self, b: &SteadyGreen, dom: FftDomain) -> f64 {
        let x = self.ret.distance(&b.ret, dom);
        let y = self.les.distance(&b.les, dom);
        (x * x + y * y).sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::panic::AssertUnwindSafe;

    use std::panic::catch_unwind;

    fn panics<F: FnOnce() -> R, R>(f: F) -> bool {
        catch_unwind(AssertUnwindSafe(f)).is_err()
    }

    #[test]
    fn steady_function_layout() {
        let a = SteadyFunction::new(16, 2);
        assert!(a.nft == 16 && a.size1 == 2 && a.size2 == 2);
        assert_eq!(a.time.len(), 2 * 2 * 16);
        assert_eq!(a.freq.len(), 2 * 2 * 16);
    }

    #[test]
    fn steady_array_elements() {
        let mut a = SteadyFunction::new(4, 2);
        let m = vec![Complex::ONE; 4];
        a.block_mut(FftDomain::Time, 0).copy_from_slice(&m);
        assert_eq!(a.block(FftDomain::Time, 0), &m[..]);
        // copies on get and set, no aliasing
        let mut x = a.block(FftDomain::Time, 0).to_vec();
        x[0] = Complex::new(5.0, 0.0);
        assert_eq!(a.block(FftDomain::Time, 0), &m[..]);
        let mut m2 = m.clone();
        m2[3] = Complex::new(7.0, 0.0);
        assert_eq!(a.block(FftDomain::Time, 0)[3], Complex::ONE);
        // scalar
        let mut s = SteadyFunction::new(4, 1);
        s.block_mut(FftDomain::Time, 0)[0] = Complex::new(1.23, 4.56);
        assert_eq!(s.block(FftDomain::Time, 0)[0], Complex::new(1.23, 4.56));
        // wrong sizes and out-of-range modes
        assert!(panics(|| a
            .block_mut(FftDomain::Time, 0)
            .copy_from_slice(&[Complex::ONE])));
        assert!(panics(|| a
            .block_mut(FftDomain::Time, 0)
            .copy_from_slice(&[Complex::ZERO; 9])));
        assert!(panics(|| a.block(FftDomain::Time, 4)));
        // set_matrixelement
        let mut b = SteadyFunction::new(4, 2);
        for t in 0..4 {
            let blk = [
                Complex::ZERO,
                Complex::new(1.0, 1.0),
                Complex::ZERO,
                Complex::ZERO,
            ];
            b.block_mut(FftDomain::Time, t).copy_from_slice(&blk);
        }
        let mut a2 = SteadyFunction::new(4, 2);
        a2.set_matrixelement(1, 0, &b, 0, 1, FftDomain::Time);
        for t in 0..4 {
            assert_eq!(a2.block(FftDomain::Time, t)[2], Complex::new(1.0, 1.0));
        }
        assert!(panics(|| a2.set_matrixelement(
            1,
            0,
            &SteadyFunction::new(8, 2),
            0,
            1,
            FftDomain::Time
        )));
    }

    #[test]
    fn steady_array_ops() {
        let mut a = SteadyFunction::new(4, 2);
        let mut b = SteadyFunction::new(4, 2);
        for t in 0..4 {
            for i in 0..2usize {
                for j in 0..2usize {
                    b.block_mut(FftDomain::Time, t)[i * 2 + j] =
                        Complex::new(t as f64 + i as f64 + j as f64 + (i * j) as f64, 0.0);
                }
            }
        }
        a.zeros_domain(FftDomain::Time);
        a.incr(&b, Complex::new(2.0, 0.0), FftDomain::Time);
        for q in 0..a.time.len() {
            assert!((a.time[q] - 2.0 * b.time[q]).abs() < 1e-12);
        }
        a.smul(Complex::new(0.5, 0.0), FftDomain::Time);
        for q in 0..a.time.len() {
            assert!((a.time[q] - b.time[q]).abs() < 1e-12);
        }
        // the frequency data is not touched
        assert!(a.freq.iter().all(|z| *z == Complex::ZERO));
        // left and right multiplication by the swap matrix
        for t in 0..4 {
            for i in 0..2usize {
                for j in 0..2usize {
                    a.block_mut(FftDomain::Time, t)[i * 2 + j] =
                        Complex::new(1.0 + i as f64 + j as f64, t as f64);
                }
            }
        }
        let mswap = [Complex::ZERO, Complex::ONE, Complex::ONE, Complex::ZERO];
        let mut cc = a.clone();
        cc.left_multiply(&mswap, FftDomain::Time, false);
        cc.left_multiply(&mswap, FftDomain::Time, false);
        for q in 0..cc.time.len() {
            assert!((cc.time[q] - a.time[q]).abs() < 1e-12);
        }
        cc.right_multiply(&mswap, FftDomain::Time, false);
        cc.right_multiply(&mswap, FftDomain::Time, false);
        for q in 0..cc.time.len() {
            assert!((cc.time[q] - a.time[q]).abs() < 1e-12);
        }
        let p = [
            Complex::ONE,
            Complex::new(0.0, 2.0),
            Complex::ZERO,
            Complex::new(3.0, 0.0),
        ];
        cc.left_multiply(&p, FftDomain::Time, false);
        // C[3] = P * A[3]
        let src = a.block(FftDomain::Time, 2).to_vec();
        let got = cc.block(FftDomain::Time, 2);
        for i in 0..2 {
            for j in 0..2 {
                let want = p[i * 2] * src[j] + p[i * 2 + 1] * src[2 + j];
                assert!((got[i * 2 + j] - want).abs() < 1e-12);
            }
        }
        let mut cc = a.clone();
        cc.right_multiply(&p, FftDomain::Time, true);
        // C[2] = A[2] * P'
        let src = a.block(FftDomain::Time, 1).to_vec();
        let got = cc.block(FftDomain::Time, 1);
        for i in 0..2 {
            for j in 0..2 {
                let want = src[i * 2] * p[j * 2].conj() + src[i * 2 + 1] * p[j * 2 + 1].conj();
                assert!((got[i * 2 + j] - want).abs() < 1e-12);
            }
        }
        let mut cc = a.clone();
        cc.left_multiply(&p, FftDomain::Time, true);
        let src = a.block(FftDomain::Time, 1).to_vec();
        let got = cc.block(FftDomain::Time, 1);
        for i in 0..2 {
            for j in 0..2 {
                let want = p[i].conj() * src[j] + p[2 + i].conj() * src[2 + j];
                assert!((got[i * 2 + j] - want).abs() < 1e-12);
            }
        }
        assert!(panics(|| cc.left_multiply(
            &[Complex::ZERO; 9],
            FftDomain::Time,
            false
        )));
        // memset / zeros / memcpy / distance
        a.memset(FftDomain::Freq, Complex::new(2.0, 0.0));
        assert!(a.freq.iter().all(|z| *z == Complex::new(2.0, 0.0)));
        a.zeros_domain(FftDomain::Time);
        a.zeros_domain(FftDomain::Freq);
        assert!(
            a.time
                .iter()
                .chain(a.freq.iter())
                .all(|z| *z == Complex::ZERO)
        );
        let d = b.clone();
        assert_eq!(d.time, b.time);
        assert!(panics(|| a.incr(
            &SteadyFunction::new(8, 2),
            Complex::ONE,
            FftDomain::Time
        )));
        assert_eq!(a.distance(&a, FftDomain::Time), 0.0);
        let mut e = SteadyFunction::new(4, 2);
        e.time[2 * 4 + 1] = Complex::new(3.0, 4.0);
        assert_eq!(e.distance(&SteadyFunction::new(4, 2), FftDomain::Time), 5.0);
    }

    #[test]
    fn steady_herm_mat_elements_and_ops() {
        let mut g = SteadyGreen::new(SteadyGrid::new(16, 0.1), 2);
        assert_eq!(g.ret.time.len(), 2 * 2 * 16);
        assert!(g.grid.nft == 16 && g.ret.size1 == 2);
        g.set_les(15, FftDomain::Time, &[Complex::ONE; 4]);
        assert_eq!(g.get_les(15, FftDomain::Time), vec![Complex::ONE; 4]);
        let m = [
            Complex::new(1.0, 0.0),
            Complex::new(2.0, 0.0),
            Complex::new(3.0, 0.0),
            Complex::new(4.0, 0.0),
        ];
        g.set_ret(0, FftDomain::Freq, &m);
        assert_eq!(g.get_ret(0, FftDomain::Freq), m.to_vec());
        // scalar
        let mut s = SteadyGreen::new(SteadyGrid::new(4, 0.1), 1);
        s.set_ret(0, FftDomain::Time, &[Complex::new(1.23, 4.56)]);
        s.set_les(0, FftDomain::Time, &[Complex::new(1.23, 4.56)]);
        assert_eq!(s.get_ret(0, FftDomain::Time)[0], Complex::new(1.23, 4.56));
        assert_eq!(s.get_les(0, FftDomain::Time)[0], Complex::new(1.23, 4.56));
        // construction against SteadyFunction
        let mut g = SteadyGreen::new(SteadyGrid::new(4, 0.1), 2);
        let mut r = SteadyFunction::new(4, 2);
        for t in 0..4i64 {
            let x: Vec<Complex> = (0..2)
                .flat_map(|i| {
                    (0..2)
                        .map(move |j| Complex::new(t as f64 + i as f64, -2.0 * t as f64 + j as f64))
                })
                .collect();
            r.block_mut(FftDomain::Time, t).copy_from_slice(&x);
            g.set_ret(t, FftDomain::Time, &x);
            g.set_les(t, FftDomain::Time, &x);
        }
        assert!(g.ret.distance(&r, FftDomain::Time) < 1e-12);
        assert!(g.les.distance(&r, FftDomain::Time) < 1e-12);
        // ops
        let mut a = SteadyGreen::new(SteadyGrid::new(4, 0.1), 2);
        a.incr(&g, Complex::new(2.0, 0.0), FftDomain::Time);
        for q in 0..a.ret.time.len() {
            assert!((a.ret.time[q] - 2.0 * g.ret.time[q]).abs() < 1e-12);
            assert!((a.les.time[q] - 2.0 * g.les.time[q]).abs() < 1e-12);
        }
        a.smul(Complex::new(0.5, 0.0), FftDomain::Time);
        assert!(a.distance(&g, FftDomain::Time) < 1e-12);
        let mswap = [Complex::ZERO, Complex::ONE, Complex::ONE, Complex::ZERO];
        let mut b = g.clone();
        b.left_multiply(&mswap, FftDomain::Time, false);
        b.left_multiply(&mswap, FftDomain::Time, false);
        b.right_multiply(&mswap, FftDomain::Time, false);
        b.right_multiply(&mswap, FftDomain::Time, false);
        assert!(b.distance(&g, FftDomain::Time) < 1e-12);
        b.memset(FftDomain::Freq, Complex::ONE);
        b.zeros_domain(FftDomain::Time);
        b.zeros_domain(FftDomain::Freq);
        assert!(
            b.ret
                .time
                .iter()
                .chain(b.les.freq.iter())
                .all(|z| *z == Complex::ZERO)
        );
        let cc = g.clone();
        assert_eq!(cc.ret.time, g.ret.time);
        let mut c2 = g.clone();
        c2.set_matrixelement(0, 0, &g, 1, 1, FftDomain::Time);
        for t in 0..4 {
            assert_eq!(
                c2.get_les(t, FftDomain::Time)[0],
                g.get_les(t, FftDomain::Time)[2 + 1]
            );
        }
        assert!(c2.is_compatible(&g));
        assert!(!c2.is_compatible(&SteadyGreen::new(SteadyGrid::new(8, 0.1), 2)));
    }
}
