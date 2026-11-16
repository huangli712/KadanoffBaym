// Project : KadanoffBaym
// File    : src/basics/pseudo.rs
// Purpose : operations on the pseudo-particle containers (blueprint
//           traits/properties dispatches)
//
// Split from types/pseudo.rs: the type definitions and conversions
// live there, the operation impl block lives here.

use crate::complex::Complex;
use crate::linalg::DenseMat;
use crate::types::{GreenTimeStep, PseudoGreen, PseudoGreenTimeStep, TimeFunction};
use crate::types::{LessComp, LmixComp, MatComp, RetComp};

use crate::types::MATSUBARA;

use crate::basics::{set_timestep_at, timestep_at};

impl PseudoGreen {
    /// Zero all the components (blueprint `zeros!(Z)`).
    pub fn zeros_all(&mut self) {
        self.mat.raw_mut().fill(Complex::ZERO);
        self.ret.raw_mut().fill(Complex::ZERO);
        self.lmix.raw_mut().fill(Complex::ZERO);
        self.less.raw_mut().fill(Complex::ZERO);
    }

    /// Elementwise `self += alpha * src` over all the components
    /// (blueprint `incr!(S, B, alpha)`).
    pub fn incr(&mut self, src: &PseudoGreen, alpha: Complex) {
        for (d, s) in [
            (self.mat.raw_mut(), src.mat.raw()),
            (self.ret.raw_mut(), src.ret.raw()),
            (self.lmix.raw_mut(), src.lmix.raw()),
            (self.less.raw_mut(), src.less.raw()),
        ] {
            for q in 0..d.len() {
                d[q] += alpha * s[q];
            }
        }
    }

    /// Elementwise scaling of all the components (blueprint
    /// `smul!(S, a)`).
    pub fn smul(&mut self, alpha: Complex) {
        for d in [
            self.mat.raw_mut(),
            self.ret.raw_mut(),
            self.lmix.raw_mut(),
            self.less.raw_mut(),
        ] {
            for z in d.iter_mut() {
                *z *= alpha;
            }
        }
    }

    /// Copy the components at time step `tstp` (blueprint
    /// `memcpy!(A, Y, t)`).
    pub fn memcpy_step(&mut self, src: &PseudoGreen, tstp: i64) {
        if tstp == MATSUBARA {
            self.mat.raw_mut().copy_from_slice(src.mat.raw());
        } else {
            assert!((0..=self.nt as i64).contains(&tstp));
            for j in 0..=tstp {
                let a = src.ret.raw_block_at(tstp, j);
                self.ret.raw_block_at_mut(tstp, j).copy_from_slice(a);
            }
            for i in 0..=tstp {
                let a = src.less.raw_block_at(i, tstp);
                self.less.raw_block_at_mut(i, tstp).copy_from_slice(a);
            }
            for m in 0..=self.ntau as i64 {
                let a = src.lmix.raw_block_at(tstp, m);
                self.lmix.raw_block_at_mut(tstp, m).copy_from_slice(a);
            }
        }
    }

    /// Fill the components at time step `tstp` with `v` (blueprint
    /// `memset!(Z, t, v)`).
    pub fn memset_step(&mut self, tstp: i64, v: Complex) {
        if tstp == MATSUBARA {
            self.mat.raw_mut().fill(v);
        } else {
            assert!((0..=self.nt as i64).contains(&tstp));
            for j in 0..=tstp {
                self.ret.raw_block_at_mut(tstp, j).fill(v);
            }
            for i in 0..=tstp {
                self.less.raw_block_at_mut(i, tstp).fill(v);
            }
            for m in 0..=self.ntau as i64 {
                self.lmix.raw_block_at_mut(tstp, m).fill(v);
            }
        }
    }

    /// Zero the components at time step `tstp` (blueprint `zeros!`;
    /// `MATSUBARA` zeroes the Matsubara component only).
    pub fn zeros_step(&mut self, tstp: i64) {
        if tstp == MATSUBARA {
            self.mat.raw_mut().fill(Complex::ZERO);
        } else {
            let n = tstp;
            for j in 0..=n {
                self.ret.raw_block_at_mut(n, j).fill(Complex::ZERO);
                self.less.raw_block_at_mut(j, n).fill(Complex::ZERO);
            }
            for m in 0..=self.ntau as i64 {
                self.lmix.raw_block_at_mut(n, m).fill(Complex::ZERO);
            }
        }
    }

    /// Scale the components at time step `tstp` by `alpha` (blueprint
    /// `smul!(S, t, a)`).
    pub fn smul_step(&mut self, tstp: i64, alpha: Complex) {
        if tstp == MATSUBARA {
            for z in self.mat.raw_mut().iter_mut() {
                *z *= alpha;
            }
        } else {
            assert!((0..=self.nt as i64).contains(&tstp));
            for j in 0..=tstp {
                for z in self.ret.raw_block_at_mut(tstp, j).iter_mut() {
                    *z *= alpha;
                }
                for z in self.less.raw_block_at_mut(j, tstp).iter_mut() {
                    *z *= alpha;
                }
            }
            for m in 0..=self.ntau as i64 {
                for z in self.lmix.raw_block_at_mut(tstp, m).iter_mut() {
                    *z *= alpha;
                }
            }
        }
    }

    /// Left multiplication by a contour function at time step `tstp`
    /// (blueprint `smul!(f, L, t)`): `f(t)` from the left, the lesser
    /// blocks are weighted by their first time argument, the Matsubara
    /// component by `f_0`.
    pub fn lmul_cf(&mut self, f: &TimeFunction, tstp: i64) {
        let f0 = f.block(MATSUBARA);
        for m in 0..=self.ntau as i64 {
            let fm = DenseMat::from_slice(self.size1, self.size2, f0);
            let blk = DenseMat::from_slice(self.size1, self.size2, self.mat.block(m as usize));
            let p = fm.mat_mul(&blk);
            self.mat.block_mut(m as usize).copy_from_slice(p.raw());
        }
        if tstp == MATSUBARA {
            return;
        }
        let ft = DenseMat::from_slice(self.size1, self.size2, f.block(tstp));
        for j in 0..=tstp {
            let blk = DenseMat::from_vec(
                self.size1,
                self.size2,
                self.ret.raw_block_at(tstp, j).to_vec(),
            );
            let p = ft.mat_mul(&blk);
            self.ret.raw_block_at_mut(tstp, j).copy_from_slice(p.raw());
            let fj = DenseMat::from_slice(self.size1, self.size2, f.block(j));
            let bl = DenseMat::from_vec(
                self.size1,
                self.size2,
                self.less.raw_block_at(j, tstp).to_vec(),
            );
            let pl = fj.mat_mul(&bl);
            self.less
                .raw_block_at_mut(j, tstp)
                .copy_from_slice(pl.raw());
        }
        for m in 0..=self.ntau as i64 {
            let blk = DenseMat::from_vec(
                self.size1,
                self.size2,
                self.lmix.raw_block_at(tstp, m).to_vec(),
            );
            let p = ft.mat_mul(&blk);
            self.lmix.raw_block_at_mut(tstp, m).copy_from_slice(p.raw());
        }
    }

    /// Right multiplication by a contour function at time step `tstp`
    /// (blueprint `smul!(R, f, t)`): `f` from the right, the ret blocks
    /// are weighted by their column time, the left-mixing component by
    /// `f_0`.
    pub fn rmul_cf(&mut self, f: &TimeFunction, tstp: i64) {
        let f0 = DenseMat::from_vec(self.size1, self.size2, f.block(MATSUBARA).to_vec());
        for m in 0..=self.ntau as i64 {
            let blk = DenseMat::from_slice(self.size1, self.size2, self.mat.block(m as usize));
            let p = blk.mat_mul(&f0);
            self.mat.block_mut(m as usize).copy_from_slice(p.raw());
        }
        if tstp == MATSUBARA {
            return;
        }
        let ft = DenseMat::from_slice(self.size1, self.size2, f.block(tstp));
        for j in 0..=tstp {
            let fj = DenseMat::from_slice(self.size1, self.size2, f.block(j));
            let blk = DenseMat::from_vec(
                self.size1,
                self.size2,
                self.ret.raw_block_at(tstp, j).to_vec(),
            );
            let p = blk.mat_mul(&fj);
            self.ret.raw_block_at_mut(tstp, j).copy_from_slice(p.raw());
            let bl = DenseMat::from_vec(
                self.size1,
                self.size2,
                self.less.raw_block_at(j, tstp).to_vec(),
            );
            let pl = bl.mat_mul(&ft);
            self.less
                .raw_block_at_mut(j, tstp)
                .copy_from_slice(pl.raw());
        }
        for m in 0..=self.ntau as i64 {
            let blk = DenseMat::from_vec(
                self.size1,
                self.size2,
                self.lmix.raw_block_at(tstp, m).to_vec(),
            );
            let p = blk.mat_mul(&f0);
            self.lmix.raw_block_at_mut(tstp, m).copy_from_slice(p.raw());
        }
    }

    /// Compatibility check (blueprint `iscompatible`).
    pub fn is_compatible(&self, h: &PseudoGreen) -> bool {
        self.sign == h.sign
            && self.nt == h.nt
            && self.ntau == h.ntau
            && self.size1 == h.size1
            && self.size2 == h.size2
    }

    /// Per-time-step distance (blueprint `distance(G, H, tstp)`).
    pub fn distance_at(&self, g2: &PseudoGreen, tstp: i64) -> f64 {
        self.to_contour().distance_at(&g2.to_contour(), tstp)
    }
}

/// Initialize the `t = 0` row (ret/les/lmix) of a pseudo-particle
/// green from its Matsubara component (blueprint NESSi
/// `set_t0_from_mat(herm_pseudo)`): `lmix(0,m) = i*sig*mat(ntau-m)`,
/// `less(0,0) = lmix(0,0)`, and `ret(0,0) = i*mat(0)`. Unlike the
/// standard variant the pseudo one omits the `- less(0,0)` correction
/// of `ret(0,0)` (the line is commented out in the NESSi source): the
/// `ret` slot stores the greater component here.
pub fn set_t0_from_mat_pseudo(g: &mut PseudoGreen) {
    let ntau = g.ntau;
    let s = Complex::new(0.0, g.sign.as_i64() as f64);
    let b = g.size1 * g.size2;
    let row = g.lmix.row_mut(0);
    for (m, chunk) in row.chunks_exact_mut(b).enumerate() {
        let src = g.mat.block(ntau - m).to_vec();
        for (d, x) in chunk.iter_mut().zip(src) {
            *d = s * x;
        }
    }
    let l00 = g.lmix.row(0)[..b].to_vec();
    g.less.raw_block_at_mut(0, 0).copy_from_slice(&l00);
    let m0 = g.mat.block(0).to_vec();
    for (d, x) in g.ret.raw_block_at_mut(0, 0).iter_mut().zip(m0) {
        *d = Complex::I * x;
    }
}

/// Extract the timestep `tstp` of a `PseudoGreen`. For the Matsubara
/// axis the returned object carries only the Matsubara component (its
/// `tstp` field is 0, matching the blueprint convention that the
/// Matsubara step is step 0).
pub fn pseudo_timestep(g: &PseudoGreen, tstp: i64) -> PseudoGreenTimeStep {
    if tstp == MATSUBARA {
        let mut x = GreenTimeStep {
            sign: g.sign,
            tstp: 0,
            ntau: g.ntau,
            size1: g.size1,
            size2: g.size2,
            mat: MatComp::new(g.ntau, g.size1, g.size2, Complex::ZERO),
            ret: RetComp::new(0, g.size1, g.size2, Complex::ZERO),
            lmix: LmixComp::new(0, g.ntau, g.size1, g.size2, Complex::ZERO),
            less: LessComp::new(0, g.size1, g.size2, Complex::ZERO),
        };
        x.mat.raw_mut().copy_from_slice(g.mat.raw());
        x
    } else {
        timestep_at(&g.to_contour(), tstp)
    }
}

/// Write the timestep `tstp` back into a `PseudoGreen`; for the
/// Matsubara axis only the Matsubara component is written.
pub fn set_pseudo_timestep(g: &mut PseudoGreen, tstp: i64, x: &PseudoGreenTimeStep) {
    if tstp == MATSUBARA {
        g.mat.raw_mut().copy_from_slice(x.mat.raw());
    } else {
        assert_eq!(
            x.tstp as i64, tstp,
            "the time step object carries tstp {}, not {tstp}",
            x.tstp
        );
        let mut gc = g.to_contour();
        set_timestep_at(&mut gc, x);
        g.copy_from_contour(&gc);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::types::{Contour, Sign};

    use crate::basics::set_t0_from_mat_pseudo;

    #[test]
    fn set_t0_from_mat_pseudo_variant() {
        let c = Contour::new(2, 3, 2, 2, 1.0, 1.0);
        let mut f = PseudoGreen::new(&c, Sign::Bose, Complex::ZERO);
        for q in 0..=3 {
            f.mat.block_mut(q).copy_from_slice(&[
                Complex::new(q as f64, 0.5),
                Complex::new(1.0, q as f64),
                Complex::new(-(q as f64), 2.0),
                Complex::new(3.0, -(q as f64)),
            ]);
        }
        set_t0_from_mat_pseudo(&mut f);
        // Bose statistics: the lmix prefactor is plain +i.
        let s = Complex::new(0.0, 1.0);
        for m in 0..=3i64 {
            let want = s * Complex::new((3 - m) as f64, 0.5);
            assert_eq!(f.lmix.row(0)[m as usize * 4], want);
        }
        let l00 = f.lmix.row(0)[0];
        assert_eq!(f.less.raw_block_at(0, 0)[0], l00);
        // pseudo variant: ret(0,0) carries NO -less correction
        for (d, x) in f.ret.raw_block_at(0, 0).iter().zip(f.mat.block(0)) {
            assert_eq!(*d, Complex::I * *x);
        }
    }
}
