// Project : KadanoffBaym
// File    : src/types/indexing.rs
// Purpose : value-level indexing and derived components
//
// Blueprint: julia/src/indexing/getindex.jl and
// julia/src/properties/property.jl. Element access honours the
// hermiticity relations of the stored triangles:
//
//   ret(t,t')  : stored for tstp >= j, else -(ret[t',t])^dagger
//   less(t,t') : stored for tstp' >= i, else -(less[t',t])^dagger
//   matminus(n): mat[ntau - n] * sign            (KMS condition)
//   rmix(n,t)  : lmix[t, ntau - n]^dagger * (-sign)
//   gtr(t,t')  : less(t,t') + ret(t,t')
//   adv(t,t')  : ret(t',t)^dagger

use super::Contour;
use super::green::{ContourGreen, GreenTimeStep};
use super::layout::MATSUBARA;
use crate::Complex;
use crate::linalg::mat::DenseMat;



        impl ContourGreen {
            // ---------------- stored-component element access ----------------

            /// `G^M(tau_n)` element `(a, b)`.
            #[inline]
            pub fn get_mat(&self, n: i64, a: usize, b: usize) -> Complex {
                self.mat.get(self.mat.check_tau(n), a, b)
            }

            /// Write `G^M(tau_n)` element `(a, b)`.
            #[inline]
            pub fn set_mat(&mut self, n: i64, a: usize, b: usize, v: Complex) {
                let k = self.mat.check_tau(n);
                self.mat.set(k, a, b, v)
            }

            /// `G^R(t, t')` element with hermiticity fallback.
            #[inline]
            pub fn get_ret(&self, tstp: i64, j: i64, a: usize, b: usize) -> Complex {
                if tstp >= j {
                    self.ret.get(self.ret.idx(tstp, j), a, b)
                } else {
                    -(self.ret.get(self.ret.idx(j, tstp), b, a)).conj()
                }
            }

            /// Write `G^R(t, t')` element (stored region `tstp >= j` only).
            #[inline]
            pub fn set_ret(&mut self, tstp: i64, j: i64, a: usize, b: usize, v: Complex) {
                let k = self.ret.idx(tstp, j);
                self.ret.set(k, a, b, v)
            }

            /// `G^(mix)(t, tau_n)` element.
            #[inline]
            pub fn get_lmix(&self, tstp: i64, n: i64, a: usize, b: usize) -> Complex {
                self.lmix.get(self.lmix_idx(tstp, n), a, b)
            }

            /// Write `G^(mix)(t, tau_n)` element.
            #[inline]
            pub fn set_lmix(&mut self, tstp: i64, n: i64, a: usize, b: usize, v: Complex) {
                let k = self.lmix_idx(tstp, n);
                self.lmix.set(k, a, b, v)
            }

            /// `G^<(t, t')` element with hermiticity fallback.
            #[inline]
            pub fn get_less(&self, i: i64, tstp: i64, a: usize, b: usize) -> Complex {
                if i <= tstp {
                    self.less.get(self.less.idx(i, tstp), a, b)
                } else {
                    -(self.less.get(self.less.idx(tstp, i), b, a)).conj()
                }
            }

            /// Write `G^<(t, t')` element (stored region `i <= tstp` only).
            #[inline]
            pub fn set_less(&mut self, i: i64, tstp: i64, a: usize, b: usize, v: Complex) {
                let k = self.less.idx(i, tstp);
                self.less.set(k, a, b, v)
            }

            // ---------------- derived components (read-only) ----------------

            /// `G^M(tau < 0)` element: `mat[ntau - n] * sign` (KMS).
            #[inline]
            pub fn get_matminus(&self, n: i64, a: usize, b: usize) -> Complex {
                let k = self.mat.check_tau(n);
                self.mat.get((self.mat.ntau - k) as i64 as usize, a, b)
                    * Complex::new(self.sign.as_i64() as f64, 0.0)
            }

            /// `G^R(t, t')`-matrix for any `(t, t')` (stored view or
            /// reconstructed copy).
            pub fn ret_block(&self, tstp: i64, j: i64) -> DenseMat {
                let (s1, s2) = (self.ret.size1, self.ret.size2);
                if tstp >= j {
                    let k = self.ret.idx(tstp, j);
                    DenseMat::from_vec(s1, s2, self.ret.block(k).to_vec())
                } else {
                    let k = self.ret.idx(j, tstp);
                    let src = self.ret.block(k);
                    // reconstructed block: -(stored)† has shape (s2, s1)
                    let mut out = DenseMat::zeros_shape(s2, s1);
                    for i in 0..s2 {
                        for j in 0..s1 {
                            out.set(i, j, -src[j * s2 + i].conj());
                        }
                    }
                    out
                }
            }

            /// `G^<(t, t')`-matrix for any `(t, t')`.
            pub fn less_block(&self, i: i64, tstp: i64) -> DenseMat {
                let (s1, s2) = (self.less.size1, self.less.size2);
                if i <= tstp {
                    let k = self.less.idx(i, tstp);
                    DenseMat::from_vec(s1, s2, self.less.block(k).to_vec())
                } else {
                    let k = self.less.idx(tstp, i);
                    let src = self.less.block(k);
                    // reconstructed block: -(stored)† has shape (s2, s1)
                    let mut out = DenseMat::zeros_shape(s2, s1);
                    for i2 in 0..s2 {
                        for j2 in 0..s1 {
                            out.set(i2, j2, -src[j2 * s2 + i2].conj());
                        }
                    }
                    out
                }
            }

            /// `G^(mix)(t, tau)`-matrix.
            pub fn lmix_block(&self, tstp: i64, n: i64) -> DenseMat {
                let (s1, s2) = (self.lmix.size1, self.lmix.size2);
                let k = self.lmix_idx(tstp, n);
                DenseMat::from_vec(s1, s2, self.lmix.block(k).to_vec())
            }

            /// `G^M(tau)`-matrix.
            pub fn mat_block(&self, n: i64) -> DenseMat {
                let (s1, s2) = (self.mat.size1, self.mat.size2);
                let k = self.mat.check_tau(n);
                DenseMat::from_vec(s1, s2, self.mat.block(k).to_vec())
            }

            /// `G^A(t, t') = G^R(t', t)^dagger` matrix (read-only).
            pub fn adv_block(&self, tstp: i64, j: i64) -> DenseMat {
                self.ret_block(j, tstp).adjoint()
            }

            /// `G^(rmix)(tau, t) = lmix[t, ntau - n]^dagger * (-sign)`
            /// matrix (read-only).
            pub fn rmix_block(&self, n: i64, tstp: i64) -> DenseMat {
                let k = self.lmix_idx(tstp, self.lmix.ntau as i64 - n);
                let (s1, s2) = (self.lmix.size1, self.lmix.size2);
                let src = self.lmix.block(k);
                // adjoint shape (s2, s1)
                let mut out = DenseMat::zeros_shape(s2, s1);
                let f = -(self.sign.as_i64() as f64);
                for i in 0..s2 {
                    for j in 0..s1 {
                        out.set(i, j, src[j * s2 + i].conj().scale(f));
                    }
                }
                out
            }

            /// `G^>(t, t') = G^<(t, t') + G^R(t, t')` matrix (read-only).
            pub fn gtr_block(&self, tstp: i64, j: i64) -> DenseMat {
                let mut out = self.less_block(tstp, j);
                out.axpy(1.0, &self.ret_block(tstp, j));
                out
            }
        }
    

        impl GreenTimeStep {
            // ---------------- stored-component element access ----------------

            /// `G^M(tau_n)` element `(a, b)`.
            #[inline]
            pub fn get_mat(&self, n: i64, a: usize, b: usize) -> Complex {
                self.mat.get(self.mat.check_tau(n), a, b)
            }

            /// Write `G^M(tau_n)` element `(a, b)`.
            #[inline]
            pub fn set_mat(&mut self, n: i64, a: usize, b: usize, v: Complex) {
                let k = self.mat.check_tau(n);
                self.mat.set(k, a, b, v)
            }

            /// `G^R(t, t')` element with hermiticity fallback.
            #[inline]
            pub fn get_ret(&self, tstp: i64, j: i64, a: usize, b: usize) -> Complex {
                if tstp >= j {
                    self.ret.get(self.ret.idx(tstp, j), a, b)
                } else {
                    -(self.ret.get(self.ret.idx(j, tstp), b, a)).conj()
                }
            }

            /// Write `G^R(t, t')` element (stored region `tstp >= j` only).
            #[inline]
            pub fn set_ret(&mut self, tstp: i64, j: i64, a: usize, b: usize, v: Complex) {
                let k = self.ret.idx(tstp, j);
                self.ret.set(k, a, b, v)
            }

            /// `G^(mix)(t, tau_n)` element.
            #[inline]
            pub fn get_lmix(&self, tstp: i64, n: i64, a: usize, b: usize) -> Complex {
                self.lmix.get(self.lmix_idx(tstp, n), a, b)
            }

            /// Write `G^(mix)(t, tau_n)` element.
            #[inline]
            pub fn set_lmix(&mut self, tstp: i64, n: i64, a: usize, b: usize, v: Complex) {
                let k = self.lmix_idx(tstp, n);
                self.lmix.set(k, a, b, v)
            }

            /// `G^<(t, t')` element with hermiticity fallback.
            #[inline]
            pub fn get_less(&self, i: i64, tstp: i64, a: usize, b: usize) -> Complex {
                if i <= tstp {
                    self.less.get(self.less.idx(i, tstp), a, b)
                } else {
                    -(self.less.get(self.less.idx(tstp, i), b, a)).conj()
                }
            }

            /// Write `G^<(t, t')` element (stored region `i <= tstp` only).
            #[inline]
            pub fn set_less(&mut self, i: i64, tstp: i64, a: usize, b: usize, v: Complex) {
                let k = self.less.idx(i, tstp);
                self.less.set(k, a, b, v)
            }

            // ---------------- derived components (read-only) ----------------

            /// `G^M(tau < 0)` element: `mat[ntau - n] * sign` (KMS).
            #[inline]
            pub fn get_matminus(&self, n: i64, a: usize, b: usize) -> Complex {
                let k = self.mat.check_tau(n);
                self.mat.get((self.mat.ntau - k) as i64 as usize, a, b)
                    * Complex::new(self.sign.as_i64() as f64, 0.0)
            }

            /// `G^R(t, t')`-matrix for any `(t, t')` (stored view or
            /// reconstructed copy).
            pub fn ret_block(&self, tstp: i64, j: i64) -> DenseMat {
                let (s1, s2) = (self.ret.size1, self.ret.size2);
                if tstp >= j {
                    let k = self.ret.idx(tstp, j);
                    DenseMat::from_vec(s1, s2, self.ret.block(k).to_vec())
                } else {
                    let k = self.ret.idx(j, tstp);
                    let src = self.ret.block(k);
                    // reconstructed block: -(stored)† has shape (s2, s1)
                    let mut out = DenseMat::zeros_shape(s2, s1);
                    for i in 0..s2 {
                        for j in 0..s1 {
                            out.set(i, j, -src[j * s2 + i].conj());
                        }
                    }
                    out
                }
            }

            /// `G^<(t, t')`-matrix for any `(t, t')`.
            pub fn less_block(&self, i: i64, tstp: i64) -> DenseMat {
                let (s1, s2) = (self.less.size1, self.less.size2);
                if i <= tstp {
                    let k = self.less.idx(i, tstp);
                    DenseMat::from_vec(s1, s2, self.less.block(k).to_vec())
                } else {
                    let k = self.less.idx(tstp, i);
                    let src = self.less.block(k);
                    // reconstructed block: -(stored)† has shape (s2, s1)
                    let mut out = DenseMat::zeros_shape(s2, s1);
                    for i2 in 0..s2 {
                        for j2 in 0..s1 {
                            out.set(i2, j2, -src[j2 * s2 + i2].conj());
                        }
                    }
                    out
                }
            }

            /// `G^(mix)(t, tau)`-matrix.
            pub fn lmix_block(&self, tstp: i64, n: i64) -> DenseMat {
                let (s1, s2) = (self.lmix.size1, self.lmix.size2);
                let k = self.lmix_idx(tstp, n);
                DenseMat::from_vec(s1, s2, self.lmix.block(k).to_vec())
            }

            /// `G^M(tau)`-matrix.
            pub fn mat_block(&self, n: i64) -> DenseMat {
                let (s1, s2) = (self.mat.size1, self.mat.size2);
                let k = self.mat.check_tau(n);
                DenseMat::from_vec(s1, s2, self.mat.block(k).to_vec())
            }

            /// `G^A(t, t') = G^R(t', t)^dagger` matrix (read-only).
            pub fn adv_block(&self, tstp: i64, j: i64) -> DenseMat {
                self.ret_block(j, tstp).adjoint()
            }

            /// `G^(rmix)(tau, t) = lmix[t, ntau - n]^dagger * (-sign)`
            /// matrix (read-only).
            pub fn rmix_block(&self, n: i64, tstp: i64) -> DenseMat {
                let k = self.lmix_idx(tstp, self.lmix.ntau as i64 - n);
                let (s1, s2) = (self.lmix.size1, self.lmix.size2);
                let src = self.lmix.block(k);
                // adjoint shape (s2, s1)
                let mut out = DenseMat::zeros_shape(s2, s1);
                let f = -(self.sign.as_i64() as f64);
                for i in 0..s2 {
                    for j in 0..s1 {
                        out.set(i, j, src[j * s2 + i].conj().scale(f));
                    }
                }
                out
            }

            /// `G^>(t, t') = G^<(t, t') + G^R(t, t')` matrix (read-only).
            pub fn gtr_block(&self, tstp: i64, j: i64) -> DenseMat {
                let mut out = self.less_block(tstp, j);
                out.axpy(1.0, &self.ret_block(tstp, j));
                out
            }
        }
    

/// Per-type lmix block-index helper: the timestep object stores a
/// single lmix row, so its accessor pins `tstp == self.tstp`.
impl ContourGreen {
    #[inline]
    fn lmix_idx(&self, tstp: i64, n: i64) -> usize {
        self.lmix.idx(tstp, n)
    }
}

impl GreenTimeStep {
    #[inline]
    fn lmix_idx(&self, tstp: i64, n: i64) -> usize {
        assert!(
            tstp == self.tstp as i64,
            "timestep lmix row is tstp = {}",
            self.tstp
        );
        self.lmix.idx(0, n)
    }
}

#[allow(dead_code)]
/// Matsubara-branch dispatch: `MATSUBARA` addresses the Matsubara
/// component, `0..=nt` the real-time rows. This helper converts a tstp
/// to an optional tau index.
#[inline]
pub(crate) fn tau_of(tstp: i64) -> Option<usize> {
    if tstp == MATSUBARA {
        Some(0) // marker; callers use the mat component explicitly
    } else {
        None
    }
}

/// Read the `lmix` row of `tstp` as raw slices (for the ops layer).
impl ContourGreen {
    /// Raw block slice of the stored ret triangle.
    #[inline]
    pub fn ret_raw_block(&self, tstp: i64, j: i64) -> &[Complex] {
        self.ret.block(self.ret.idx(tstp, j))
    }

    /// Raw mutable block slice of the stored ret triangle.
    #[inline]
    pub fn ret_raw_block_mut(&mut self, tstp: i64, j: i64) -> &mut [Complex] {
        let k = self.ret.idx(tstp, j);
        self.ret.block_mut(k)
    }

    /// Raw block slice of the stored less triangle.
    #[inline]
    pub fn less_raw_block(&self, i: i64, tstp: i64) -> &[Complex] {
        self.less.block(self.less.idx(i, tstp))
    }

    /// Raw mutable block slice of the stored less triangle.
    #[inline]
    pub fn less_raw_block_mut(&mut self, i: i64, tstp: i64) -> &mut [Complex] {
        let k = self.less.idx(i, tstp);
        self.less.block_mut(k)
    }

    /// Raw block slice of lmix.
    #[inline]
    pub fn lmix_raw_block(&self, tstp: i64, n: i64) -> &[Complex] {
        self.lmix.block(self.lmix.idx(tstp, n))
    }

    /// Raw mutable block slice of lmix.
    #[inline]
    pub fn lmix_raw_block_mut(&mut self, tstp: i64, n: i64) -> &mut [Complex] {
        let k = self.lmix.idx(tstp, n);
        self.lmix.block_mut(k)
    }

    /// Raw block slice of the mat component.
    #[inline]
    pub fn mat_raw_block(&self, n: i64) -> &[Complex] {
        self.mat.block(self.mat.check_tau(n))
    }

    /// Raw mutable block slice of the mat component.
    #[inline]
    pub fn mat_raw_block_mut(&mut self, n: i64) -> &mut [Complex] {
        let k = self.mat.check_tau(n);
        self.mat.block_mut(k)
    }
}

/// Raw block accessors for the timestep object (the lmix component is
/// a single row, so its accessor pins `tstp == self.tstp`).
impl GreenTimeStep {
    /// Raw block slice of the stored ret triangle.
    #[inline]
    pub fn ret_raw_block(&self, tstp: i64, j: i64) -> &[Complex] {
        self.ret.block(self.ret.idx(tstp, j))
    }

    /// Raw mutable block slice of the stored ret triangle.
    #[inline]
    pub fn ret_raw_block_mut(&mut self, tstp: i64, j: i64) -> &mut [Complex] {
        let k = self.ret.idx(tstp, j);
        self.ret.block_mut(k)
    }

    /// Raw block slice of the stored less triangle.
    #[inline]
    pub fn less_raw_block(&self, i: i64, tstp: i64) -> &[Complex] {
        self.less.block(self.less.idx(i, tstp))
    }

    /// Raw mutable block slice of the stored less triangle.
    #[inline]
    pub fn less_raw_block_mut(&mut self, i: i64, tstp: i64) -> &mut [Complex] {
        let k = self.less.idx(i, tstp);
        self.less.block_mut(k)
    }

    /// Raw block slice of the single lmix row.
    #[inline]
    pub fn lmix_raw_block(&self, tstp: i64, n: i64) -> &[Complex] {
        assert!(tstp == self.tstp as i64, "timestep lmix row is tstp = {}", self.tstp);
        self.lmix.block(self.lmix.idx(0, n))
    }

    /// Raw mutable block slice of the single lmix row.
    #[inline]
    pub fn lmix_raw_block_mut(&mut self, tstp: i64, n: i64) -> &mut [Complex] {
        assert!(tstp == self.tstp as i64, "timestep lmix row is tstp = {}", self.tstp);
        let k = self.lmix.idx(0, n);
        self.lmix.block_mut(k)
    }

    /// Raw block slice of the mat component.
    #[inline]
    pub fn mat_raw_block(&self, n: i64) -> &[Complex] {
        self.mat.block(self.mat.check_tau(n))
    }

    /// Raw mutable block slice of the mat component.
    #[inline]
    pub fn mat_raw_block_mut(&mut self, n: i64) -> &mut [Complex] {
        let k = self.mat.check_tau(n);
        self.mat.block_mut(k)
    }
}

#[cfg(test)]
mod tests {
    use super::super::contour::Contour;
    use super::super::sign::Sign;
    use super::super::green::ContourGreen;
    use super::MATSUBARA;
    use super::tau_of;
    use crate::complex::Complex;

    fn mk_green() -> ContourGreen {
        let c = Contour::new(3, 3, 2, 2, 1.0, 1.0);
        ContourGreen::new(&c, Sign::Fermi, Complex::ZERO)
    }

    #[test]
    fn ret_hermiticity_fallback() {
        let mut g = mk_green();
        // stored: ret[2,1][0,0] = 1+2i
        g.set_ret(2, 1, 0, 0, Complex::new(1.0, 2.0));
        assert_eq!(g.get_ret(2, 1, 0, 0), Complex::new(1.0, 2.0));
        // fallback: ret[1,2] = -(ret[2,1])† element (0,0) = -conj(1+2i)
        assert_eq!(g.get_ret(1, 2, 0, 0), Complex::new(-1.0, 2.0));
    }

    #[test]
    fn less_hermiticity_fallback() {
        let mut g = mk_green();
        g.set_less(1, 2, 0, 1, Complex::new(0.5, -0.5));
        assert_eq!(g.get_less(1, 2, 0, 1), Complex::new(0.5, -0.5));
        // fallback: less[2,1] element (1,0) = -(less[1,2])†[1,0] = -conj(0.5-0.5i)
        assert_eq!(g.get_less(2, 1, 1, 0), Complex::new(-0.5, -0.5));
    }

    #[test]
    fn matsubara_minus_kms_and_rmix() {
        let mut g = mk_green();
        g.set_mat(1, 0, 0, Complex::new(3.0, 0.0));
        // ntau = 3: matminus(2) = mat[3 - 2] * (-1) = -mat[1]
        assert_eq!(g.get_matminus(2, 0, 0), Complex::new(-3.0, 0.0));
        // bosons flip the sign
        let mut gb = mk_green();
        gb.sign = Sign::Bose;
        gb.set_mat(1, 0, 0, Complex::new(3.0, 0.0));
        assert_eq!(gb.get_matminus(2, 0, 0), Complex::new(3.0, 0.0));
        // rmix(n=1, tstp=0) = lmix[0, ntau-1]† * (-sign) = conj(i)*(+1) = -i
        g.set_lmix(0, 2, 0, 0, Complex::new(0.0, 1.0));
        let r = g.rmix_block(1, 0);
        assert_eq!(r.get(0, 0), Complex::new(0.0, -1.0));
    }

    #[test]
    fn gtr_and_adv_blocks() {
        let mut g = mk_green();
        g.set_ret(2, 1, 0, 0, Complex::new(1.0, 0.0));
        g.set_less(1, 2, 0, 0, Complex::new(0.25, 0.5));
        // gtr(1,2) = less(1,2) + ret(1,2)[fallback]
        let gtr = g.gtr_block(1, 2);
        let ret_fb = -(g.ret_block(2, 1).get(0, 0));
        assert_eq!(gtr.get(0, 0), Complex::new(0.25, 0.5) + ret_fb);
        // adv(1,2) = ret(2,1)†
        let adv = g.adv_block(1, 2);
        assert_eq!(adv.get(0, 0), Complex::new(1.0, 0.0));
    }

    #[test]
    fn msubara_marker() {
        assert_eq!(MATSUBARA, -1);
        assert!(tau_of(MATSUBARA).is_some());
        assert!(tau_of(3).is_none());
    }
}

/// Extract an owned single-timestep object from a `ContourGreen`
/// (blueprint `G[tstp]`): the mat component is shared by reference
/// semantics (copied), ret/less hold the rows `0..=tstp`, lmix the row
/// `tstp`.
pub fn timestep_at(g: &ContourGreen, tstp: i64) -> GreenTimeStep {
    assert!(
        (0..=g.nt as i64).contains(&tstp),
        "timestep_at: tstp {tstp} out of range 0..={}",
        g.nt
    );
    let mut f = GreenTimeStep::new(
        // only the shape fields are consumed; nt is clamped to >= 1
        // because Contour rejects a degenerate single-point grid
        &Contour::new(
            tstp.max(1),
            g.ntau as i64,
            g.size1 as i64,
            g.size2 as i64,
            1.0,
            1.0,
        ),
        tstp,
        g.sign,
        Complex::ZERO,
    );
    f.mat.memcpy_from(&g.mat);
    let b = g.size1 * g.size2;
    // the ret/less triangles of the timestep are exactly the parent's
    // triangle PREFIX through tstp (contiguous in both layouts)
    let n_blocks = (tstp as usize + 1) * (tstp as usize + 2) / 2;
    f.ret.raw_mut()[..n_blocks * b].copy_from_slice(&g.ret.raw()[..n_blocks * b]);
    f.less.raw_mut()[..n_blocks * b].copy_from_slice(&g.less.raw()[..n_blocks * b]);
    for m in 0..=(g.ntau as i64) {
        // the timestep lmix component stores a SINGLE row: block index m
        f.lmix.raw_mut()[m as usize * b..(m as usize + 1) * b]
            .copy_from_slice(g.lmix_raw_block(tstp, m));
    }
    f
}

/// Write a single-timestep object back into a `ContourGreen`
/// (blueprint `G[tstp] = cfv`): mat, the ret/less triangle prefixes,
/// and the lmix row of `f` land at `f.tstp` of `g`.
pub fn set_timestep_at(g: &mut ContourGreen, f: &GreenTimeStep) {
    let tstp = f.tstp as i64;
    assert!(
        (0..=g.nt as i64).contains(&tstp),
        "set_timestep_at: tstp {tstp} out of range 0..={}",
        g.nt
    );
    assert!(
        f.ntau == g.ntau && f.size1 == g.size1 && f.size2 == g.size2,
        "set_timestep_at: incompatible shapes"
    );
    g.mat.memcpy_from(&f.mat);
    let b = g.size1 * g.size2;
    let n_blocks = (tstp as usize + 1) * (tstp as usize + 2) / 2;
    g.ret.raw_mut()[..n_blocks * b].copy_from_slice(&f.ret.raw()[..n_blocks * b]);
    g.less.raw_mut()[..n_blocks * b].copy_from_slice(&f.less.raw()[..n_blocks * b]);
    let rowlen = (g.ntau + 1) * b;
    let off = tstp as usize * rowlen;
    g.lmix.raw_mut()[off..off + rowlen].copy_from_slice(f.lmix.raw());
}
