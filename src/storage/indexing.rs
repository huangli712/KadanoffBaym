//
// Project : KadanoffBaym
// File    : src/storage/indexing.rs
// Purpose : value-level indexing and derived components
//
// hermiticity relations of the stored triangles:
//
//   ret(t,t')  : stored for tstp >= j, else -(ret[t',t])^dagger
//   less(t,t') : stored for tstp' >= i, else -(less[t',t])^dagger
//   matminus(n): mat[ntau - n] * sign            (KMS condition)
//   rmix(n,t)  : lmix[t, ntau - n]^dagger * (-sign)
//   gtr(t,t')  : less(t,t') + ret(t,t')
//   adv(t,t')  : ret(t',t)^dagger
//
// The element relations and the derived blocks are implemented once on
// the components (the free functions before the two impl blocks); each
// container type has ONE impl block with the sections: element access
// (get/set pairs in the component order mat, ret, lmix, less), derived
// values and blocks (read-only, `get_matminus` first), raw block access
// (the ops layer), and the private lmix row helper of the timestep
// object at the bottom.

//!
//! *Remarks* : *Hermitian Conjugate Relation of the Retarded Component*
//!
//! In principle, when ``t < t'``, ``G^{R}(t,t') \equiv 0``. Here, we assume
//! that the modified retarded component also fulfills the following hermitian
//! conjugate relation:
//!
//! ```math
//! \begin{equation}
//! \tilde{G}^{R}(t,t') = - \tilde{G}^{R}(t',t)^{*}
//! \end{equation}
//! ```
//!
//! *Remarks* : *Hermitian Conjugate Relation of the Lesser Component*
//!
//! Here we apply the following hermitian conjugate relation:
//!
//! ```math
//! \begin{equation}
//! G^{<}(t,t') = -G^{<}(t',t)^{*}
//! \end{equation}
//! ```
//!
//! *Remarks* : *Kubo-Martin-Schwinger Boundary Condition*
//!
//! The following Kubo-Martin-Schwinger boundary condition is used:
//!
//! ```math
//! \begin{equation}
//! G(\tau) = \pm G(\tau + \beta)
//! \end{equation}
//! ```
//!
//! where `+` is for bosonic system, while `-` is for fermionic system.
//!

use crate::complex::Complex;
use crate::linalg::DenseMat;
use crate::types::{ContourGreen, GreenTimeStep, LessComp, LmixComp, MatComp, RetComp, Sign};

impl ContourGreen {
    // ---------------- element access ----------------

    pub fn get_mat(&self, n: i64, a: usize, b: usize) -> Complex {
        self.mat.get(self.mat.check_tau(n), a, b)
    }

    pub fn set_mat(&mut self, n: i64, a: usize, b: usize, v: Complex) {
        let k = self.mat.check_tau(n);
        self.mat.set(k, a, b, v)
    }

    pub fn get_ret(&self, tstp: i64, j: i64, a: usize, b: usize) -> Complex {
        ret_get(&self.ret, tstp, j, a, b)
    }

    pub fn set_ret(&mut self, tstp: i64, j: i64, a: usize, b: usize, v: Complex) {
        let k = self.ret.idx(tstp, j);
        self.ret.set(k, a, b, v)
    }

    pub fn get_lmix(&self, tstp: i64, n: i64, a: usize, b: usize) -> Complex {
        self.lmix.get(self.lmix.idx(tstp, n), a, b)
    }

    pub fn set_lmix(&mut self, tstp: i64, n: i64, a: usize, b: usize, v: Complex) {
        let k = self.lmix.idx(tstp, n);
        self.lmix.set(k, a, b, v)
    }

    pub fn get_less(&self, i: i64, tstp: i64, a: usize, b: usize) -> Complex {
        less_get(&self.less, i, tstp, a, b)
    }

    pub fn set_less(&mut self, i: i64, tstp: i64, a: usize, b: usize, v: Complex) {
        let k = self.less.idx(i, tstp);
        self.less.set(k, a, b, v)
    }

    // ---------------- derived values and blocks (read-only) ----------------

    pub fn get_matminus(&self, n: i64, a: usize, b: usize) -> Complex {
        matminus_get(&self.mat, n, a, b, self.sign)
    }

    pub fn ret_block(&self, tstp: i64, j: i64) -> DenseMat {
        ret_block(&self.ret, tstp, j)
    }

    pub fn lmix_block(&self, tstp: i64, n: i64) -> DenseMat {
        lmix_block(&self.lmix, self.lmix.idx(tstp, n))
    }

    pub fn less_block(&self, i: i64, tstp: i64) -> DenseMat {
        less_block(&self.less, i, tstp)
    }

    pub fn adv_block(&self, tstp: i64, j: i64) -> DenseMat {
        self.ret_block(j, tstp).adjoint()
    }

    pub fn rmix_block(&self, n: i64, tstp: i64) -> DenseMat {
        let k = self.lmix.idx(tstp, self.lmix.ntau as i64 - n);
        rmix_block(&self.lmix, k, self.sign)
    }

    pub fn gtr_block(&self, tstp: i64, j: i64) -> DenseMat {
        gtr_block(&self.less, &self.ret, tstp, j)
    }

    // ---------------- raw block access (ops layer) ----------------

    pub fn mat_raw_block(&self, n: i64) -> &[Complex] {
        self.mat.block(self.mat.check_tau(n))
    }

    pub fn mat_raw_block_mut(&mut self, n: i64) -> &mut [Complex] {
        let k = self.mat.check_tau(n);
        self.mat.block_mut(k)
    }

    pub fn ret_raw_block(&self, tstp: i64, j: i64) -> &[Complex] {
        self.ret.block(self.ret.idx(tstp, j))
    }

    pub fn ret_raw_block_mut(&mut self, tstp: i64, j: i64) -> &mut [Complex] {
        let k = self.ret.idx(tstp, j);
        self.ret.block_mut(k)
    }

    pub fn lmix_raw_block(&self, tstp: i64, n: i64) -> &[Complex] {
        self.lmix.block(self.lmix.idx(tstp, n))
    }

    pub fn lmix_raw_block_mut(&mut self, tstp: i64, n: i64) -> &mut [Complex] {
        let k = self.lmix.idx(tstp, n);
        self.lmix.block_mut(k)
    }

    pub fn less_raw_block(&self, i: i64, tstp: i64) -> &[Complex] {
        self.less.block(self.less.idx(i, tstp))
    }

    pub fn less_raw_block_mut(&mut self, i: i64, tstp: i64) -> &mut [Complex] {
        let k = self.less.idx(i, tstp);
        self.less.block_mut(k)
    }
}

impl GreenTimeStep {
    // ---------------- element access ----------------

    pub fn get_mat(&self, n: i64, a: usize, b: usize) -> Complex {
        self.mat.get(self.mat.check_tau(n), a, b)
    }

    pub fn set_mat(&mut self, n: i64, a: usize, b: usize, v: Complex) {
        let k = self.mat.check_tau(n);
        self.mat.set(k, a, b, v)
    }

    pub fn get_ret(&self, tstp: i64, j: i64, a: usize, b: usize) -> Complex {
        ret_get(&self.ret, tstp, j, a, b)
    }

    pub fn set_ret(&mut self, tstp: i64, j: i64, a: usize, b: usize, v: Complex) {
        let k = self.ret.idx(tstp, j);
        self.ret.set(k, a, b, v)
    }

    pub fn get_lmix(&self, tstp: i64, n: i64, a: usize, b: usize) -> Complex {
        self.lmix.get(self.lmix_idx(tstp, n), a, b)
    }

    pub fn set_lmix(&mut self, tstp: i64, n: i64, a: usize, b: usize, v: Complex) {
        let k = self.lmix_idx(tstp, n);
        self.lmix.set(k, a, b, v)
    }

    pub fn get_less(&self, i: i64, tstp: i64, a: usize, b: usize) -> Complex {
        less_get(&self.less, i, tstp, a, b)
    }

    pub fn set_less(&mut self, i: i64, tstp: i64, a: usize, b: usize, v: Complex) {
        let k = self.less.idx(i, tstp);
        self.less.set(k, a, b, v)
    }

    // ---------------- derived values and blocks (read-only) ----------------

    pub fn get_matminus(&self, n: i64, a: usize, b: usize) -> Complex {
        matminus_get(&self.mat, n, a, b, self.sign)
    }

    pub fn ret_block(&self, tstp: i64, j: i64) -> DenseMat {
        ret_block(&self.ret, tstp, j)
    }

    pub fn lmix_block(&self, tstp: i64, n: i64) -> DenseMat {
        lmix_block(&self.lmix, self.lmix_idx(tstp, n))
    }

    pub fn less_block(&self, i: i64, tstp: i64) -> DenseMat {
        less_block(&self.less, i, tstp)
    }

    pub fn adv_block(&self, tstp: i64, j: i64) -> DenseMat {
        self.ret_block(j, tstp).adjoint()
    }

    pub fn rmix_block(&self, n: i64, tstp: i64) -> DenseMat {
        let k = self.lmix_idx(tstp, self.lmix.ntau as i64 - n);
        rmix_block(&self.lmix, k, self.sign)
    }

    pub fn gtr_block(&self, tstp: i64, j: i64) -> DenseMat {
        gtr_block(&self.less, &self.ret, tstp, j)
    }

    // ---------------- raw block access (ops layer) ----------------

    pub fn mat_raw_block(&self, n: i64) -> &[Complex] {
        self.mat.block(self.mat.check_tau(n))
    }

    pub fn mat_raw_block_mut(&mut self, n: i64) -> &mut [Complex] {
        let k = self.mat.check_tau(n);
        self.mat.block_mut(k)
    }

    pub fn ret_raw_block(&self, tstp: i64, j: i64) -> &[Complex] {
        self.ret.block(self.ret.idx(tstp, j))
    }

    pub fn ret_raw_block_mut(&mut self, tstp: i64, j: i64) -> &mut [Complex] {
        let k = self.ret.idx(tstp, j);
        self.ret.block_mut(k)
    }

    pub fn lmix_raw_block(&self, tstp: i64, n: i64) -> &[Complex] {
        self.lmix.block(self.lmix_idx(tstp, n))
    }

    pub fn lmix_raw_block_mut(&mut self, tstp: i64, n: i64) -> &mut [Complex] {
        let k = self.lmix_idx(tstp, n);
        self.lmix.block_mut(k)
    }

    pub fn less_raw_block(&self, i: i64, tstp: i64) -> &[Complex] {
        self.less.block(self.less.idx(i, tstp))
    }

    pub fn less_raw_block_mut(&mut self, i: i64, tstp: i64) -> &mut [Complex] {
        let k = self.less.idx(i, tstp);
        self.less.block_mut(k)
    }

    // ---------------- private helpers ----------------

    fn lmix_idx(&self, tstp: i64, n: i64) -> usize {
        assert!(
            tstp == self.tstp as i64,
            "timestep lmix row is tstp = {}",
            self.tstp
        );
        self.lmix.idx(0, n)
    }
}

// ------------------------------------------------------------------
// shared implementations on the components (both containers forward
// to them)
// ------------------------------------------------------------------

/// Element of the retarded component: `ret[t,t']` inside the stored
/// triangle, else `-(ret[t',t])^dagger`.
#[inline]
fn ret_get(r: &RetComp, tstp: i64, j: i64, a: usize, b: usize) -> Complex {
    if tstp >= j {
        r.get(r.idx(tstp, j), a, b)
    } else {
        -(r.get(r.idx(j, tstp), b, a)).conj()
    }
}

/// Element of the lesser component: `less[t,t']` inside the stored
/// triangle, else `-(less[t',t])^dagger`.
#[inline]
fn less_get(l: &LessComp, i: i64, tstp: i64, a: usize, b: usize) -> Complex {
    if i <= tstp {
        l.get(l.idx(i, tstp), a, b)
    } else {
        -(l.get(l.idx(tstp, i), b, a)).conj()
    }
}

/// `mat[ntau - n] * sign` (KMS condition).
#[inline]
fn matminus_get(m: &MatComp, n: i64, a: usize, b: usize, sign: Sign) -> Complex {
    let k = m.check_tau(n);
    m.get((m.ntau - k) as i64 as usize, a, b) * Complex::new(sign.as_i64() as f64, 0.0)
}

/// The `s1 x s2` block `src` reconstructed as `-(src)^dagger`: the
/// result has the shape `(s2, s1)`.
fn adjoint_block(src: &[Complex], s1: usize, s2: usize) -> DenseMat {
    let mut out = DenseMat::zeros_shape(s2, s1);
    for i in 0..s2 {
        for j in 0..s1 {
            out.set(i, j, -src[j * s2 + i].conj());
        }
    }
    out
}

/// Block of the retarded component: inside the stored triangle the
/// stored block, else `-(ret[t',t])^dagger`.
fn ret_block(r: &RetComp, tstp: i64, j: i64) -> DenseMat {
    let (s1, s2) = (r.size1, r.size2);
    if tstp >= j {
        let k = r.idx(tstp, j);
        DenseMat::from_slice(s1, s2, r.block(k))
    } else {
        let k = r.idx(j, tstp);
        adjoint_block(r.block(k), s1, s2)
    }
}

/// Block of the lesser component: inside the stored triangle the stored
/// block, else `-(less[t',t])^dagger`.
fn less_block(l: &LessComp, i: i64, tstp: i64) -> DenseMat {
    let (s1, s2) = (l.size1, l.size2);
    if i <= tstp {
        let k = l.idx(i, tstp);
        DenseMat::from_slice(s1, s2, l.block(k))
    } else {
        let k = l.idx(tstp, i);
        adjoint_block(l.block(k), s1, s2)
    }
}

/// Block of the left-mixing component at the flat block index `k`.
fn lmix_block(l: &LmixComp, k: usize) -> DenseMat {
    DenseMat::from_slice(l.size1, l.size2, l.block(k))
}

/// `G^> = G^R + G^<`. Away from equal times one of the two blocks is
/// reconstructed as the adjoint of a stored one, which for non-square
/// blocks has the transposed shape.
fn gtr_block(l: &LessComp, r: &RetComp, tstp: i64, j: i64) -> DenseMat {
    assert!(
        l.size1 == l.size2 || tstp == j,
        "gtr_block: non-square {}x{} blocks only have the greater component at equal times (tstp == j)",
        l.size1,
        l.size2
    );
    let mut out = less_block(l, tstp, j);
    out.axpy(1.0, &ret_block(r, tstp, j));
    out
}

/// `lmix[tstp, ntau - n]^dagger * (-sign)` for the lmix block `k`.
fn rmix_block(l: &LmixComp, k: usize, sign: Sign) -> DenseMat {
    let (s1, s2) = (l.size1, l.size2);
    let src = l.block(k);
    // adjoint shape (s2, s1)
    let mut out = DenseMat::zeros_shape(s2, s1);
    let f = -(sign.as_i64() as f64);
    for i in 0..s2 {
        for j in 0..s1 {
            out.set(i, j, src[j * s2 + i].conj().scale(f));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::types::{Contour, Sign};

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
}
