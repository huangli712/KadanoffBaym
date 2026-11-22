// Project : KadanoffBaym
// File    : src/types/steady.rs
// Purpose : the steady-state types
//
// The steady-state functions depend on the relative time only. They are
// stored on a periodic grid of `Nft` points. The mode `m` (0-based,
// used as the storage index here) corresponds to the
// time `t_m = m h` for `m < Nft/2` and `t_m = (m - Nft) h` otherwise,
// and to the frequency `omega_m` with `dw = pi/((Nft/2) h)`.
//
// The storage layout of `SteadyFunction` is `(size1, size2, Nft)` with the
// mode running slowest, so that the orbital matrix of every mode is
// contiguous and the Fourier transforms run along the last dimension.

//!
//! *Remarks* : *Steady-State Grid*
//!
//! The steady-state functions depend on the relative time ``t`` only.
//! They are stored on a periodic grid of `Nft` points. The mode `m` (0-based)
//! corresponds to the time ``t_m = m h`` for ``m < N_{ft}/2``
//! and ``t_m = (m - N_{ft}) h`` otherwise, and to the frequency ``\omega_m``
//! with ``\Delta\omega = \pi/((N_{ft}/2) h) = 2\pi/(N_{ft} h)``.

use std::f64::consts::PI;

use crate::complex::Complex;

/// Domain of the steady-state data.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum FftDomain {
    Time,
    Freq,
}

/// Grid of the steady-state functions. It is an
/// immutable value and not a contour function.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct SteadyGrid {
    /// Number of the grid points, an even number >= 4.
    pub nft: i64,

    /// Time step.
    pub h: f64,

    /// Frequency step, `pi / ((Nft/2) h)`.
    pub dw: f64,
}

impl SteadyGrid {
    /// Constructor with the frequency step `pi / ((Nft/2) h)`.
    pub fn new(nft: i64, h: f64) -> SteadyGrid {
        assert!(h > 0.0, "h must be positive, got {h}");
        SteadyGrid::new_full(nft, h, PI / ((nft / 2) as f64 * h))
    }

    /// Constructor with all three fields.
    pub fn new_full(nft: i64, h: f64, dw: f64) -> SteadyGrid {
        assert!(
            nft >= 4 && nft % 2 == 0,
            "Nft must be an even number >= 4, got {nft}"
        );
        assert!(h > 0.0, "h must be positive, got {h}");
        assert!(dw > 0.0, "dw must be positive, got {dw}");
        SteadyGrid { nft, h, dw }
    }

    /// A new grid with the time step `h`, the frequency step is updated
    /// accordingly.
    pub fn set_h(&self, h: f64) -> SteadyGrid {
        assert!(h > 0.0, "h must be positive, got {h}");
        SteadyGrid::new_full(self.nft, h, PI / ((self.nft / 2) as f64 * h))
    }

    /// A new grid with the frequency step `dw`, the time step is
    /// updated accordingly.
    pub fn set_dw(&self, dw: f64) -> SteadyGrid {
        assert!(dw > 0.0, "dw must be positive, got {dw}");
        SteadyGrid::new_full(self.nft, PI / ((self.nft / 2) as f64 * dw), dw)
    }
}

/// Time and frequency data of a steady-state matrix function on the
/// steady-state grid. A plain numerical storage, not a contour function.
#[derive(Clone)]
pub struct SteadyFunction {
    pub nft: i64,

    pub size1: usize,

    pub size2: usize,

    /// Time data, the matrix of the mode `m` is `time[m*block..]`.
    pub time: Vec<Complex>,

    /// Frequency data, the matrix of the mode `m` is `freq[m*block..]`.
    pub freq: Vec<Complex>,
}

impl SteadyFunction {
    /// Constructor with zero time and frequency data.
    pub fn new(nft: i64, size1: usize) -> SteadyFunction {
        assert!(nft >= 1, "Nft must be positive, got {nft}");
        assert!(size1 >= 1, "size1 must be positive, got {size1}");
        let block = size1 * size1;
        let total = nft as usize * block;
        SteadyFunction {
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

    /// The matrix of the mode `m` (0-based) in the domain `dom`.
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

    /// Zero one domain of the data.
    pub(crate) fn zeros_domain(&mut self, dom: FftDomain) {
        let data = if dom == FftDomain::Time {
            &mut self.time
        } else {
            &mut self.freq
        };
        data.fill(Complex::ZERO);
    }
}

/// Hermitian Green's function on the steady-state grid.
/// Only the retarded and the lesser components are stored, both in the
/// time and in the frequency domain. The lesser component at negative
/// times is not reconstructed automatically, the algorithms keep the
/// relation `G^<(-t) = -G^<(t)^dagger` explicitly.
#[derive(Clone)]
pub struct SteadyGreen {
    pub grid: SteadyGrid,
    pub ret: SteadyFunction,
    pub les: SteadyFunction,
}

impl SteadyGreen {
    /// Constructor with zero values.
    pub fn new(grid: SteadyGrid, size1: usize) -> SteadyGreen {
        let ret = SteadyFunction::new(grid.nft, size1);
        let les = SteadyFunction::new(grid.nft, size1);
        SteadyGreen { grid, ret, les }
    }

    /// Same grid and matrix sizes.
    pub(crate) fn is_compatible(&self, other: &SteadyGreen) -> bool {
        self.grid == other.grid
            && self.ret.size1 == other.ret.size1
            && self.ret.size2 == other.ret.size2
            && self.ret.nft == other.ret.nft
    }

    /// A copy of the retarded matrix at the mode `m`.
    pub fn get_ret(&self, m: i64, dom: FftDomain) -> Vec<Complex> {
        self.ret.block(dom, m).to_vec()
    }

    /// A copy of the lesser matrix at the mode `m`.
    pub fn get_les(&self, m: i64, dom: FftDomain) -> Vec<Complex> {
        self.les.block(dom, m).to_vec()
    }

    /// Copy the matrix `M` to the retarded component at the mode `m`.
    pub fn set_ret(&mut self, m: i64, dom: FftDomain, mm: &[Complex]) {
        self.ret.block_mut(dom, m).copy_from_slice(mm);
    }

    /// Copy the matrix `M` to the lesser component at the mode `m`.
    pub fn set_les(&mut self, m: i64, dom: FftDomain, mm: &[Complex]) {
        self.les.block_mut(dom, m).copy_from_slice(mm);
    }

    /// Zero one domain of both components.
    pub(crate) fn zeros_domain(&mut self, dom: FftDomain) {
        self.ret.zeros_domain(dom);
        self.les.zeros_domain(dom);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::f64::consts::FRAC_PI_2;

    use std::panic::catch_unwind;

    #[test]
    fn steady_grid_info() {
        let (nt, h1, h2, dw2) = (1000i64, 0.1f64, 0.01, 0.1);
        let g = SteadyGrid::new(nt, h1);
        assert!(g.h == h1 && g.nft == nt);
        assert!((g.dw - PI / ((nt / 2) as f64 * h1)).abs() < 1e-10);
        let expect =
            |g: &SteadyGrid, m: i64, s: f64| (if m < g.nft / 2 { m } else { m - g.nft }) as f64 * s;
        for gg in [g, g.set_h(h2), g.set_dw(dw2)] {
            let mut st = 0.0;
            let mut sf = 0.0;
            for m in 0..nt {
                st += (gg.time_at(m) - expect(&gg, m, gg.h)).abs();
                sf += (gg.freq_at(m) - expect(&gg, m, gg.dw)).abs();
            }
            assert!((st / nt as f64) < 1e-10);
            assert!((sf / nt as f64) < 1e-10);
        }
        assert!(g.set_h(h2).h == h2);
        assert!(g.set_dw(dw2).dw == dw2);
        assert!((g.set_dw(dw2).h - PI / ((nt / 2) as f64 * dw2)).abs() < 1e-14);
        // storage modes
        let g = SteadyGrid::new(16, 0.25);
        assert!((g.dw - FRAC_PI_2).abs() < 1e-12);
        assert_eq!(g.time_at(0), 0.0);
        assert_eq!(g.time_at(7), 7.0 * 0.25);
        assert_eq!(g.time_at(8), -8.0 * 0.25);
        assert_eq!(g.time_at(15), -0.25);
        assert_eq!(g.freq_at(8), -8.0 * g.dw);
        assert!(catch_unwind(|| g.freq_at(-1)).is_err());
        assert!(catch_unwind(|| g.time_at(16)).is_err());
        assert!(catch_unwind(|| SteadyGrid::new(15, 0.1)).is_err());
        assert!(catch_unwind(|| SteadyGrid::new(2, 0.1)).is_err());
        assert!(catch_unwind(|| SteadyGrid::new(16, 0.0)).is_err());
        assert!(catch_unwind(|| g.set_h(-1.0)).is_err());
        assert!(catch_unwind(|| g.set_dw(0.0)).is_err());
    }
}
