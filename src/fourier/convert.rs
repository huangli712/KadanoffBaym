// Project : KadanoffBaym
// File    : src/fourier/convert.rs
// Purpose : resampling of the NESS2 data and the conversion with the
//           contour functions

use crate::complex::Complex;
use crate::fourier::FftIntegralMethod;
use crate::types::ContourGreen;
use crate::types::{FftDomain, SteadyFunction, SteadyGreen, SteadyGrid};

use crate::fourier::{integral_transform_to_freq, integral_transform_to_time};

/// Keep every `factor`-th time point and the low frequency modes.
/// `Nft` must be divisible by `2 factor`, the result has `Nft/factor`
/// points.
pub fn downsample_function(a: &SteadyFunction, factor: i64) -> SteadyFunction {
    let nft = a.nft;
    assert!(
        factor >= 1 && nft % (2 * factor) == 0,
        "downsample: wrong factor {factor}"
    );
    let nft1 = nft / factor;
    let mut out = SteadyFunction::new(nft1, a.size1);
    for t in 0..nft1 {
        let src = a.block(FftDomain::Time, t * factor).to_vec();
        out.block_mut(FftDomain::Time, t).copy_from_slice(&src);
    }
    for w in 0..(nft1 / 2) {
        let src = a.block(FftDomain::Freq, w).to_vec();
        out.block_mut(FftDomain::Freq, w).copy_from_slice(&src);
    }
    for w in 1..=(nft1 / 2) {
        let src = a.block(FftDomain::Freq, nft - w).to_vec();
        out.block_mut(FftDomain::Freq, nft1 - w)
            .copy_from_slice(&src);
    }
    out
}

/// Downsample both components. The
/// new grid has the time step `factor h` and the same frequency step.
pub fn downsample(g: &SteadyGreen, factor: i64) -> SteadyGreen {
    let nft = g.grid.nft;
    assert!(
        factor >= 1 && nft % (2 * factor) == 0,
        "downsample: wrong factor {factor}"
    );
    let grid = SteadyGrid::new(nft / factor, g.grid.h * factor as f64);
    SteadyGreen {
        grid,
        ret: downsample_function(&g.ret, factor),
        les: downsample_function(&g.les, factor),
    }
}

/// Increase the number of the grid points by `factor` (the time step
/// becomes `h/factor`, the frequency step is kept). The frequency data
/// is regenerated from the time data by the trapezoidal rule, the new
/// modes are zero, and the result is transformed back to the time
/// domain. `g` is not changed.
pub fn upsample(g: &SteadyGreen, factor: i64) -> SteadyGreen {
    assert!(factor >= 1, "upsample: wrong factor {factor}");
    let nft = g.grid.nft;
    let nft1 = nft * factor;
    let mut gin = g.clone();
    integral_transform_to_freq(&mut gin, FftIntegralMethod::Trapez);
    let mut out = SteadyGreen::new(SteadyGrid::new(nft1, g.grid.h / factor as f64), g.ret.size1);
    for (x, y) in [(&mut out.ret, &gin.ret), (&mut out.les, &gin.les)] {
        for w in 0..(nft / 2) {
            let src = y.block(FftDomain::Freq, w).to_vec();
            x.block_mut(FftDomain::Freq, w).copy_from_slice(&src);
        }
        for w in 1..=(nft / 2) {
            let src = y.block(FftDomain::Freq, nft - w).to_vec();
            x.block_mut(FftDomain::Freq, nft1 - w).copy_from_slice(&src);
        }
    }
    integral_transform_to_time(&mut out, FftIntegralMethod::Trapez);
    out
}

/// Downsample for `factor > 0`, upsample by `-factor` for
/// `factor < 0`.
pub fn resample(g: &SteadyGreen, factor: i64) -> SteadyGreen {
    assert!(factor != 0, "resample: factor must not be 0");
    if factor < 0 {
        upsample(g, -factor)
    } else {
        downsample(g, factor)
    }
}

/// Sample a contour Green's function at the 0-based time step `tstp`:
/// `G^R(dt) = G^R(t_n, t_{n-dt})` and `G^<(-dt) = G^<(t_{n-dt}, t_n)`
/// for `dt = 0, ..., min(n, Nft/2 - 1)`, and
/// `G^<(dt) = -G^<(-dt)^dagger`. The other time data is zero, the
/// frequency data is not changed.
pub fn contour_to_steady(gsteady: &mut SteadyGreen, g: &ContourGreen, tstp: i64) {
    assert!(
        gsteady.ret.size1 == g.ret.size1 && gsteady.ret.size2 == g.ret.size2,
        "contour_to_steady: matrix size mismatch"
    );
    assert!(
        (0..=g.nt as i64).contains(&tstp),
        "contour_to_steady: tstp must be in 0..={}, got {tstp}",
        g.nt
    );
    let n = tstp;
    let nft = gsteady.grid.nft;
    let d = gsteady.ret.size1;
    gsteady.zeros_domain(FftDomain::Time);
    for dt in 0..=(n.min(nft / 2 - 1)) {
        let r = g.ret.raw_block_at(n, n - dt).to_vec();
        gsteady
            .ret
            .block_mut(FftDomain::Time, dt)
            .copy_from_slice(&r);
        let mm = g.less.raw_block_at(n - dt, n).to_vec();
        if dt == 0 {
            gsteady
                .les
                .block_mut(FftDomain::Time, 0)
                .copy_from_slice(&mm);
        } else {
            gsteady
                .les
                .block_mut(FftDomain::Time, nft - dt)
                .copy_from_slice(&mm);
            let dst = gsteady.les.block_mut(FftDomain::Time, dt);
            for i in 0..d {
                for j in 0..d {
                    dst[i * d + j] = -mm[j * d + i].conj();
                }
            }
        }
    }
}

/// Fill the real-time retarded and lesser components of a contour
/// Green's function assuming the time translation invariance.
/// If `n - m >= Nft/2` the values are zero.
/// The Matsubara and mixing components are not changed.
pub fn steady_to_contour(g: &mut ContourGreen, gsteady: &SteadyGreen) {
    assert!(
        gsteady.ret.size1 == g.ret.size1 && gsteady.ret.size2 == g.ret.size2,
        "steady_to_contour: matrix size mismatch"
    );
    let nft = gsteady.grid.nft;
    let d = g.ret.size1;
    let z = vec![Complex::ZERO; d * d];
    for n in 0..=g.nt as i64 {
        for m in 0..=n {
            let dt = n - m;
            if dt < nft / 2 {
                let r = gsteady.ret.block(FftDomain::Time, dt).to_vec();
                g.ret.raw_block_at_mut(n, m).copy_from_slice(&r);
                let l = gsteady.les.block(FftDomain::Time, dt).to_vec();
                let dst = g.less.raw_block_at_mut(m, n);
                for i in 0..d {
                    for j in 0..d {
                        dst[i * d + j] = -l[j * d + i].conj();
                    }
                }
            } else {
                g.ret.raw_block_at_mut(n, m).copy_from_slice(&z);
                g.less.raw_block_at_mut(m, n).copy_from_slice(&z);
            }
        }
    }
}
