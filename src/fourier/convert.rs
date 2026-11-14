// Project : KadanoffBaym
// File    : src/fourier/convert.rs
// Purpose : resampling of the NESS2 data and the conversion with the
//           contour functions (blueprint julia/src/fourier/convert.jl)

use crate::complex::c64::C64;
use crate::fourier::transforms::{integral_transform_to_freq, integral_transform_to_time, FftIntegralMethod};
use crate::types::green::ContourGreen;
use crate::types::ness::{FftDomain, NessArray, NessGrid, NessHermMat};

/// Keep every `factor`-th time point and the low frequency modes
/// (blueprint `downsample(A, factor)`). `Nft` must be divisible by
/// `2 factor`, the result has `Nft/factor` points.
pub fn downsample_array(a: &NessArray, factor: i64) -> NessArray {
    let nft = a.nft;
    assert!(factor >= 1 && nft % (2 * factor) == 0, "downsample: wrong factor {factor}");
    let nft1 = nft / factor;
    let b = a.size1 * a.size2;
    let mut out = NessArray::new(nft1, a.size1);
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
        out.block_mut(FftDomain::Freq, nft1 - w).copy_from_slice(&src);
    }
    let _ = b;
    out
}

/// Downsample both components (blueprint `downsample(G, factor)`). The
/// new grid has the time step `factor h` and the same frequency step.
pub fn downsample(g: &NessHermMat, factor: i64) -> NessHermMat {
    let nft = g.grid.nft;
    assert!(factor >= 1 && nft % (2 * factor) == 0, "downsample: wrong factor {factor}");
    let grid = NessGrid::new(nft / factor, g.grid.h * factor as f64);
    NessHermMat {
        grid,
        ret: downsample_array(&g.ret, factor),
        les: downsample_array(&g.les, factor),
    }
}

/// Increase the number of the grid points by `factor` (the time step
/// becomes `h/factor`, the frequency step is kept; blueprint
/// `upsample`). The frequency data is regenerated from the time data
/// by the trapezoidal rule, the new modes are zero, and the result is
/// transformed back to the time domain. `g` is not changed.
pub fn upsample(g: &NessHermMat, factor: i64) -> NessHermMat {
    assert!(factor >= 1, "upsample: wrong factor {factor}");
    let nft = g.grid.nft;
    let nft1 = nft * factor;
    let mut gin = g.clone();
    integral_transform_to_freq(&mut gin, FftIntegralMethod::Trapez);
    let mut out = NessHermMat::new(NessGrid::new(nft1, g.grid.h / factor as f64), g.ret.size1);
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
/// `factor < 0` (blueprint `resample`).
pub fn resample(g: &NessHermMat, factor: i64) -> NessHermMat {
    assert!(factor != 0, "resample: factor must not be 0");
    if factor < 0 {
        upsample(g, -factor)
    } else {
        downsample(g, factor)
    }
}

/// Sample a contour Green's function at the 0-based time step `tstp`
/// (blueprint `cntr2ness!` with the default `tstp = getntime(G)`):
/// `G^R(dt) = G^R(t_n, t_{n-dt})` and `G^<(-dt) = G^<(t_{n-dt}, t_n)`
/// for `dt = 0, ..., min(n, Nft/2 - 1)`, and
/// `G^<(dt) = -G^<(-dt)^dagger`. The other time data is zero, the
/// frequency data is not changed.
pub fn cntr2ness(gness: &mut NessHermMat, g: &ContourGreen, tstp: i64) {
    assert!(
        gness.ret.size1 == g.ret.size1 && gness.ret.size2 == g.ret.size2,
        "cntr2ness: matrix size mismatch"
    );
    assert!(
        (0..=g.nt as i64).contains(&tstp),
        "cntr2ness: tstp must be in 0..={}, got {tstp}",
        g.nt
    );
    let n = tstp;
    let nft = gness.grid.nft;
    let d = gness.ret.size1;
    gness.zeros_domain(FftDomain::Time);
    for dt in 0..=(n.min(nft / 2 - 1)) {
        let r = g.ret.raw_block_at(n, n - dt).to_vec();
        gness
            .ret
            .block_mut(FftDomain::Time, dt)
            .copy_from_slice(&r);
        let mm = g.less.raw_block_at(n - dt, n).to_vec();
        if dt == 0 {
            gness
                .les
                .block_mut(FftDomain::Time, 0)
                .copy_from_slice(&mm);
        } else {
            gness
                .les
                .block_mut(FftDomain::Time, nft - dt)
                .copy_from_slice(&mm);
            let dst = gness.les.block_mut(FftDomain::Time, dt);
            for i in 0..d {
                for j in 0..d {
                    dst[i * d + j] = -mm[j * d + i].conj();
                }
            }
        }
    }
}

/// Fill the real-time retarded and lesser components of a contour
/// Green's function assuming the time translation invariance
/// (blueprint `ness2cntr!`). If `n - m >= Nft/2` the values are zero.
/// The Matsubara and mixing components are not changed.
pub fn ness2cntr(g: &mut ContourGreen, gness: &NessHermMat) {
    assert!(
        gness.ret.size1 == g.ret.size1 && gness.ret.size2 == g.ret.size2,
        "ness2cntr: matrix size mismatch"
    );
    let nft = gness.grid.nft;
    let d = g.ret.size1;
    let z = vec![C64::ZERO; d * d];
    for n in 0..=g.nt as i64 {
        for m in 0..=n {
            let dt = n - m;
            if dt < nft / 2 {
                let r = gness.ret.block(FftDomain::Time, dt).to_vec();
                g.ret.raw_block_at_mut(n, m).copy_from_slice(&r);
                let l = gness.les.block(FftDomain::Time, dt).to_vec();
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
