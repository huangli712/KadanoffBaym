//
// Project : KadanoffBaym
// File    : src/observables/spectrum.rs
// Purpose : Wigner slices and windowed spectral functions
//

use crate::complex::Complex;
use crate::types::ContourGreen;

use crate::fourier::dft_cplx;

/// Method of the windowed spectral function.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SpectrumMethod {
    Cubic,
    Linear,
}

/// Extract the retarded and lesser components at the fixed average
/// time step `tav` (0-based) as functions of the relative time: for
/// `j = 0, ..., min(tav, nt - tav)` with `nt` the number of real time
/// intervals, the times are `t1 = tav + j` and `t2 = tav - j`.
///
/// Returns the relative times `2jh`, the retarded blocks `G^R(t_1, t_2)`
/// and the lesser blocks `G^<(t_1, t_2)`, one flat `d x d` vector per
/// time point.
///
/// HDF5 is not supported; the slices are taken from `G` directly.
pub fn wigner_slices(
    g: &ContourGreen,
    tav: i64,
    h: f64,
) -> (Vec<f64>, Vec<Vec<Complex>>, Vec<Vec<Complex>>) {
    let d1 = g.ret.size1;
    let d2 = g.ret.size2;
    assert!(
        d1 == d2,
        "the matrices of G must be square, got {d1} x {d2}"
    );
    let nt = g.nt as i64;
    assert!(
        (0..=nt).contains(&tav),
        "tav must be in 0..={nt}, got {tav}"
    );
    assert!(h > 0.0, "h must be positive, got {h}");

    let nj = (tav.min(nt - tav) + 1) as usize;
    let t_rel: Vec<f64> = (0..nj).map(|j| 2.0 * j as f64 * h).collect();
    let mut ret = Vec::with_capacity(nj);
    let mut les = Vec::with_capacity(nj);
    for j in 0..nj as i64 {
        let t1 = tav + j;
        let t2 = tav - j;
        ret.push(g.ret_block(t1, t2).raw().to_vec());
        les.push(g.less_block(t1, t2).raw().to_vec());
    }
    (t_rel, ret, les)
}

/// Fourier integral `integral dt e^{i w t} v(t)` of the piecewise
/// linear interpolation of `v` on the equidistant grid `t`. At `w = 0`
/// the trapezoidal rule is used.
fn linear_fourier(v: &[Complex], t: &[f64], w: f64) -> Complex {
    let nn = v.len() - 1;
    let dt = t[1] - t[0];
    if w == 0.0 {
        let mut res = 0.5 * (v[0] + v[nn]);
        res += v[1..nn].iter().copied().sum::<Complex>();
        return res * dt;
    }
    let mut res = Complex::ZERO;
    for it in 0..nn {
        res += v[it] * Complex::from_polar(1.0, w * t[it]);
    }
    let s = (w * dt / 2.0).sin();
    res *= Complex::new(4.0 * s * s / w / w / dt, 0.0);
    let cis_t = |x: f64| Complex::from_polar(1.0, x);
    res += (v[nn] * cis_t(w * t[nn]) - v[0] * cis_t(w * t[0]))
        * (Complex::new(0.0, -1.0 / w) - (cis_t(-w * dt) - Complex::ONE) / (w * w * dt));
    res
}

/// Windowed spectral function of the diagonal elements,
///
/// ```math
/// A_{ii}(\\omega) = -\\frac{1}{\\pi}\\mathrm{Im}\\int dt\\ e^{i\\omega t}
/// W(t) G^R_{ii}(t),
/// ```
///
/// on the arbitrary frequencies `omega`. The integral runs over the
/// equidistant grid `t_rel`; `gret` holds one `d x d` block per time
/// point (see `wigner_slices`).
///
/// * `method = SpectrumMethod::Cubic`: the corrected DFT `dft_cplx()`. The number of the
///   intervals `t_rel.len() - 1` must be even and larger than 16,
///   otherwise the call panics (there is no fallback to
///   `SpectrumMethod::Linear`).
/// * `method = SpectrumMethod::Linear`: the exact integral of the linear interpolation, at
///   least two points are needed.
///
/// Returns the spectral function `A[w][i]` for every frequency index `w`
/// of `omega` and every orbital `i`. `window` maps the time grid to a
/// complex vector of the length `nt`.
pub fn evaluate_spectrum_windowed(
    gret: &[Vec<Complex>],
    t_rel: &[f64],
    omega: &[f64],
    method: SpectrumMethod,
    window: &dyn Fn(&[f64]) -> Vec<Complex>,
) -> Vec<Vec<f64>> {
    let nt = gret.len();
    let b = gret[0].len();
    let d = (b as f64).sqrt() as usize;
    assert!(d * d == b, "gret must contain square matrices");
    assert_eq!(t_rel.len(), nt, "t_rel must have {nt} points");
    assert!(nt >= 2, "at least two time points are needed");
    let nn = nt - 1;
    if method == SpectrumMethod::Cubic {
        assert!(
            nn > 16 && nn.is_multiple_of(2),
            "SpectrumMethod::Cubic needs an even number (> 16) of intervals, got {nn}"
        );
    }
    let dt = t_rel[1] - t_rel[0];
    assert!(dt > 0.0, "t_rel must be increasing");
    for i in 0..nn {
        assert!(
            (t_rel[i + 1] - t_rel[i] - dt).abs() <= 1e-10 * dt.abs(),
            "t_rel must be equidistant"
        );
    }
    assert!(omega.iter().all(|w| w.is_finite()), "omega must be finite");

    let w = window(t_rel);
    assert_eq!(
        w.len(),
        nt,
        "window(t_rel) must return a vector of the length {nt}"
    );

    let mut a_out = vec![vec![0.0f64; d]; omega.len()];
    let mut v = vec![Complex::ZERO; nt];
    for i in 0..d {
        for (q, item) in v.iter_mut().enumerate() {
            *item = gret[q][i * d + i] * w[q];
        }
        for (k, om) in omega.iter().enumerate() {
            let res = if method == SpectrumMethod::Cubic {
                dft_cplx(*om, t_rel[0], t_rel[nt - 1], &v).0
            } else {
                linear_fourier(&v, t_rel, *om)
            };
            a_out[k][i] = -res.im / std::f64::consts::PI;
        }
    }
    a_out
}
