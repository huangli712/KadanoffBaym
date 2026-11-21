// Project : KadanoffBaym
// File    : src/io/pseudo.rs
// Purpose : pseudo-particle plain text I/O; the sections are shared with the standard
//           format, the header marks the pseudo-particle convention

//!
//! *Pseudo-Particle Text Format*
//!
//! The file consists of a header line followed by the `mat`, `ret`, `lmix`
//! and `less` sections, exactly as in the standard format. The header marks
//! the pseudo-particle convention, so that the reader rejects a standard
//! file.

use std::fmt;
use std::fs;
use std::str::FromStr;

use crate::io::IoError;
use crate::types::PseudoGreen;

use crate::io::green::green_from_str;
use crate::io::{write_less_section, write_lmix_section, write_mat_section, write_ret_section};

/// Serialize a `PseudoGreen` to the plain text format.
pub(crate) fn pseudo_green_to_string(g: &PseudoGreen) -> String {
    let mut out = String::new();
    out.push_str("# Pseudo-Particle Contour-Ordered Green's Function:\n");
    out.push_str(&format!("sign  : {}\n", g.sign.as_i64()));
    out.push('\n');
    write_mat_section(&mut out, &g.mat);
    out.push('\n');
    write_ret_section(&mut out, &g.ret);
    out.push('\n');
    write_lmix_section(&mut out, &g.lmix);
    out.push('\n');
    write_less_section(&mut out, &g.less);
    out.push('\n');
    out
}

/// Parse the `Display` text of a `PseudoGreen` into the preallocated
/// `g` (the sizes of `g` are never changed, the sign is taken from the
/// file).
pub fn pseudo_green_from_str(text: &str, g: &mut PseudoGreen) -> Result<(), IoError> {
    let first = text.lines().next().unwrap_or("");
    if !first.contains("Pseudo") {
        return Err(IoError {
            line: 1,
            msg: "not a pseudo-particle Green's function file".to_string(),
        });
    }
    let src = green_from_str(text)?;
    if src.nt != g.nt || src.ntau != g.ntau || src.size1 != g.size1 || src.size2 != g.size2 {
        return Err(IoError {
            line: 1,
            msg: format!(
                "dimension mismatch: file {}x{} nt={} ntau={}, target {}x{} nt={} ntau={}",
                src.size1, src.size2, src.nt, src.ntau, g.size1, g.size2, g.nt, g.ntau
            ),
        });
    }
    g.sign = src.sign;
    g.mat = src.mat.clone();
    g.ret = src.ret.clone();
    g.lmix = src.lmix.clone();
    g.less = src.less.clone();
    Ok(())
}

/// Write the pseudo-particle Green's function to a file.
pub fn write_pseudo_green(path: &str, g: &PseudoGreen) -> std::io::Result<()> {
    fs::write(path, pseudo_green_to_string(g))
}

/// Read the pseudo-particle Green's function from a file into `g`.
pub fn read_pseudo_green(path: &str, g: &mut PseudoGreen) -> Result<(), IoError> {
    let text = fs::read_to_string(path).map_err(|e| IoError {
        line: 0,
        msg: format!("{path}: {e}"),
    })?;
    pseudo_green_from_str(&text, g)
}

// ------------------------------------------------------------------
// string-level serialization: the free helpers stay crate-private, the
// public surface is the standard Display / FromStr traits
// ------------------------------------------------------------------

impl fmt::Display for PseudoGreen {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&pseudo_green_to_string(self))
    }
}

impl FromStr for PseudoGreen {
    type Err = IoError;
    /// Factory parse: the shared sections carry the dimensions, so the
    /// object is allocated from the text itself. The fill-style
    /// [`pseudo_green_from_str`] remains the validating read-into API.
    fn from_str(text: &str) -> Result<Self, IoError> {
        let first = text.lines().next().unwrap_or("");
        if !first.contains("Pseudo") {
            return Err(IoError {
                line: 1,
                msg: "not a pseudo-particle Green's function file".to_string(),
            });
        }
        let src = green_from_str(text)?;
        Ok(PseudoGreen {
            sign: src.sign,
            nt: src.nt,
            ntau: src.ntau,
            size1: src.size1,
            size2: src.size2,
            mat: src.mat,
            ret: src.ret,
            lmix: src.lmix,
            less: src.less,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::complex::Complex;
    use crate::types::{Contour, ContourGreen, Sign};

    use crate::types::MATSUBARA;

    use crate::io::green::green_to_string;

    fn rand_c64(state: &mut u64) -> Complex {
        let a = (splitmix64(state) >> 11) as f64 / (1u64 << 53) as f64 - 0.5;
        let b = (splitmix64(state) >> 11) as f64 / (1u64 << 53) as f64 - 0.5;
        Complex::new(a * 4.0, b * 4.0)
    }

    fn splitmix64(state: &mut u64) -> u64 {
        *state = state.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = *state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }

    /// Fill all the stored blocks with random numbers.
    fn ps_fill(g: &mut PseudoGreen, state: &mut u64) {
        let (nt, ntau) = (g.nt as i64, g.ntau as i64);
        let b = g.size1 * g.size2;
        for m in 0..=ntau {
            for q in 0..b {
                g.mat.block_mut(m as usize)[q] = rand_c64(state);
            }
        }
        for i in 0..=nt {
            for j in 0..=i {
                for q in 0..b {
                    g.ret.raw_block_at_mut(i, j)[q] = rand_c64(state);
                }
            }
            for m in 0..=ntau {
                for q in 0..b {
                    g.lmix.raw_block_at_mut(i, m)[q] = rand_c64(state);
                }
            }
            for j in i..=nt {
                for q in 0..b {
                    g.less.raw_block_at_mut(i, j)[q] = rand_c64(state);
                }
            }
        }
    }

    #[test]
    fn from_str_factory_round_trip() {
        let c = Contour::new(6, 10, 2, 2, 1.0, 2.0);
        let mut g1 = PseudoGreen::new(&c, Sign::Bose, Complex::ONE);
        for i in 0..g1.mat.raw().len() {
            g1.mat.raw_mut()[i] = Complex::new(i as f64, -(i as f64) / 3.0);
        }
        let text = g1.to_string();
        let g2 = PseudoGreen::from_str(&text).unwrap();
        assert_eq!(g2.to_string(), text);
    }

    #[test]
    fn pseudo_plain_text_io() {
        let cc = Contour::new(5, 6, 2, 2, 0.5, 2.0);
        let mut state = 13u64;
        let mut a = PseudoGreen::new(&cc, Sign::Bose, Complex::ZERO);
        ps_fill(&mut a, &mut state);
        let text = pseudo_green_to_string(&a);
        assert!(text.lines().next().unwrap().contains("Pseudo"));
        let mut b = PseudoGreen::new(&cc, Sign::Fermi, Complex::ZERO);
        assert!(pseudo_green_from_str(&text, &mut b).is_ok());
        assert!(b.sign == Sign::Bose);
        for t in MATSUBARA..=5 {
            assert!(a.distance_at(&b, t) < 1e-10);
        }
        // a standard file can not be read into a pseudo-particle object
        let f = ContourGreen::new(&cc, Sign::Fermi, Complex::ZERO);
        let std_text = green_to_string(&f);
        assert!(pseudo_green_from_str(&std_text, &mut b).is_err());
        // the sizes of the preallocated object are never changed
        let small = Contour::new(4, 6, 2, 2, 0.5, 2.0);
        let mut d = PseudoGreen::new(&small, Sign::Fermi, Complex::ZERO);
        assert!(pseudo_green_from_str(&text, &mut d).is_err());
        assert_eq!(d.nt, 4);
        // a missing file
        assert!(read_pseudo_green("/tmp/opencode/definitely-missing-pseudo.dat", &mut b).is_err());
        // the file wrappers round trip
        let path = "/tmp/opencode/pseudo_structs_io.dat";
        write_pseudo_green(path, &a).unwrap();
        let mut b2 = PseudoGreen::new(&cc, Sign::Fermi, Complex::ZERO);
        assert!(read_pseudo_green(path, &mut b2).is_ok());
        for t in MATSUBARA..=5 {
            assert!(b2.distance_at(&a, t) < 1e-10);
        }
    }
}
