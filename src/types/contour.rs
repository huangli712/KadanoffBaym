// Project : KadanoffBaym
// File    : src/types/contour.rs
// Purpose : L-shape Kadanoff-Baym contour (blueprint: types/contour.jl)
//
// INDEX CONVENTION (decision D6, NESSi-aligned; differs from the Julia
// blueprint by -1 everywhere):
//
//   Julia blueprint              this Rust version
//   -----------------------      -----------------------
//   ntime  (points on [0,tmax])  nt    = ntime - 1  (max tstp)
//   ntau   (points on [0,beta])  ntau  = julia ntau - 1 (intervals)
//   tstp n (1-based, 1..ntime)   tstp  = n - 1      (0..=nt, 0-based)
//   mat index i (1-based)        0..=ntau
//   Matsubara flag tstp = 0      tstp = -1 (MATSUBARA)
//
// Hence: real-time points = nt + 1, tau points = ntau + 1,
// dt = tmax / nt, dtau = beta / ntau.
//!
//! *Remarks : Kadanoff-Baym Contour*
//!
//! We adopt an 𝐿-shaped `Kadanoff-Baym` contour ``\mathcal{C}`` with
//! three branches:
//!
//! * ``\mathcal{C}_1``: ``0 \longrightarrow t_{\text{max}}``
//! * ``\mathcal{C}_2``: ``t_{\text{max}} \longrightarrow 0``
//! * ``\mathcal{C}_3``: ``0 \longrightarrow -i\beta``
//!
//! where ``t_{\text{max}}`` is the maximal time up to which one wants to
//! let the system evolve and ``\beta`` is the inverse temperature
//! (``\beta \equiv 1/T``).
//!
//! The expectation value of an observable ``\mathcal{O}`` measured at time
//! ``t`` is given by
//!
//! ```math
//! \begin{equation}
//! \langle \mathcal{O}(t) \rangle =
//!     \frac{\text{Tr}\left[
//!                    \mathcal{T}_{\mathcal{C}}
//!                    e^{-i\int_{\mathcal{C}} d\bar{t} \mathcal{H}(\bar{t})}
//!                    \mathcal{O}(t)
//!                    \right]}
//!          {\text{Tr}\left[
//!                    \mathcal{T}_{\mathcal{C}}
//!                    e^{-i\int_{\mathcal{C}} d\bar{t} \mathcal{H}(\bar{t})}
//!          \right]},
//! \end{equation}
//! ```text
//!
//! where ``\mathcal{T}_{\mathcal{C}}`` is a contour-ordering operator
//! that arranges operators on the contour ``\mathcal{C}`` in the order
//! ``0 \to t_{\text{max}} \to 0 \to -i\beta``, while ``\mathcal{T}_{\tau}``
//! is the time-ordering operator only on the imaginary-time axis.
//!
//! The contour-ordered formalism reveals its full power when it is applied
//! to higher-order correlation functions,
//!
//! ```math
//! \begin{equation}
//! \langle \mathcal{T}_{\mathcal{C}} \mathcal{A}(t) \mathcal{B}(t') \rangle
//!     =
//!     \frac{1}{Z} \text{Tr}
//!     \left[
//!         \mathcal{T}_{\mathcal{C}}
//!         e^{-i\int_{\mathcal{C}} d\bar{t} \mathcal{H}(\bar{t})}
//!         \mathcal{A}(t) \mathcal{B}(t')
//!     \right].
//! \end{equation}
//! ```text
//!
//! Here ``\mathcal{A}`` and ``\mathcal{B}`` are combinations of particle's
//! creation and annihilation operators. We call them ''fermionic'' if they
//! contain odd number of fermion's creation or annihilation operators, and
//! ''bosonic'' otherwise. In this expression, ``t`` and ``t'`` can lie
//! anywhere on ``\mathcal{C}``, and the contour-ordered product of two
//! operators ``\mathcal{A}`` and ``\mathcal{B}`` is defined as
//!
//! ```math
//! \begin{equation}
//! \mathcal{T}_{\mathcal{C}} \mathcal{A}(t) \mathcal{B}(t')
//!     = \theta_{\mathcal{C}}(t,t') \mathcal{A}(t) \mathcal{B}(t')
//!     \pm \theta_{\mathcal{C}}(t',t) \mathcal{B}(t') \mathcal{A}(t),
//! \end{equation}
//! ```text
//!
//! where ``\theta_{\mathcal{C}}(t,t') = 1`` when ``t'`` comes earlier than
//! ``t`` in the contour ordering (denoted by ``t \succ t'``) and 0 otherwise
//! (``t \prec t'``). The sign ``\pm`` is taken to be minus when the operators
//! ``\mathcal{A}`` and ``\mathcal{B}`` are both fermionic and plus otherwise.

/// L-shape Kadanoff-Baym contour (blueprint `Cn`).
#[derive(Clone, Debug, PartialEq)]
pub struct Contour {
    /// Number of real-time intervals on `[0, tmax]` (points = nt + 1).
    pub nt: i64,
    
    /// Number of imaginary-time intervals on `[0, beta]` (points = ntau + 1).
    pub ntau: i64,
    
    /// Row size of the orbital blocks.
    pub size1: i64,
    
    /// Column size of the orbital blocks.
    pub size2: i64,
    
    /// Maximum real time.
    pub tmax: f64,
    
    /// Inverse temperature.
    pub beta: f64,
    
    /// Real-time step (`tmax / nt`).
    pub dt: f64,
    
    /// Imaginary-time step (`beta / ntau`).
    pub dtau: f64,
}

impl Contour {
    /// General constructor (mirror of the blueprint `Cn(ntime, ntau, ...)`
    /// with the -1 translation of both grid sizes).
    pub fn new(nt: i64, ntau: i64, size1: i64, size2: i64, tmax: f64, beta: f64) -> Contour {
        assert!(nt >= 1, "Contour: nt >= 1, got {nt}");
        assert!(ntau >= 1, "Contour: ntau >= 1, got {ntau}");
        assert!(size1 >= 1, "Contour: size1 >= 1, got {size1}");
        assert!(size2 >= 1, "Contour: size2 >= 1, got {size2}");
        assert!(tmax > 0.0, "Contour: tmax > 0");
        assert!(beta > 0.0, "Contour: beta > 0");
        Contour {
            nt,
            ntau,
            size1,
            size2,
            tmax,
            beta,
            dt: tmax / nt as f64,
            dtau: beta / ntau as f64,
        }
    }

    /// Square-block constructor with the blueprint defaults
    /// (`Julia ntime = 201, ntau = 1001` -> `nt = 200, ntau = 1000`).
    pub fn with_defaults(size1: i64, size2: i64, tmax: f64, beta: f64) -> Contour {
        Contour::new(200, 1000, size1, size2, tmax, beta)
    }

    /// Recompute `dt` and `dtau` after the grid was changed
    /// (blueprint `refresh!`).
    pub fn refresh(&mut self) {
        assert!(self.nt >= 1, "refresh: nt >= 1");
        assert!(self.ntau >= 1, "refresh: ntau >= 1");
        self.dt = self.tmax / self.nt as f64;
        self.dtau = self.beta / self.ntau as f64;
    }

    /// Real time of timestep `tstp` (`tstp * dt`, 0-based).
    #[inline]
    pub fn time(&self, tstp: i64) -> f64 {
        tstp as f64 * self.dt
    }

    /// Imaginary time of tau point `n` (`n * dtau`, 0-based).
    #[inline]
    pub fn tau(&self, n: i64) -> f64 {
        n as f64 * self.dtau
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constructor_and_refresh() {
        let c = Contour::new(10, 20, 2, 3, 1.0, 5.0);
        assert_eq!(c.dt, 0.1);
        assert_eq!(c.dtau, 0.25);
        c.clone().refresh();
        let mut c2 = c.clone();
        c2.nt = 4;
        c2.tmax = 2.0;
        c2.refresh();
        assert_eq!(c2.dt, 0.5);
        let d = Contour::with_defaults(1, 1, 10.0, 20.0);
        assert_eq!((d.nt, d.ntau), (200, 1000));
        assert_eq!(d.time(3), 3.0 * d.dt);
        assert_eq!(d.tau(7), 7.0 * d.dtau);
    }

    #[test]
    #[should_panic(expected = "nt >= 1")]
    fn rejects_degenerate_grid() {
        let _ = Contour::new(0, 10, 1, 1, 1.0, 1.0);
    }
}
