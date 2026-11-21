// Project : KadanoffBaym
// File    : src/lib.rs
// Purpose : crate root and shared numeric type aliases
//
// Numerical results must match NESSi 2.0.0 point by point.
//!
//! KadanoffBaym
//!
//! `KadanoffBaym` is a Rust crate for simulating nonequilibrium
//! strongly correlated electron systems. It provides application
//! programming interfaces to manipulate the nonequilibrium Green's
//! functions defined on the L-shaped Kadanoff-Baym contour, including:
//!
//! * contour-ordered containers and their element-wise operations
//!   (`types`, `basics`)
//! * convolutions between two contour-ordered Green's functions
//!   (`convolution`)
//! * shared Matsubara-axis solver components (`matsubara`)
//! * Dyson and Volterra integro-differential equation solvers (`dyson`)
//! * second-order Volterra integral equation solvers (`vie2`)
//! * equilibrium and steady-state solvers (`equilibrium::from_dos_steady`,
//!   `dyson::dyson_steady`)
//! * Fourier tools and contour/steady conversion (`fourier`)
//! * quadrature weights and integrators (`weights`, `math`)
//! * complex dense linear algebra (`linalg`)
//! * plain-text IO (`io`) and observables (`observables`)
//!
//! The crate is a Rust reimplementation of the `NESSi` (the
//! Non-Equilibrium Systems Simulation package) code by Martin Eckstein
//! *et al*; the numerical results match NESSi 2.0.0 point by point.
//!

#![forbid(unsafe_code)]

pub mod basics;
pub mod complex;
pub mod convolution;
pub mod dyson;
pub mod equilibrium;
pub mod fourier;
pub mod io;
pub mod linalg;
pub mod math;
pub mod matsubara;
pub mod observables;
pub mod stage;
pub mod types;
pub mod vie2;
pub mod weights;

/// Crate version as reported by `Cargo.toml`.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// The curated entry surface: scalar and contour scaffolding, the
/// contour-ordered containers, the quadrature integrator, the
/// Matsubara-axis method selector, the Dyson/VIE2/convolution drivers
/// of the standard and moving paths with their reusable workspaces
/// and `_ws` entry points, and the equilibrium and steady-state
/// constructors and solver.
///
/// The surface is workflow-complete and self-consistent: the doctest
/// below runs the real boot sequences (Matsubara solve -> start ->
/// `_ws` time stepping; moving-window init -> `_ws` -> `forward`) on
/// the prelude alone, so no exported function needs a type that the
/// prelude does not list either. The pseudo-particle and
/// singular-kernel solvers stay on their module paths (`dyson::pseudo_*`,
/// `vie2::pseudo_*`, `vie2::vie2_timestep_singular`), as do the
/// power-user items (bubbles, density matrices, near-kernel exports,
/// component-level containers, and the plain-text IO functions). Every
/// serialized flavor implements `Display` and `FromStr`; the moving and
/// pseudo-particle flavors additionally keep their validating fill-style
/// parsers (`moving_green_from_str`, `pseudo_green_from_str`, ...) as
/// named functions.
///
/// ```no_run
/// use kadanoffbaym::prelude::*;
///
/// fn main() {
///     let d = DenseMat::identity(1);
///     let c = Contour::new(8, 16, 1, 1, 0.8, 4.0);
///     let ht = TimeFunction::new(8, 1, 1, Complex::ZERO);
///
///     // standard Dyson workflow: Matsubara -> start -> _ws stepping
///     let mut g = ContourGreen::new(&c, Sign::Fermi, Complex::ZERO);
///     let sig = g.clone();
///     dyson_mat(&mut g, &sig, 0.0, &ht, 4.0, 3, MatsubaraMethod::Fixpoint, true, None, 1e-12);
///     dyson_start(&mut g, 0.0, &ht, &sig, 4.0, 0.1, 3);
///     let mut sg = SteadyGreen::new(SteadyGrid::new(16, 0.1), 1);
///     let ssig = SteadyGreen::new(SteadyGrid::new(16, 0.1), 1);
///     from_dos_steady(
///         &mut sg,
///         Sign::Fermi,
///         &|_: f64| 1.0,
///         4.0,
///         0.0,
///         FftIntegralMethod::Trapez,
///         None,
///         None,
///         0,
///         16,
///     );
///     dyson_steady(
///         &mut sg,
///         0.0,
///         &d,
///         &ssig,
///         FftIntegralMethod::Trapez,
///         DysonBathType::None,
///         0.0,
///         4.0,
///         0.0,
///     );
///     let _ = sigma_bath(0.0, 0.1, DysonBathType::None, 0.0, 4.0, 0.0);
///     let _ = sg.get_ret(0, FftDomain::Time);
///     let _ = sg.get_les(0, FftDomain::Freq);
///
///     let integ = Integrator::new(3);
///
///     // standard VIE2 workflow: Matsubara -> start -> _ws stepping
///     let mut vg = ContourGreen::new(&c, Sign::Fermi, Complex::ZERO);
///     vie2_mat(&mut vg, &sig, &sig, &sig, 4.0, &integ, MatsubaraMethod::Fixpoint, 6, 40, 1e-12);
///     vie2_start(&mut vg, &sig, &sig, &sig, 3, 4.0, 0.1);
///     let mut vws = Vie2Workspace::new(3, 1, 16);
///     vie2_timestep_ws(4, &mut vg, &sig, &sig, &sig, &integ, 4.0, 0.1, &mut vws);
///     let mut cws = ConvWorkspace::new(1, 1, 1, 1);
///     convolution_time_step_ws(1, &mut vg, &sig, &sig, None, &sig, &sig, &integ, 4.0, 0.1, &mut cws);
///
///     // moving workflow: init -> _ws stepping -> forward
///     let mut mh = MovingTimeFunction::new(4, 1);
///     init_from_function(&mut mh, &ht, 4);
///     let mut mg = MovingGreen::new(4, 1, Sign::Fermi);
///     init_from_green_herm(&mut mg, &g, 4);
///     let mut ms = MovingGreen::new(4, 1, Sign::Fermi);
///     init_from_green(&mut ms, &g, &g, 4);
///     let mut mdws = MovingDysonWorkspace::new(3, 1);
///     dyson_timestep_moving_ws(4, &mut mg, 0.0, &mh, &ms, 0.1, &integ, &mut mdws);
///     let mut mvws = MovingVie2Workspace::new(3, 1);
///     vie2_timestep_moving_ws(4, &mut mg, &ms, &ms, &ms, 0.1, &integ, &mut mvws);
///     forward(&mut mg);
///     forward(&mut ms);
/// }
/// ```
pub mod prelude {
    // scalars and the contour
    pub use crate::complex::Complex;
    pub use crate::types::{Contour, Sign, TimeFunction};

    // contour-ordered containers
    pub use crate::types::{ContourGreen, GreenTimeStep};

    // moving-window (truncated) propagation
    pub use crate::stage::{
        extrapolate_timestep_moving, forward, init_from_function, init_from_green,
        init_from_green_herm, init_from_steady,
    };
    pub use crate::types::{MovingGreen, MovingGreenTimeStep, MovingTimeFunction};

    // steady-state (NESS) containers
    pub use crate::types::{FftDomain, SteadyFunction, SteadyGreen, SteadyGrid};

    // dense matrices and quadrature
    pub use crate::linalg::DenseMat;
    pub use crate::weights::Integrator;

    // Matsubara-axis solution method
    pub use crate::matsubara::MatsubaraMethod;

    // Dyson solvers and workspaces (standard and moving paths)
    pub use crate::dyson::dyson_start;
    pub use crate::dyson::dyson_timestep_moving;
    pub use crate::dyson::dyson_timestep_moving_ws;
    pub use crate::dyson::{DysonWorkspace, MovingDysonWorkspace};
    pub use crate::dyson::{dyson, dyson_mat, dyson_timestep, dyson_timestep_ws};

    // VIE2 solvers and workspaces (standard and moving paths)
    pub use crate::vie2::vie2_start;
    pub use crate::vie2::vie2_timestep_moving;
    pub use crate::vie2::vie2_timestep_moving_ws;
    pub use crate::vie2::{MovingVie2Workspace, Vie2Workspace};
    pub use crate::vie2::{vie2, vie2_mat, vie2_timestep, vie2_timestep_ws};

    // convolution drivers
    pub use crate::convolution::ConvWorkspace;
    pub use crate::convolution::{convolution, convolution_time_step, convolution_time_step_ws};

    // equilibrium and steady-state construction and solver
    pub use crate::dyson::{DysonBathType, dyson_steady, sigma_bath};
    pub use crate::equilibrium::{equilibrate_lesser, from_dos_steady};
    pub use crate::equilibrium::{from_dos, from_ham, from_mat, green_single_pole};
    pub use crate::fourier::FftIntegralMethod;
}
