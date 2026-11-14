//
// Model independent stages of the examples (the Rust port of
// julia/tutor/common/contour_loop.jl).
//
// The KB time step n carries t = (n - 1) h; my library steps are the
// NESSi ones (t = n h), so `kb_step` below follows the blueprint and the
// examples convert.
//

/// A Matsubara or bootstrap iteration did not converge.
#[derive(Clone, Debug)]
pub struct ConvergenceError {
    pub stage: &'static str,
    pub iterations: i64,
    pub err: f64,
}

impl std::fmt::Display for ConvergenceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "ConvergenceError: the {} iteration did not converge after {} iterations (error = {})",
            self.stage, self.iterations, self.err
        )
    }
}

impl std::error::Error for ConvergenceError {}

/// An event of the loop (passed to the log callback).
pub struct LoopEvent {
    pub stage: &'static str,
    pub kb_step: i64,
    pub nessi_step: i64,
    pub t: f64,
    pub iter: i64,
    pub err: f64,
}

/// Matsubara iteration: `step!(iter)` performs the iteration `iter = 0,
/// 1, .., max_iter` and returns its error; the loop ends after the first
/// iteration with `err < tol`.
pub fn iterate_matsubara<F>(
    mut step: F,
    max_iter: i64,
    tol: f64,
    mut log: impl FnMut(&LoopEvent),
) -> Result<(i64, f64), ConvergenceError>
where
    F: FnMut(i64) -> f64,
{
    let mut err = f64::NAN;
    for iter in 0..=max_iter {
        err = step(iter);
        log(&LoopEvent {
            stage: "matsubara",
            kb_step: 0,
            nessi_step: -1,
            t: 0.0,
            iter,
            err,
        });
        if err < tol {
            return Ok((iter + 1, err));
        }
    }
    Err(ConvergenceError {
        stage: "matsubara",
        iterations: max_iter + 1,
        err,
    })
}

/// Bootstrapping: the loop ends after the first iteration with
/// `err < tol` and `iter >= min_iterations`.
pub fn iterate_bootstrap<F>(
    mut step: F,
    max_iter: i64,
    tol: f64,
    min_iterations: i64,
    mut log: impl FnMut(&LoopEvent),
) -> Result<(i64, f64), ConvergenceError>
where
    F: FnMut(i64) -> f64,
{
    if min_iterations < 0 {
        return Err(ConvergenceError {
            stage: "bootstrap",
            iterations: 0,
            err: f64::NAN,
        });
    }
    let mut err = f64::NAN;
    for iter in 0..=max_iter {
        err = step(iter);
        log(&LoopEvent {
            stage: "bootstrap",
            kb_step: -1,
            nessi_step: -1,
            t: f64::NAN,
            iter,
            err,
        });
        if err < tol && iter >= min_iterations {
            return Ok((iter + 1, err));
        }
    }
    Err(ConvergenceError {
        stage: "bootstrap",
        iterations: max_iter + 1,
        err,
    })
}

/// Real-time propagation of the KB time steps `first_step..=last_step`:
/// `predictor!(n)` once, then `corrector!(n, iter)` exactly
/// `corrector_steps` times.
pub fn propagate_timesteps<P, C>(
    mut predictor: P,
    mut corrector: C,
    first_step: i64,
    last_step: i64,
    corrector_steps: i64,
    h: f64,
    mut log: impl FnMut(&LoopEvent),
) where
    P: FnMut(i64),
    C: FnMut(i64, i64),
{
    assert!(corrector_steps >= 1, "corrector_steps must be >= 1");
    for n in first_step..=last_step {
        let t = (n - 1) as f64 * h;
        predictor(n);
        log(&LoopEvent {
            stage: "predictor",
            kb_step: n,
            nessi_step: n - 1,
            t,
            iter: -1,
            err: f64::NAN,
        });
        for iter in 0..corrector_steps {
            corrector(n, iter);
            log(&LoopEvent {
                stage: "corrector",
                kb_step: n,
                nessi_step: n - 1,
                t,
                iter,
                err: f64::NAN,
            });
        }
    }
}
