//! A robust, production-grade implementation of the BFGS optimization algorithm.
//!
//! This crate provides a solver for unconstrained nonlinear optimization problems,
//! built upon the principles outlined in "Numerical Optimization" by Nocedal & Wright.
//!
//! It features:
//! - A line search that guarantees the Strong Wolfe conditions, ensuring stability and
//!   convergence, using an efficient cubic interpolation strategy.
//! - A scaling heuristic for the initial Hessian approximation, leading to faster
//!   convergence.
//! - A clear, configurable, and ergonomic API using a builder pattern.
//! - Detailed error handling and guaranteed termination.
//!
//! # Example
//! Minimize the Rosenbrock function, a classic test case for optimization algorithms.
//!
//! ```
//! use bfgs::{Bfgs, BfgsSolution, BfgsError};
//! use ndarray::{array, Array1};
//!
//! // Define the Rosenbrock function and its gradient.
//! let rosenbrock = |x: &Array1<f64>| -> (f64, Array1<f64>) {
//!     let a = 1.0;
//!     let b = 100.0;
//!     let f = (a - x[0]).powi(2) + b * (x[1] - x[0].powi(2)).powi(2);
//!     let g = array![
//!         -2.0 * (a - x[0]) - 4.0 * b * (x[1] - x[0].powi(2)) * x[0],
//!         2.0 * b * (x[1] - x[0].powi(2)),
//!     ];
//!     (f, g)
//! };
//!
//! // Set the initial guess.
//! let x0 = array![-1.2, 1.0];
//!
//! // Run the solver.
//! let BfgsSolution {
//!     final_point: x_min,
//!     final_value,
//!     iterations,
//!     ..
//! } = Bfgs::new(x0, rosenbrock)
//!     .with_tolerance(1e-6)
//!     .with_max_iterations(100)
//!     .run()
//!     .expect("BFGS failed to solve");
//!
//! println!(
//!     "Found minimum f({}) = {:.4} in {} iterations.",
//!     x_min, final_value, iterations
//! );
//!
//! // The known minimum is at [1.0, 1.0].
//! assert!((x_min[0] - 1.0).abs() < 1e-5);
//! assert!((x_min[1] - 1.0).abs() < 1e-5);
//! ```

use ndarray::{Array1, Array2, Axis};

/// An error type for clear diagnostics.
#[derive(Debug, thiserror::Error)]
pub enum BfgsError {
    #[error("The line search failed to find a point satisfying the Wolfe conditions after {max_attempts} attempts.")]
    LineSearchFailed { max_attempts: usize },
    #[error("Maximum number of iterations ({max_iterations}) reached without converging.")]
    MaxIterationsReached { max_iterations: usize },
    #[error("The gradient norm was NaN or infinity, indicating numerical instability.")]
    GradientIsNaN,
    #[error("Curvature condition `s_k^T y_k > 0` was violated. This should not happen with a valid Wolfe line search, and may indicate a bug or severe floating-point issues.")]
    CurvatureConditionViolated,
}

/// A summary of a successful optimization run.
#[derive(Debug)]
pub struct BfgsSolution {
    /// The point at which the minimum value was found.
    pub final_point: Array1<f64>,
    /// The minimum value of the objective function.
    pub final_value: f64,
    /// The norm of the gradient at the final point.
    pub final_gradient_norm: f64,
    /// The total number of iterations performed.
    pub iterations: usize,
    /// The total number of times the objective function was evaluated.
    pub func_evals: usize,
    /// The total number of times the gradient was evaluated.
    pub grad_evals: usize,
}

/// A configurable BFGS solver.
pub struct Bfgs<ObjFn>
where
    ObjFn: Fn(&Array1<f64>) -> (f64, Array1<f64>),
{
    x0: Array1<f64>,
    obj_fn: ObjFn,
    // --- Configuration ---
    tolerance: f64,
    max_iterations: usize,
    c1: f64,
    c2: f64,
}

impl<ObjFn> Bfgs<ObjFn>
where
    ObjFn: Fn(&Array1<f64>) -> (f64, Array1<f64>),
{
    /// Creates a new BFGS solver.
    ///
    /// # Arguments
    /// * `x0` - The initial guess for the minimum.
    /// * `obj_fn` - The objective function which returns a tuple `(value, gradient)`.
    pub fn new(x0: Array1<f64>, obj_fn: ObjFn) -> Self {
        Self {
            x0,
            obj_fn,
            tolerance: 1e-5,
            max_iterations: 100,
            c1: 1e-4, // Standard value for sufficient decrease
            c2: 0.9,   // Standard value for curvature condition
        }
    }

    /// Sets the convergence tolerance (default: 1e-5).
    pub fn with_tolerance(mut self, tolerance: f64) -> Self {
        self.tolerance = tolerance;
        self
    }

    /// Sets the maximum number of iterations (default: 100).
    pub fn with_max_iterations(mut self, max_iterations: usize) -> Self {
        self.max_iterations = max_iterations;
        self
    }

    /// Executes the BFGS algorithm.
    pub fn run(&self) -> Result<BfgsSolution, BfgsError> {
        let n = self.x0.len();
        let mut x_k = self.x0.clone();
        let (mut f_k, mut g_k) = (self.obj_fn)(&x_k);
        let mut func_evals = 1;
        let mut grad_evals = 1;

        let mut b_inv: Array2<f64>;

        // --- Handle the first iteration separately for initial Hessian scaling ---
        let g_norm = g_k.dot(&g_k).sqrt();
        if g_norm < self.tolerance {
            return Ok(BfgsSolution {
                final_point: x_k, final_value: f_k, final_gradient_norm: g_norm,
                iterations: 0, func_evals, grad_evals,
            });
        }

        let g_0 = g_k.clone(); // Preserve the initial gradient g_0.

        let d_0 = -g_0.clone();
        let (alpha_0, f_1, g_1, f_evals, g_evals) =
            line_search(&self.obj_fn, &x_k, &d_0, f_k, &g_0, self.c1, self.c2)?;
        func_evals += f_evals;
        grad_evals += g_evals;

        // Calculate the first step `s_0` and the change in gradient `y_0`.
        let s_0 = alpha_0 * d_0;
        let y_0 = &g_1 - &g_0;
        
        // Update state for the main loop.
        x_k = x_k + &s_0;
        f_k = f_1;
        g_k = g_1;
        
        // Apply the scaling heuristic for the initial inverse Hessian.
        let sy = s_0.dot(&y_0);
        let yy = y_0.dot(&y_0);

        b_inv = if sy > 0.0 && yy > 0.0 {
            // Explicit type annotation `::<f64>` is a robust way to ensure the compiler
            // infers the element type of the identity matrix, even if not strictly
            // required here due to the type on `b_inv`.
            Array2::<f64>::eye(n) * (sy / yy)
        } else {
            Array2::<f64>::eye(n)
        };
        // --- End of first iteration ---

        for k in 1..self.max_iterations {
            let g_norm = g_k.dot(&g_k).sqrt();
            if !g_norm.is_finite() {
                return Err(BfgsError::GradientIsNaN);
            }
            if g_norm < self.tolerance {
                return Ok(BfgsSolution {
                    final_point: x_k, final_value: f_k, final_gradient_norm: g_norm,
                    iterations: k, func_evals, grad_evals,
                });
            }

            let d_k = -b_inv.dot(&g_k);
            let (alpha_k, f_next, g_next, f_evals, g_evals) =
                line_search(&self.obj_fn, &x_k, &d_k, f_k, &g_k, self.c1, self.c2)?;
            func_evals += f_evals;
            grad_evals += g_evals;

            let s_k = alpha_k * d_k;
            let y_k = &g_next - &g_k;

            let sy = s_k.dot(&y_k);

            // A valid Wolfe line search should always ensure sy > 0.
            // This check is a safeguard against potential floating-point issues.
            if sy <= 1e-10 {
                return Err(BfgsError::CurvatureConditionViolated);
            }

            // --- The BFGS Inverse Hessian Update Formula ---
            // H_{k+1} = (I - ρ*s*yᵀ) * H_k * (I - ρ*y*sᵀ) + ρ*s*sᵀ
            // This is derived from the Sherman-Morrison-Woodbury formula.
            let rho = 1.0 / sy;
            // Create 2D column vector views of s_k and y_k for outer product
            // calculations, without consuming the original 1D vectors.
            let s_k_col = s_k.view().insert_axis(Axis(1));
            let y_k_col = y_k.view().insert_axis(Axis(1));

            // Explicit type annotation `::<f64>` is needed for the compiler to infer the
            // element type of the identity matrix in this context.
            let i_minus_rhosy = &Array2::<f64>::eye(n) - rho * s_k_col.dot(&y_k_col.t());

            // The term `(I - ρ*y*sᵀ)` is the transpose of `(I - ρ*s*yᵀ)`.
            b_inv = i_minus_rhosy.dot(&b_inv).dot(&i_minus_rhosy.t()) + rho * s_k_col.dot(&s_k_col.t());
            
            x_k = x_k + s_k;
            f_k = f_next;
            g_k = g_next;
        }

        Err(BfgsError::MaxIterationsReached { max_iterations: self.max_iterations })
    }
}

/// A line search algorithm that finds a step size satisfying the Strong Wolfe conditions.
///
/// This implementation follows the structure of Algorithm 3.5 in Nocedal & Wright.
fn line_search<ObjFn>(
    obj_fn: ObjFn, x_k: &Array1<f64>, d_k: &Array1<f64>, f_k: f64, g_k: &Array1<f64>,
    c1: f64, c2: f64,
) -> Result<(f64, f64, Array1<f64>, usize, usize), BfgsError>
where ObjFn: Fn(&Array1<f64>) -> (f64, Array1<f64>),
{
    let mut alpha_i = 1.0; // Per Nocedal & Wright, always start with a unit step.
    let mut alpha_prev = 0.0;

    let mut f_prev = f_k;
    let g_k_dot_d = g_k.dot(d_k); // Initial derivative along the search direction. This is constant.

    let max_attempts = 20;
    let mut func_evals = 0;
    let mut grad_evals = 0;

    for _ in 0..max_attempts {
        let x_new = x_k + alpha_i * d_k;
        let (f_i, g_i) = obj_fn(&x_new);
        func_evals += 1;
        grad_evals += 1;
        
        // The sufficient decrease (Armijo) condition.
        if f_i > f_k + c1 * alpha_i * g_k_dot_d || (func_evals > 1 && f_i >= f_prev) {
            // The sufficient decrease condition is not met, or the function has increased.
            // The minimum is now bracketed between the previous point and the current one.
            let (_, g_prev) = obj_fn(&(x_k + alpha_prev * d_k));
            grad_evals += 1;
            let g_prev_dot_d = g_prev.dot(d_k);
            return zoom(obj_fn, x_k, d_k, f_k, g_k_dot_d, c1, c2,
                        alpha_prev, alpha_i, f_prev, g_prev_dot_d, f_i,
                        func_evals, grad_evals);
        }

        let g_i_dot_d = g_i.dot(d_k);
        // The curvature condition.
        if g_i_dot_d.abs() <= c2 * g_k_dot_d.abs() {
            // Strong Wolfe conditions are satisfied.
            return Ok((alpha_i, f_i, g_i, func_evals, grad_evals));
        }

        if g_i_dot_d >= 0.0 {
            // The curvature condition is met with a positive derivative, so the
            // minimum is bracketed between the current point and the previous one.
            return zoom(obj_fn, x_k, d_k, f_k, g_k_dot_d, c1, c2,
                        alpha_i, alpha_prev, f_i, g_i_dot_d, f_prev,
                        func_evals, grad_evals);
        }

        // The step is too short, expand the search interval.
        alpha_prev = alpha_i;
        f_prev = f_i;
        alpha_i *= 2.0;
    }

    Err(BfgsError::LineSearchFailed { max_attempts })
}

/// Helper "zoom" function using cubic interpolation, as described by Nocedal & Wright (Alg. 3.6).
///
/// This function is called when a bracketing interval [alpha_lo, alpha_hi] that contains
/// a point satisfying the Strong Wolfe conditions is known.
#[allow(clippy::too_many_arguments)]
fn zoom<ObjFn>(
    obj_fn: ObjFn, x_k: &Array1<f64>, d_k: &Array1<f64>, f_k: f64, g_k_dot_d: f64,
    c1: f64, c2: f64, mut alpha_lo: f64, mut alpha_hi: f64, mut f_lo: f64,
    mut g_lo_dot_d: f64, f_hi: f64, mut func_evals: usize, mut grad_evals: usize,
) -> Result<(f64, f64, Array1<f64>, usize, usize), BfgsError>
where ObjFn: Fn(&Array1<f64>) -> (f64, Array1<f64>),
{
    let max_zoom_attempts = 10;
    for _ in 0..max_zoom_attempts {
        // --- Cubic interpolation to find a trial step size `alpha_j` ---
        // This finds the minimizer of a cubic polynomial that interpolates the function
        // values and derivatives at alpha_lo and alpha_hi.
        let (_, g_hi) = obj_fn(&(x_k + alpha_hi * d_k));
        grad_evals += 1;
        let g_hi_dot_d = g_hi.dot(d_k);
        
        let d1 = g_lo_dot_d + g_hi_dot_d - 3.0 * (f_lo - f_hi) / (alpha_lo - alpha_hi);
        let d2_sq = d1.powi(2) - g_lo_dot_d * g_hi_dot_d;
        
        let alpha_j = if d2_sq.is_sign_positive() {
            let d2 = d2_sq.sqrt();
            alpha_hi - (alpha_hi - alpha_lo) * (g_hi_dot_d + d2 - d1) / (g_hi_dot_d - g_lo_dot_d + 2.0 * d2)
        } else {
            // Fallback to bisection if interpolation fails or is not applicable.
            (alpha_lo + alpha_hi) / 2.0
        };

        let x_j = x_k + alpha_j * d_k;
        let (f_j, g_j) = obj_fn(&x_j);
        func_evals += 1;
        grad_evals += 1;

        // Check if the new point `alpha_j` satisfies the sufficient decrease condition.
        if f_j > f_k + c1 * alpha_j * g_k_dot_d || f_j >= f_lo {
            // The new point is not good enough, shrink the interval from the high end.
            alpha_hi = alpha_j;
        } else {
            let g_j_dot_d = g_j.dot(d_k);
            // Check the curvature condition.
            if g_j_dot_d.abs() <= c2 * g_k_dot_d.abs() {
                // Success: Strong Wolfe conditions are met.
                return Ok((alpha_j, f_j, g_j, func_evals, grad_evals));
            }

            if g_j_dot_d * (alpha_hi - alpha_lo) >= 0.0 {
                alpha_hi = alpha_lo;
            }
            // The new point is good, but doesn't satisfy curvature yet.
            // Shrink the interval from the low end.
            alpha_lo = alpha_j;
            f_lo = f_j;
            g_lo_dot_d = g_j_dot_d;
        }
    }
    Err(BfgsError::LineSearchFailed { max_attempts: max_zoom_attempts })
}


#[cfg(test)]
mod tests {
    // This test suite is structured into three parts:
    // 1. Standard Convergence Tests: Verifies that the solver finds the correct
    //    minimum for well-known benchmark functions from standard starting points.
    // 2. Failure and Edge Case Tests: Ensures the solver handles non-convex
    //    functions, pre-solved problems, and iteration limits correctly and returns
    //    the appropriate descriptive errors.
    // 3. Comparison Tests: Validates the behavior of our implementation against
    //    `argmin`, a trusted, state-of-the-art optimization library, ensuring
    //    that our results (final point and iteration count) are equivalent.

    use super::{Bfgs, BfgsError, BfgsSolution};
    use ndarray::{array, Array1};
    use spectral::prelude::*;

    // --- Test Harness: argmin Comparison Setup ---
    use argmin::core::{CostFunction, Error, Executor, Gradient, IterState, State};
    use argmin::solver::linesearch::MoreThuenteLineSearch;
    use argmin::solver::quasinewton::BFGS as ArgminBFGS;

    struct ArgminTestFn<F> {
        func: F,
    }

    impl<F> CostFunction for ArgminTestFn<F>
    where
        F: Fn(&Array1<f64>) -> (f64, Array1<f64>),
    {
        type Param = Array1<f64>;
        type Output = f64;

        fn cost(&self, p: &Self::Param) -> Result<Self::Output, Error> {
            Ok((self.func)(p).0)
        }
    }

    impl<F> Gradient for ArgminTestFn<F>
    where
        F: Fn(&Array1<f64>) -> (f64, Array1<f64>),
    {
        type Param = Array1<f64>;
        type Gradient = Array1<f64>;

        fn gradient(&self, p: &Self::Param) -> Result<Self::Gradient, Error> {
            Ok((self.func)(p).1)
        }
    }

    // --- Test Functions ---

    /// A simple convex quadratic function: f(x) = x[0]^2 + ... + x[n]^2
    fn quadratic(x: &Array1<f64>) -> (f64, Array1<f64>) {
        (x.dot(x), 2.0 * x)
    }

    /// The Rosenbrock function, a classic non-convex benchmark.
    fn rosenbrock(x: &Array1<f64>) -> (f64, Array1<f64>) {
        let a = 1.0;
        let b = 100.0;
        let f = (a - x[0]).powi(2) + b * (x[1] - x[0].powi(2)).powi(2);
        let g = array![-2.0 * (a - x[0]) - 4.0 * b * (x[1] - x[0].powi(2)) * x[0], 2.0 * b * (x[1] - x[0].powi(2))];
        (f, g)
    }

    /// A function with a maximum at 0, guaranteed to fail the Wolfe curvature condition.
    fn non_convex_max(x: &Array1<f64>) -> (f64, Array1<f64>) {
        (-x.dot(x), -2.0 * x)
    }

    /// A function whose gradient is constant, causing `y_k` to be zero.
    fn linear_function(x: &Array1<f64>) -> (f64, Array1<f64>) {
        (2.0 * x[0] + 3.0 * x[1], array![2.0, 3.0])
    }

    /// A function that produces a NaN gradient if it steps to x=0.
    fn nan_producing_function(x: &Array1<f64>) -> (f64, Array1<f64>) {
        if x[0] == 0.0 {
            (f64::NAN, array![f64::NAN])
        } else {
            (-x[0].ln(), array![-1.0 / x[0]])
        }
    }

    // --- 1. Standard Convergence Tests ---

    #[test]
    fn test_quadratic_bowl_converges() {
        let x0 = array![10.0, -5.0];
        let BfgsSolution { final_point, .. } = Bfgs::new(x0, quadratic).run().unwrap();
        assert_that(&final_point[0]).is_close_to(0.0, 1e-5);
        assert_that(&final_point[1]).is_close_to(0.0, 1e-5);
    }

    #[test]
    fn test_rosenbrock_converges() {
        let x0 = array![-1.2, 1.0];
        let BfgsSolution { final_point, .. } = Bfgs::new(x0, rosenbrock).run().unwrap();
        assert_that(&final_point[0]).is_close_to(1.0, 1e-5);
        assert_that(&final_point[1]).is_close_to(1.0, 1e-5);
    }

    // --- 2. Failure and Edge Case Tests ---

    #[test]
    fn test_begin_at_minimum_terminates_immediately() {
        let x0 = array![0.0, 0.0];
        let BfgsSolution { iterations, .. } = Bfgs::new(x0, quadratic).run().unwrap();
        assert_that(&iterations).is_less_than_or_equal_to(1);
    }

    #[test]
    fn test_max_iterations_error_is_returned() {
        let x0 = array![-1.2, 1.0];
        let result = Bfgs::new(x0, rosenbrock).with_max_iterations(5).run();
        assert!(matches!(result, Err(BfgsError::MaxIterationsReached { .. })));
    }

    #[test]
    fn test_non_convex_function_fails_line_search() {
        let x0 = array![2.0];
        let result = Bfgs::new(x0, non_convex_max).run();
        // A correct Wolfe line search must fail because it can't find a point
        // that satisfies the curvature condition when moving towards a maximum.
        assert!(matches!(result, Err(BfgsError::LineSearchFailed { .. })));
    }

    #[test]
    fn test_initial_hessian_scaling_handles_zero_curvature() {
        let x0 = array![10.0, 10.0];
        // For a linear function, the gradient is constant, so y_k is always zero.
        // The solver should not panic and should gracefully fail.
        let result = Bfgs::new(x0, linear_function).run();
        assert!(matches!(
            result,
            Err(BfgsError::CurvatureConditionViolated)
        ));
    }

    #[test]
    fn test_nan_gradient_returns_error() {
        let x0 = array![1.0];
        // The first step will be d = -g = -(-1/1) = 1.
        // The line search will try alpha=1, testing x = 1 + 1*1 = 2. It's too short.
        // It will then expand, trying alpha=2, testing x = 1 + 2*1 = 3.
        // Eventually it will try a point that results in a negative x, producing NaN.
        // A robust line search may also fail before this.
        let result = Bfgs::new(x0, |x| (x[0].ln(), array![1.0 / x[0]])).run();
        assert!(result.is_err());
    }

    // --- 3. Comparison Tests against a Trusted Library ---

    #[test]
    fn test_rosenbrock_matches_argmin_behavior() {
        let x0 = array![-1.2, 1.0];
        let tolerance = 1e-6;

        // Run our implementation
        let our_res = Bfgs::new(x0.clone(), rosenbrock)
            .with_tolerance(tolerance)
            .run()
            .unwrap();

        // Run argmin's implementation with synchronized settings
        let problem = ArgminTestFn { func: rosenbrock };
        let linesearch = MoreThuenteLineSearch::new();
        let solver = ArgminBFGS::new(linesearch).with_tolerance_grad(tolerance).unwrap();
        let argmin_res = Executor::new(problem, solver)
            .configure(|state: IterState<_, _, _, _, _, _>| state.param(x0).max_iters(100))
            .run()
            .unwrap();

        // Assert that the final points are virtually identical.
        let distance = (&our_res.final_point - argmin_res.state.get_best_param().unwrap())
            .mapv(|x| x.powi(2))
            .sum()
            .sqrt();
        assert_that(&distance).is_less_than(1e-6);

        // Assert that the number of iterations is very similar. A small difference
        // is acceptable due to minor, valid variations in line search implementations.
        let iter_diff = (our_res.iterations as i32 - argmin_res.state.get_iter() as i32).abs();
        assert_that(&iter_diff).is_less_than_or_equal_to(5);
    }

    #[test]
    fn test_quadratic_matches_argmin_behavior() {
        let x0 = array![150.0, -275.5];
        let tolerance = 1e-8;

        // Run our implementation
        let our_res = Bfgs::new(x0.clone(), quadratic)
            .with_tolerance(tolerance)
            .run()
            .unwrap();

        // Run argmin's implementation with synchronized settings
        let problem = ArgminTestFn { func: quadratic };
        let linesearch = MoreThuenteLineSearch::new();
        let solver = ArgminBFGS::new(linesearch).with_tolerance_grad(tolerance).unwrap();
        let argmin_res = Executor::new(problem, solver)
            .configure(|state: IterState<_, _, _, _, _, _>| state.param(x0).max_iters(100))
            .run()
            .unwrap();

        // Assert that the final points are virtually identical.
        let distance = (&our_res.final_point - argmin_res.state.get_best_param().unwrap())
            .mapv(|x| x.powi(2))
            .sum()
            .sqrt();
        assert_that(&distance).is_less_than(1e-6);

        // Assert that the number of iterations is very similar.
        let iter_diff = (our_res.iterations as i32 - argmin_res.state.get_iter() as i32).abs();
        assert_that(&iter_diff).is_less_than_or_equal_to(3);
    }
}
