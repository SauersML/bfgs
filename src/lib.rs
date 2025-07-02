#![feature(test)]
//! This package contains an implementation of the
//! [BFGS](https://en.wikipedia.org/wiki/BFGS_method) algorithm, a classic method for
//! solving unconstrained nonlinear optimization problems.
//!
//! It is a quasi-Newton method that approximates the inverse Hessian matrix of the
//! target function to find the minimum.
//!
//! BFGS is explained at a high level in
//! [the blog post](https://paulkernfeld.com/2018/08/06/rust-needs-bfgs.html) that
//! introduced this package.
//!
//! # Example
//! In this example, we minimize the simple quadratic function `f(x) = xᵀx`,
//! which has a minimum at `x = [0, 0]`.
//!
//! ```
//! use bfgs::bfgs;
//! use ndarray::{array, Array1};
//!
//! // Define the objective function and its gradient.
//! let f = |x: &Array1<f64>| x.dot(x);
//! let g = |x: &Array1<f64>| 2.0 * x;
//!
//! // Choose an arbitrary starting point.
//! let x0 = array![8.888, 1.234];
//!
//! // Run the optimizer.
//! let x_min = bfgs(x0, f, g).expect("BFGS failed to find a solution");
//!
//! // Check that the result is close to the known minimum.
//! let expected = array![0.0, 0.0];
//! let distance = x_min.iter().zip(expected.iter()).map(|(a, b)| (a - b).powi(2)).sum::<f64>().sqrt();
//! assert!(distance < 1e-5);
//! ```

use ndarray::{Array1, Array2, Zip};
use std::f64::INFINITY;

const F64_MACHINE_EPSILON: f64 = 2e-53;

// From "Numerical Optimization" by Nocedal & Wright. A typical value for the stopping
// condition based on function value improvement.
const FACTR: f64 = 1e7;

// The tolerance for the stopping criterion.
const F_TOLERANCE: f64 = FACTR * F64_MACHINE_EPSILON;

/// A primitive line search that tries a fixed set of step sizes.
///
/// This method iterates through a predefined range of powers of 2 to find the
/// step size `epsilon` that minimizes the objective function `f` along the search
/// direction. It is simple but not guaranteed to satisfy Wolfe conditions.
///
/// Returns the best step size found, or an error if no improvement is made.
fn line_search<F>(f: F) -> Result<f64, ()>
where
    F: Fn(f64) -> f64,
{
    let mut best_epsilon = 0.0;
    let mut best_val_f = INFINITY;

    for i in -20..20 {
        let epsilon = 2.0_f64.powi(i);
        let val_f = f(epsilon);
        if val_f < best_val_f {
            best_epsilon = epsilon;
            best_val_f = val_f;
        }
    }

    if best_epsilon > 0.0 {
        Ok(best_epsilon)
    } else {
        // No step size resulted in an improvement.
        Err(())
    }
}

/// Creates an identity matrix of a given size.
fn new_identity_matrix(size: usize) -> Array2<f64> {
    Array2::eye(size)
}

/// Checks the stopping criterion based on the relative reduction in the function value.
///
/// The algorithm stops if `(f_k - f_{k+1}) / max(|f_k|, |f_{k+1}|, 1) <= F_TOLERANCE`.
/// This criterion is from the original L-BFGS paper (Zhu et al., 1994).
fn stop(f_x_old: f64, f_x: f64) -> bool {
    let relative_improvement = (f_x_old - f_x) / f_x_old.abs().max(f_x.abs()).max(1.0);
    relative_improvement <= F_TOLERANCE
}

/// Returns a value of `x` that minimizes `f` using the BFGS algorithm.
///
/// `f` must be convex and twice-differentiable for the algorithm to be guaranteed
/// to converge.
///
/// # Arguments
/// * `x0` - An initial guess for the minimum.
/// * `f` - The objective function to minimize, `fn(&Array1<f64>) -> f64`.
/// * `g` - The gradient of the objective function, `fn(&Array1<f64>) -> Array1<f64>`.
///
/// # Returns
/// `Ok(Array1<f64>)` containing the approximate minimum, or `Err(())` if the
/// algorithm fails to converge.
#[allow(clippy::many_single_char_names)]
pub fn bfgs<F, G>(x0: Array1<f64>, f: F, g: G) -> Result<Array1<f64>, ()>
where
    F: Fn(&Array1<f64>) -> f64,
    G: Fn(&Array1<f64>) -> Array1<f64>,
{
    let mut x = x0;
    let mut f_x = f(&x);
    let mut g_x = g(&x);
    let p = x.len();
    assert_eq!(g_x.dim(), p, "Dimension of gradient must match dimension of x");

    // Initialize the inverse approximate Hessian to the identity matrix.
    let mut b_inv = new_identity_matrix(p);

    loop {
        // Check for convergence before starting the next iteration.
        // This handles cases where the initial guess is already the minimum.
        if g_x.dot(&g_x).sqrt() < 1e-5 {
            return Ok(x);
        }

        // Find the search direction by multiplying the inverse Hessian by the negative gradient.
        let search_dir = -b_inv.dot(&g_x);

        // Find a suitable step size `epsilon` along the search direction.
        let epsilon = line_search(|eps| f(&(eps * &search_dir + &x))).map_err(|_| ())?;

        // Store the state from the previous iteration.
        let f_x_old = f_x;
        let g_x_old = g_x;

        // Update the position `x` and re-evaluate the function and gradient.
        // This replaces the deprecated `scaled_add` with the modern `zip_mut_with`.
        x.zip_mut_with(&search_dir, |val_x, &val_search| {
            *val_x += epsilon * val_search
        });
        f_x = f(&x);
        g_x = g(&x);

        // Check the stopping criterion based on function value improvement.
        if stop(f_x_old, f_x) {
            return Ok(x);
        }

        // Compute the change in position (s) and gradient (y).
        let s: Array2<f64> = (epsilon * &search_dir)
            .into_shape((p, 1))
            .expect("BFGS internal: Failed to reshape step delta `s`");
        let y: Array2<f64> = (&g_x - &g_x_old)
            .into_shape((p, 1))
            .expect("BFGS internal: Failed to reshape gradient delta `y`");

        // Calculate sᵀy. This is the dot product of the change in position and change in gradient.
        // Replaces `...[()]` with the modern `.into_scalar()`.
        let sy: f64 = s.t().dot(&y).into_scalar();

        // The curvature condition: sᵀy must be positive for the Hessian update to be
        // stable and maintain positive-definiteness. If not met, we cannot continue.
        if sy <= 1e-10 {
            return Err(());
        }

        // Update the inverse Hessian approximation using the BFGS formula.
        // The formula is broken down into terms for clarity and correctness.
        // B_{k+1}^{-1} = B_k^{-1} + term1 - term2
        let b_inv_y = b_inv.dot(&y);
        let y_t_b_inv = y.t().dot(&b_inv);
        let y_t_b_inv_y = y_t_b_inv.dot(&y).into_scalar();

        let term1_factor = (sy + y_t_b_inv_y) / sy.powi(2);
        let term1 = s.dot(&s.t()) * term1_factor;

        let term2 = (b_inv_y.dot(&s.t()) + s.dot(&y_t_b_inv)) / sy;

        b_inv = b_inv + term1 - term2;
    }
}

// ====== Test and Benchmark Modules ======

#[cfg(test)]
mod benchmark;

#[cfg(test)]
mod tests {
    use crate::bfgs;
    use ndarray::{array, Array1};
    use spectral::assert_that;
    use spectral::numeric::OrderedAssertions;

    fn l2_distance(xs: &Array1<f64>, ys: &Array1<f64>) -> f64 {
        xs.iter().zip(ys.iter()).map(|(x, y)| (y - x).powi(2)).sum()
    }

    #[test]
    fn test_x_squared_1d() {
        let x0 = array![2.0];
        let f = |x: &Array1<f64>| x.iter().map(|xx| xx * xx).sum();
        let g = |x: &Array1<f64>| 2.0 * x;
        let x_min = bfgs(x0, f, g);
        assert_eq!(x_min, Ok(array![0.0]));
    }

    #[test]
    fn test_begin_at_minimum() {
        let x0 = array![0.0];
        let f = |x: &Array1<f64>| x.iter().map(|xx| xx * xx).sum();
        let g = |x: &Array1<f64>| 2.0 * x;
        let x_min = bfgs(x0, f, g);
        assert_eq!(x_min, Ok(array![0.0]));
    }

    #[test]
    fn test_negative_x_squared() {
        let x0 = array![2.0];
        let f = |x: &Array1<f64>| x.iter().map(|xx| -xx * xx).sum();
        let g = |x: &Array1<f64>| -2.0 * x;
        let x_min = bfgs(x0, f, g);
        // The algorithm should fail because the curvature condition is not met for a maximum.
        assert_eq!(x_min, Err(()));
    }

    #[test]
    fn test_rosenbrock() {
        let x0 = array![0.0, 0.0];
        let f = |x: &Array1<f64>| (1.0 - x[0]).powi(2) + 100.0 * (x[1] - x[0].powi(2)).powi(2);
        let g = |x: &Array1<f64>| {
            array![
                -400.0 * (x[1] - x[0].powi(2)) * x[0] - 2.0 * (1.0 - x[0]),
                200.0 * (x[1] - x[0].powi(2)),
            ]
        };
        let x_min = bfgs(x0, f, g).expect("Rosenbrock test failed");
        assert_that(&l2_distance(&x_min, &array![1.0, 1.0])).is_less_than(&0.01);
    }
}
