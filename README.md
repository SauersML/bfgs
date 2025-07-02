# bfgs

A pure Rust implementation of the [BFGS optimization algorithm](https://en.wikipedia.org/wiki/Broyden%E2%80%93Fletcher%E2%80%93Goldfarb%E2%80%93Shanno_algorithm) for unconstrained nonlinear optimization problems.

## Features

- **Strong Wolfe Line Search**: Ensures stability and global convergence for convex problems
- **Scaling Heuristic**: Improves convergence rate with intelligent initial Hessian approximation
- **Ergonomic API**: Clean, configurable interface using the builder pattern
- **Robust Error Handling**: Comprehensive error diagnostics with descriptive messages
- **Well Tested**: Extensive test suite including comparison against SciPy's implementation

## Usage

Add this to your `Cargo.toml`:

```toml
[dependencies]
bfgs = "0.1.0"
```

### Example: Minimizing the Rosenbrock Function

```rust
use bfgs::{Bfgs, BfgsSolution};
use ndarray::{array, Array1};

// Define the Rosenbrock function and its gradient
let rosenbrock = |x: &Array1<f64>| -> (f64, Array1<f64>) {
    let a = 1.0;
    let b = 100.0;
    let f = (a - x[0]).powi(2) + b * (x[1] - x[0].powi(2)).powi(2);
    let g = array![
        -2.0 * (a - x[0]) - 4.0 * b * (x[1] - x[0].powi(2)) * x[0],
        2.0 * b * (x[1] - x[0].powi(2)),
    ];
    (f, g)
};

// Set the initial guess
let x0 = array![-1.2, 1.0];

// Run the optimizer
let BfgsSolution {
    final_point: x_min,
    final_value,
    iterations,
    ..
} = Bfgs::new(x0, rosenbrock)
    .with_tolerance(1e-6)
    .with_max_iterations(100)
    .run()
    .expect("BFGS failed to solve");

println!(
    "Found minimum f({}) = {:.4} in {} iterations.",
    x_min, final_value, iterations
);

// The known minimum is at [1.0, 1.0]
assert!((x_min[0] - 1.0).abs() < 1e-5);
assert!((x_min[1] - 1.0).abs() < 1e-5);
```

## API

The main entry point is the `Bfgs` struct, which uses the builder pattern for configuration:

- `Bfgs::new(x0, objective_function)` - Create a new optimizer
- `.with_tolerance(tol)` - Set convergence tolerance (default: 1e-5)
- `.with_max_iterations(max_iter)` - Set maximum iterations (default: 100)
- `.run()` - Execute the optimization

The objective function should have the signature `Fn(&Array1<f64>) -> (f64, Array1<f64>)`, returning both the function value and its gradient.

## Testing

Run the test suite with:

```bash
cargo test
```

The tests include comparison against SciPy's BFGS implementation using a Python harness to ensure correctness.

## License

Licensed under either of:
- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.
