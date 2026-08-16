//! 1D Poisson problem solved with linear (P1) finite elements.
//!
//! ```text
//! -u''(x) = pi^2 sin(pi x)   on (0, 1),   u(0) = u(1) = 0
//! ```
//!
//! whose exact solution is u(x) = sin(pi x).
//!
//! The crate is split so that the maths and the reporting stay apart:
//!
//! * [`sparse`] -- triplet assembly, CSR storage, a tridiagonal direct solver
//!   and conjugate gradient. All hand-written, no dependencies.
//! * [`fem`] -- mesh, element matrices, scatter-add assembly, boundary
//!   conditions, full solves and error norms.
//!
//! The binary in `src/main.rs` uses these to print the convergence tables.

pub mod fem;
pub mod sparse;
