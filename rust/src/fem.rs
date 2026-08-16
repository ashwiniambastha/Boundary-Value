//! The finite element method itself: mesh, element matrices, assembly,
//! boundary conditions and error norms.
//!
//! This mirrors `python/fem1d.py` function for function, on purpose. The two
//! implementations are checked against each other, so keeping them structurally
//! the same makes any disagreement easy to localise.
//!
//! Problem:  -u''(x) = pi^2 sin(pi x) on (0, 1),  u(0) = u(1) = 0,
//! exact solution u(x) = sin(pi x).

use crate::sparse::{conjugate_gradient, CsrMatrix, Triplet};
use std::f64::consts::PI;

// ---------------------------------------------------------------------------
// The problem data
// ---------------------------------------------------------------------------
pub fn source_term(x: f64) -> f64 {
    PI * PI * (PI * x).sin()
}

pub fn exact_solution(x: f64) -> f64 {
    (PI * x).sin()
}

pub fn exact_derivative(x: f64) -> f64 {
    PI * (PI * x).cos()
}

// ---------------------------------------------------------------------------
// Gauss-Legendre quadrature on the reference interval [-1, 1].
//
// A k-point rule is exact for polynomials of degree up to 2k-1. Two points is
// plenty for the load vector; five points is used for error measurement so the
// measurement itself never limits the accuracy of what we report.
// ---------------------------------------------------------------------------
const GAUSS_2_POINTS: [f64; 2] = [-0.577_350_269_189_625_8, 0.577_350_269_189_625_8];
const GAUSS_2_WEIGHTS: [f64; 2] = [1.0, 1.0];

const GAUSS_5_POINTS: [f64; 5] = [
    -0.906_179_845_938_664,
    -0.538_469_310_105_683,
    0.0,
    0.538_469_310_105_683,
    0.906_179_845_938_664,
];
const GAUSS_5_WEIGHTS: [f64; 5] = [
    0.236_926_885_056_189,
    0.478_628_670_499_366,
    0.568_888_888_888_889,
    0.478_628_670_499_366,
    0.236_926_885_056_189,
];

/// Map a reference rule onto [x_left, x_right], returning physical points and
/// weights that already include the h/2 Jacobian factor.
fn map_rule(
    reference_points: &[f64],
    reference_weights: &[f64],
    x_left: f64,
    x_right: f64,
) -> (Vec<f64>, Vec<f64>) {
    let half_length = 0.5 * (x_right - x_left);
    let midpoint = 0.5 * (x_left + x_right);

    let points = reference_points.iter().map(|p| midpoint + half_length * p).collect();
    let weights = reference_weights.iter().map(|w| half_length * w).collect();
    (points, weights)
}

// ---------------------------------------------------------------------------
// Mesh
// ---------------------------------------------------------------------------
/// Node coordinates for `n_elements` equal elements covering (0, 1).
///
/// The connectivity of a uniform 1D mesh is implicit -- element e joins nodes e
/// and e+1 -- so unlike the Python version we do not store it. The assembly
/// loop below still goes through explicit `global_indices`, so it reads the
/// same way.
pub fn uniform_mesh(n_elements: usize) -> Vec<f64> {
    assert!(n_elements >= 1, "need at least one element");
    let h = 1.0 / n_elements as f64;
    (0..=n_elements).map(|i| i as f64 * h).collect()
}

// ---------------------------------------------------------------------------
// Element matrices
// ---------------------------------------------------------------------------
/// Local stiffness matrix for an element of length h: (1/h) * [[1, -1], [-1, 1]].
///
/// On an element the two basis functions are the straight lines
/// N_0 = (x_right - x)/h and N_1 = (x - x_left)/h, with constant derivatives
/// -1/h and +1/h. Entry (i, j) is the integral of N_i' N_j' over the element,
/// and since the integrand is constant that is just (+/-1/h)(+/-1/h)*h.
pub fn element_stiffness(h: f64) -> [[f64; 2]; 2] {
    let inverse_h = 1.0 / h;
    [[inverse_h, -inverse_h], [-inverse_h, inverse_h]]
}

/// Local load vector: the integral of f * N_i over one element, by 2-point
/// Gauss-Legendre quadrature.
///
/// See the Python module for the full reasoning. Short version: 2-point Gauss
/// is exact up to cubics, and its error on an element scales like h^5, well
/// below the h^2 discretisation error we are trying to measure. The trapezoid
/// rule would instead add its own O(h^2) error and contaminate the result.
pub fn element_load(x_left: f64, x_right: f64, source: fn(f64) -> f64) -> [f64; 2] {
    let (points, weights) = map_rule(&GAUSS_2_POINTS, &GAUSS_2_WEIGHTS, x_left, x_right);
    let h = x_right - x_left;

    let mut load = [0.0; 2];
    for (point, weight) in points.iter().zip(weights.iter()) {
        let shape_left = (x_right - point) / h;
        let shape_right = (point - x_left) / h;
        let f_value = source(*point);

        load[0] += weight * f_value * shape_left;
        load[1] += weight * f_value * shape_right;
    }
    load
}

// ---------------------------------------------------------------------------
// Assembly
// ---------------------------------------------------------------------------
/// Build the global stiffness matrix and load vector by scatter-add.
///
/// For each element we compute the 2x2 local matrix and add each of its four
/// entries into the global system at the position given by the element's global
/// node numbers. Contributions are collected as triplets; converting to CSR
/// sums the duplicates, and that summation is the assembly.
pub fn assemble_system(nodes: &[f64], source: fn(f64) -> f64) -> (CsrMatrix, Vec<f64>) {
    let n_nodes = nodes.len();
    let n_elements = n_nodes - 1;

    let mut triplets = Vec::with_capacity(4 * n_elements);
    let mut load = vec![0.0; n_nodes];

    for element in 0..n_elements {
        let global_indices = [element, element + 1];
        let x_left = nodes[global_indices[0]];
        let x_right = nodes[global_indices[1]];
        let h = x_right - x_left;

        let local_stiffness = element_stiffness(h);
        let local_load = element_load(x_left, x_right, source);

        for i in 0..2 {
            load[global_indices[i]] += local_load[i];
            for j in 0..2 {
                triplets.push(Triplet {
                    row: global_indices[i],
                    column: global_indices[j],
                    value: local_stiffness[i][j],
                });
            }
        }
    }

    let stiffness = CsrMatrix::from_triplets(&triplets, n_nodes, n_nodes);
    (stiffness, load)
}

// ---------------------------------------------------------------------------
// Boundary conditions
// ---------------------------------------------------------------------------
/// Impose u(0) = u(1) = 0 by deleting the two boundary rows and columns.
///
/// The boundary values are known, so they are not unknowns. Removing them keeps
/// the matrix symmetric positive definite, which conjugate gradient requires.
/// With zero boundary data there is no lifting term to move to the right-hand
/// side; for non-zero data we would subtract A[interior, boundary] * u_boundary.
pub fn apply_homogeneous_dirichlet(
    stiffness: &CsrMatrix,
    load: &[f64],
) -> (CsrMatrix, Vec<f64>, Vec<usize>) {
    let n_nodes = stiffness.n_rows();
    let interior: Vec<usize> = (1..n_nodes - 1).collect();

    let reduced_stiffness = stiffness.submatrix(&interior);
    let reduced_load = interior.iter().map(|&index| load[index]).collect();

    (reduced_stiffness, reduced_load, interior)
}

/// Put interior unknowns back into a full nodal vector with zeros at the ends.
pub fn scatter_interior_solution(
    interior_values: &[f64],
    n_nodes: usize,
    interior_indices: &[usize],
) -> Vec<f64> {
    let mut full = vec![0.0; n_nodes];
    for (position, &node) in interior_indices.iter().enumerate() {
        full[node] = interior_values[position];
    }
    full
}

// ---------------------------------------------------------------------------
// Full solves
// ---------------------------------------------------------------------------
/// Assemble, apply boundary conditions, and solve directly with the Thomas
/// algorithm. Returns the nodes and the nodal values of the FE solution.
pub fn solve_poisson_direct(n_elements: usize) -> (Vec<f64>, Vec<f64>) {
    let nodes = uniform_mesh(n_elements);
    let (stiffness, load) = assemble_system(&nodes, source_term);
    let (reduced_stiffness, reduced_load, interior) =
        apply_homogeneous_dirichlet(&stiffness, &load);

    let (sub, main, super_diagonal) = reduced_stiffness.to_tridiagonal();
    let interior_values =
        crate::sparse::solve_tridiagonal(&sub, &main, &super_diagonal, &reduced_load);

    let u_h = scatter_interior_solution(&interior_values, nodes.len(), &interior);
    (nodes, u_h)
}

/// Same problem, solved iteratively. Returns the solution and the iteration
/// count so the optional CG study can report it.
pub fn solve_poisson_cg(
    n_elements: usize,
    relative_tolerance: f64,
) -> (Vec<f64>, Vec<f64>, usize) {
    let nodes = uniform_mesh(n_elements);
    let (stiffness, load) = assemble_system(&nodes, source_term);
    let (reduced_stiffness, reduced_load, interior) =
        apply_homogeneous_dirichlet(&stiffness, &load);

    let max_iterations = 20 * reduced_load.len() + 100;
    let result = conjugate_gradient(
        &reduced_stiffness,
        &reduced_load,
        relative_tolerance,
        max_iterations,
    );
    assert!(result.converged, "CG failed to converge for n = {n_elements}");

    let u_h = scatter_interior_solution(&result.solution, nodes.len(), &interior);
    (nodes, u_h, result.iterations)
}

// ---------------------------------------------------------------------------
// Error measurement
// ---------------------------------------------------------------------------
/// L2 norm of the error: sqrt(integral of (u - u_h)^2 over (0, 1)).
///
/// Integrated element by element with a 5-point Gauss rule. Inside an element
/// u_h is the straight line through its two nodal values, so it can be
/// evaluated anywhere by linear interpolation.
pub fn l2_error(nodes: &[f64], u_h: &[f64]) -> f64 {
    let mut total = 0.0;

    for element in 0..nodes.len() - 1 {
        let x_left = nodes[element];
        let x_right = nodes[element + 1];
        let h = x_right - x_left;
        let (points, weights) = map_rule(&GAUSS_5_POINTS, &GAUSS_5_WEIGHTS, x_left, x_right);

        for (point, weight) in points.iter().zip(weights.iter()) {
            let shape_left = (x_right - point) / h;
            let shape_right = (point - x_left) / h;
            let u_h_value = u_h[element] * shape_left + u_h[element + 1] * shape_right;

            let difference = exact_solution(*point) - u_h_value;
            total += weight * difference * difference;
        }
    }

    total.sqrt()
}

/// H1 seminorm ("energy") error: sqrt(integral of (u' - u_h')^2 over (0, 1)).
///
/// On each element u_h is a straight line, so u_h' is the constant slope
/// (u_right - u_left) / h. This is the norm the method actually minimises, so
/// it is the honest one to report next to L2.
pub fn h1_seminorm_error(nodes: &[f64], u_h: &[f64]) -> f64 {
    let mut total = 0.0;

    for element in 0..nodes.len() - 1 {
        let x_left = nodes[element];
        let x_right = nodes[element + 1];
        let h = x_right - x_left;
        let slope = (u_h[element + 1] - u_h[element]) / h;
        let (points, weights) = map_rule(&GAUSS_5_POINTS, &GAUSS_5_WEIGHTS, x_left, x_right);

        for (point, weight) in points.iter().zip(weights.iter()) {
            let difference = exact_derivative(*point) - slope;
            total += weight * difference * difference;
        }
    }

    total.sqrt()
}

/// Largest |u(x_i) - u_h(x_i)| over the nodes. Reported as a curiosity: in 1D
/// this method is nodally exact, so this number is tiny and is limited only by
/// the load-vector quadrature, not by the mesh.
pub fn max_nodal_error(nodes: &[f64], u_h: &[f64]) -> f64 {
    nodes
        .iter()
        .zip(u_h.iter())
        .map(|(x, value)| (exact_solution(*x) - value).abs())
        .fold(0.0, f64::max)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn element_stiffness_rows_sum_to_zero() {
        // A constant function has zero derivative, so the stiffness matrix must
        // annihilate the vector [1, 1]. This catches sign and scaling slips.
        let matrix = element_stiffness(0.25);
        for row in matrix.iter() {
            assert!((row[0] + row[1]).abs() < 1e-14);
        }
    }

    #[test]
    fn element_load_is_exact_for_a_constant_source() {
        // For f = 1 the exact local load vector is [h/2, h/2].
        fn one(_x: f64) -> f64 {
            1.0
        }
        let load = element_load(0.2, 0.5, one);
        assert!((load[0] - 0.15).abs() < 1e-14);
        assert!((load[1] - 0.15).abs() < 1e-14);
    }

    #[test]
    fn assembled_matrix_has_the_expected_stencil() {
        // For a uniform mesh the interior rows should read [-1/h, 2/h, -1/h].
        let nodes = uniform_mesh(4);
        let (stiffness, _) = assemble_system(&nodes, source_term);
        let h = 0.25;

        assert!((stiffness.get(2, 1) + 1.0 / h).abs() < 1e-12);
        assert!((stiffness.get(2, 2) - 2.0 / h).abs() < 1e-12);
        assert!((stiffness.get(2, 3) + 1.0 / h).abs() < 1e-12);
    }

    #[test]
    fn boundary_values_are_exactly_zero() {
        let (_, u_h) = solve_poisson_direct(16);
        assert_eq!(u_h[0], 0.0);
        assert_eq!(*u_h.last().unwrap(), 0.0);
    }

    #[test]
    fn l2_error_halves_four_times_when_the_mesh_is_refined() {
        // The expected rate is 2, so the error should drop by about 4x.
        let (nodes_coarse, u_coarse) = solve_poisson_direct(20);
        let (nodes_fine, u_fine) = solve_poisson_direct(40);

        let ratio = l2_error(&nodes_coarse, &u_coarse) / l2_error(&nodes_fine, &u_fine);
        assert!((ratio - 4.0).abs() < 0.1, "observed ratio {ratio}");
    }

    #[test]
    fn direct_and_iterative_solvers_agree() {
        let (_, u_direct) = solve_poisson_direct(50);
        let (_, u_iterative, _) = solve_poisson_cg(50, 1e-12);

        for (a, b) in u_direct.iter().zip(u_iterative.iter()) {
            assert!((a - b).abs() < 1e-10);
        }
    }
}
