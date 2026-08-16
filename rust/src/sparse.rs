//! Minimal sparse linear algebra: triplet assembly, CSR storage, and two
//! solvers.
//!
//! This is deliberately hand-rolled rather than pulled from a crate, for the
//! same reason the assembly loop is hand-rolled: the point of the exercise is
//! the algorithm, not the plumbing.

/// A single (row, column, value) contribution produced during assembly.
///
/// Several triplets may land on the same (row, column); the conversion to CSR
/// sums them, and that summation is exactly what "assembly" means.
#[derive(Clone, Copy, Debug)]
pub struct Triplet {
    pub row: usize,
    pub column: usize,
    pub value: f64,
}

/// Compressed sparse row matrix.
///
/// `row_offsets` has `n_rows + 1` entries: the non-zeros of row `i` occupy the
/// slice `row_offsets[i] .. row_offsets[i + 1]` of `column_indices` and
/// `values`. Column indices within a row are kept sorted, which makes lookups
/// and the tridiagonal extraction below straightforward.
#[derive(Clone, Debug)]
pub struct CsrMatrix {
    n_rows: usize,
    n_columns: usize,
    row_offsets: Vec<usize>,
    column_indices: Vec<usize>,
    values: Vec<f64>,
}

impl CsrMatrix {
    /// Build a CSR matrix from triplets, summing duplicate positions.
    pub fn from_triplets(triplets: &[Triplet], n_rows: usize, n_columns: usize) -> Self {
        let mut sorted: Vec<Triplet> = triplets.to_vec();
        sorted.sort_by_key(|triplet| (triplet.row, triplet.column));

        let mut row_offsets = vec![0usize; n_rows + 1];
        let mut column_indices: Vec<usize> = Vec::new();
        let mut values: Vec<f64> = Vec::new();

        let mut current_row = 0usize;
        for triplet in sorted {
            // Close off any rows we have just moved past (including empty ones).
            while current_row < triplet.row {
                current_row += 1;
                row_offsets[current_row] = values.len();
            }

            // Same position as the previous entry? Then accumulate into it.
            let is_duplicate = !values.is_empty()
                && *column_indices.last().unwrap() == triplet.column
                && row_offsets[current_row] < values.len();

            if is_duplicate {
                *values.last_mut().unwrap() += triplet.value;
            } else {
                column_indices.push(triplet.column);
                values.push(triplet.value);
            }
        }

        while current_row < n_rows {
            current_row += 1;
            row_offsets[current_row] = values.len();
        }

        Self { n_rows, n_columns, row_offsets, column_indices, values }
    }

    pub fn n_rows(&self) -> usize {
        self.n_rows
    }

    /// Look up a single entry. Linear in the row length, which is fine here
    /// because every row has at most three non-zeros.
    pub fn get(&self, row: usize, column: usize) -> f64 {
        let start = self.row_offsets[row];
        let end = self.row_offsets[row + 1];
        for index in start..end {
            if self.column_indices[index] == column {
                return self.values[index];
            }
        }
        0.0
    }

    /// Matrix-vector product. This is the only operation conjugate gradient
    /// needs from the matrix, which is why CG suits sparse problems so well.
    pub fn multiply(&self, vector: &[f64]) -> Vec<f64> {
        assert_eq!(vector.len(), self.n_columns, "dimension mismatch in multiply");

        let mut result = vec![0.0; self.n_rows];
        for row in 0..self.n_rows {
            let mut sum = 0.0;
            for index in self.row_offsets[row]..self.row_offsets[row + 1] {
                sum += self.values[index] * vector[self.column_indices[index]];
            }
            result[row] = sum;
        }
        result
    }

    /// Keep only the rows and columns in `keep`, renumbered to 0, 1, 2, ...
    ///
    /// This is how the Dirichlet boundary conditions are imposed: the boundary
    /// nodes are known, so their rows and columns are simply dropped.
    pub fn submatrix(&self, keep: &[usize]) -> Self {
        // Map old index -> new index, or usize::MAX for "removed".
        let mut new_index = vec![usize::MAX; self.n_columns];
        for (position, &old) in keep.iter().enumerate() {
            new_index[old] = position;
        }

        let mut triplets = Vec::new();
        for (new_row, &old_row) in keep.iter().enumerate() {
            for index in self.row_offsets[old_row]..self.row_offsets[old_row + 1] {
                let old_column = self.column_indices[index];
                if new_index[old_column] != usize::MAX {
                    triplets.push(Triplet {
                        row: new_row,
                        column: new_index[old_column],
                        value: self.values[index],
                    });
                }
            }
        }

        Self::from_triplets(&triplets, keep.len(), keep.len())
    }

    /// Pull out the three diagonals, checking there is nothing outside them.
    ///
    /// Returns (sub_diagonal, main_diagonal, super_diagonal) where
    /// `sub_diagonal[i]` is the entry at (i, i-1) for i >= 1 and
    /// `super_diagonal[i]` is the entry at (i, i+1).
    pub fn to_tridiagonal(&self) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let n = self.n_rows;
        let mut sub = vec![0.0; n];
        let mut main = vec![0.0; n];
        let mut super_diagonal = vec![0.0; n];

        for row in 0..n {
            for index in self.row_offsets[row]..self.row_offsets[row + 1] {
                let column = self.column_indices[index];
                let value = self.values[index];
                let offset = column as isize - row as isize;
                match offset {
                    -1 => sub[row] = value,
                    0 => main[row] = value,
                    1 => super_diagonal[row] = value,
                    _ => panic!(
                        "matrix is not tridiagonal: entry at ({row}, {column})"
                    ),
                }
            }
        }

        (sub, main, super_diagonal)
    }
}

/// Solve a tridiagonal system with the Thomas algorithm.
///
/// This is just Gaussian elimination with the zeros skipped: because each row
/// touches at most its two neighbours, elimination never creates fill-in, so
/// the whole solve is O(n) instead of O(n^3). It is the natural "direct solver"
/// for a 1D P1 stiffness matrix, and it is what stands in for scipy's `spsolve`
/// on the Rust side.
///
/// No pivoting is used. That is safe here rather than lazy: the matrix is
/// symmetric positive definite and diagonally dominant, so the pivots stay well
/// away from zero.
pub fn solve_tridiagonal(
    sub_diagonal: &[f64],
    main_diagonal: &[f64],
    super_diagonal: &[f64],
    right_hand_side: &[f64],
) -> Vec<f64> {
    let n = main_diagonal.len();
    assert_eq!(right_hand_side.len(), n, "dimension mismatch in tridiagonal solve");

    // Forward sweep: eliminate the sub-diagonal, carrying modified copies of
    // the super-diagonal and the right-hand side.
    let mut modified_super = vec![0.0; n];
    let mut modified_rhs = vec![0.0; n];

    modified_super[0] = super_diagonal[0] / main_diagonal[0];
    modified_rhs[0] = right_hand_side[0] / main_diagonal[0];

    for i in 1..n {
        let denominator = main_diagonal[i] - sub_diagonal[i] * modified_super[i - 1];
        modified_super[i] = super_diagonal[i] / denominator;
        modified_rhs[i] =
            (right_hand_side[i] - sub_diagonal[i] * modified_rhs[i - 1]) / denominator;
    }

    // Back substitution.
    let mut solution = vec![0.0; n];
    solution[n - 1] = modified_rhs[n - 1];
    for i in (0..n - 1).rev() {
        solution[i] = modified_rhs[i] - modified_super[i] * solution[i + 1];
    }

    solution
}

/// Result of an iterative solve.
pub struct CgResult {
    pub solution: Vec<f64>,
    pub iterations: usize,
    pub converged: bool,
}

/// Unpreconditioned conjugate gradient.
///
/// Valid because the matrix is symmetric positive definite once the Dirichlet
/// rows and columns have been removed. The stopping test is on the relative
/// residual so that it means the same thing on every mesh.
pub fn conjugate_gradient(
    matrix: &CsrMatrix,
    right_hand_side: &[f64],
    relative_tolerance: f64,
    max_iterations: usize,
) -> CgResult {
    let n = right_hand_side.len();
    let mut solution = vec![0.0; n];

    // Starting from x = 0 means the initial residual is just b.
    let mut residual = right_hand_side.to_vec();
    let mut direction = residual.clone();
    let mut residual_dot = dot(&residual, &residual);

    let target = relative_tolerance * norm(right_hand_side);
    if norm(&residual) <= target {
        return CgResult { solution, iterations: 0, converged: true };
    }

    for iteration in 1..=max_iterations {
        let matrix_times_direction = matrix.multiply(&direction);
        let step_length = residual_dot / dot(&direction, &matrix_times_direction);

        for i in 0..n {
            solution[i] += step_length * direction[i];
            residual[i] -= step_length * matrix_times_direction[i];
        }

        let new_residual_dot = dot(&residual, &residual);
        if new_residual_dot.sqrt() <= target {
            return CgResult { solution, iterations: iteration, converged: true };
        }

        // Fletcher-Reeves style update: the new search direction is the new
        // residual corrected so that it stays A-conjugate to the previous one.
        let beta = new_residual_dot / residual_dot;
        for i in 0..n {
            direction[i] = residual[i] + beta * direction[i];
        }
        residual_dot = new_residual_dot;
    }

    CgResult { solution, iterations: max_iterations, converged: false }
}

pub fn dot(left: &[f64], right: &[f64]) -> f64 {
    left.iter().zip(right).map(|(a, b)| a * b).sum()
}

pub fn norm(vector: &[f64]) -> f64 {
    dot(vector, vector).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn triplets_with_the_same_position_are_summed() {
        let triplets = vec![
            Triplet { row: 0, column: 0, value: 1.0 },
            Triplet { row: 0, column: 0, value: 2.0 },
            Triplet { row: 1, column: 1, value: 5.0 },
        ];
        let matrix = CsrMatrix::from_triplets(&triplets, 2, 2);

        assert_eq!(matrix.get(0, 0), 3.0);
        assert_eq!(matrix.get(1, 1), 5.0);
        assert_eq!(matrix.get(0, 1), 0.0);
    }

    #[test]
    fn thomas_algorithm_solves_a_small_system() {
        // [[2, -1,  0],      [1]
        //  [-1, 2, -1],  x = [0]
        //  [0, -1,  2]]      [1]
        // has the exact solution [1, 1, 1].
        let sub = vec![0.0, -1.0, -1.0];
        let main = vec![2.0, 2.0, 2.0];
        let super_diagonal = vec![-1.0, -1.0, 0.0];
        let rhs = vec![1.0, 0.0, 1.0];

        let solution = solve_tridiagonal(&sub, &main, &super_diagonal, &rhs);
        for value in solution {
            assert!((value - 1.0).abs() < 1e-12);
        }
    }

    #[test]
    fn conjugate_gradient_matches_the_direct_solver() {
        let triplets = vec![
            Triplet { row: 0, column: 0, value: 2.0 },
            Triplet { row: 0, column: 1, value: -1.0 },
            Triplet { row: 1, column: 0, value: -1.0 },
            Triplet { row: 1, column: 1, value: 2.0 },
            Triplet { row: 1, column: 2, value: -1.0 },
            Triplet { row: 2, column: 1, value: -1.0 },
            Triplet { row: 2, column: 2, value: 2.0 },
        ];
        let matrix = CsrMatrix::from_triplets(&triplets, 3, 3);
        let rhs = vec![1.0, 0.0, 1.0];

        let result = conjugate_gradient(&matrix, &rhs, 1e-12, 100);
        assert!(result.converged);
        for value in result.solution {
            assert!((value - 1.0).abs() < 1e-10);
        }
    }
}
