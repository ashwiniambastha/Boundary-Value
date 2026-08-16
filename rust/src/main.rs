//! Reporting driver: runs the convergence study and the optional conjugate
//! gradient study, prints tables, and writes CSV files next to the Python ones.
//!
//! Run with:  cargo run --release
//!
//! The numbers printed here should match `python/run_convergence.py` and
//! `python/run_cg.py` to many digits. That agreement is the main cross-check on
//! both implementations, and `scripts/compare_python_rust.py` automates it.

use std::f64::consts::PI;
use std::fs;
use std::io::Write;
use std::path::Path;

use fem_poisson_1d::fem;
use fem_poisson_1d::sparse::{conjugate_gradient, norm};

const MESH_SIZES: [usize; 5] = [10, 20, 40, 80, 160];
const CG_MESH_SIZES: [usize; 7] = [10, 20, 40, 80, 160, 320, 640];
const CG_RELATIVE_TOLERANCE: f64 = 1e-10;
const RESULTS_DIR: &str = "../results";

// ---------------------------------------------------------------------------
// Convergence study
// ---------------------------------------------------------------------------
struct ConvergenceRow {
    n_elements: usize,
    h: f64,
    l2: f64,
    h1: f64,
    nodal: f64,
    l2_rate: Option<f64>,
    h1_rate: Option<f64>,
}

/// Estimate p in error ~ C h^p from two successive halvings of h.
fn observed_rate(coarse_error: f64, fine_error: f64) -> f64 {
    (coarse_error / fine_error).log2()
}

fn run_convergence_study() -> Vec<ConvergenceRow> {
    let mut rows: Vec<ConvergenceRow> = Vec::new();

    for &n_elements in MESH_SIZES.iter() {
        let (nodes, u_h) = fem::solve_poisson_direct(n_elements);

        let l2 = fem::l2_error(&nodes, &u_h);
        let h1 = fem::h1_seminorm_error(&nodes, &u_h);
        let nodal = fem::max_nodal_error(&nodes, &u_h);

        let (l2_rate, h1_rate) = match rows.last() {
            None => (None, None),
            Some(previous) => (
                Some(observed_rate(previous.l2, l2)),
                Some(observed_rate(previous.h1, h1)),
            ),
        };

        rows.push(ConvergenceRow {
            n_elements,
            h: 1.0 / n_elements as f64,
            l2,
            h1,
            nodal,
            l2_rate,
            h1_rate,
        });
    }

    rows
}

fn print_convergence_table(rows: &[ConvergenceRow]) {
    println!(
        "{:>5} {:>10} {:>12} {:>6} {:>12} {:>6} {:>12}",
        "n", "h", "L2 error", "rate", "H1 error", "rate", "max nodal"
    );
    println!("{}", "-".repeat(69));

    for row in rows {
        let l2_rate = format_rate(row.l2_rate);
        let h1_rate = format_rate(row.h1_rate);
        println!(
            "{:>5} {:>10.5} {:>12.3e} {:>6} {:>12.3e} {:>6} {:>12.3e}",
            row.n_elements, row.h, row.l2, l2_rate, row.h1, h1_rate, row.nodal
        );
    }
}

fn format_rate(rate: Option<f64>) -> String {
    match rate {
        None => "  -  ".to_string(),
        Some(value) => format!("{value:.2}"),
    }
}

// ---------------------------------------------------------------------------
// Conjugate gradient study
// ---------------------------------------------------------------------------
struct CgRow {
    n_elements: usize,
    iterations: usize,
    condition_number: f64,
    check: f64,
}

/// Closed-form condition number of the interior stiffness matrix.
///
/// Its eigenvalues are (4/h) sin^2(k pi h / 2) for k = 1 .. n-1, so the extremes
/// come from k = 1 and k = n-1. Using the formula avoids a numerical eigenvalue
/// solve on the finest meshes.
fn estimate_condition_number(n_elements: usize) -> f64 {
    let h = 1.0 / n_elements as f64;
    let smallest = (4.0 / h) * ((PI * h / 2.0).sin()).powi(2);
    let largest = (4.0 / h) * (((n_elements - 1) as f64 * PI * h / 2.0).sin()).powi(2);
    largest / smallest
}

/// A tiny deterministic generator, so the "generic right-hand side" experiment
/// is reproducible without pulling in the `rand` crate. This is a standard
/// xorshift64*; statistical quality is irrelevant here, we only need a vector
/// that is not accidentally an eigenvector.
struct Xorshift64Star {
    state: u64,
}

impl Xorshift64Star {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_uniform(&mut self) -> f64 {
        self.state ^= self.state >> 12;
        self.state ^= self.state << 25;
        self.state ^= self.state >> 27;
        let value = self.state.wrapping_mul(0x2545_F491_4F6C_DD1D);
        // Map to (-1, 1).
        2.0 * (value >> 11) as f64 / (1u64 << 53) as f64 - 1.0
    }
}

fn run_cg_study(use_random_rhs: bool) -> Vec<CgRow> {
    let mut rows = Vec::new();

    for &n_elements in CG_MESH_SIZES.iter() {
        let nodes = fem::uniform_mesh(n_elements);
        let (stiffness, load) = fem::assemble_system(&nodes, fem::source_term);
        let (reduced_stiffness, reduced_load, interior) =
            fem::apply_homogeneous_dirichlet(&stiffness, &load);

        let right_hand_side: Vec<f64> = if use_random_rhs {
            let mut generator = Xorshift64Star::new(0x2026_0816);
            (0..reduced_load.len()).map(|_| generator.next_uniform()).collect()
        } else {
            reduced_load.clone()
        };

        let max_iterations = 20 * right_hand_side.len() + 100;
        let result = conjugate_gradient(
            &reduced_stiffness,
            &right_hand_side,
            CG_RELATIVE_TOLERANCE,
            max_iterations,
        );
        assert!(result.converged, "CG failed to converge for n = {n_elements}");

        // With the physical right-hand side we can check the discretisation
        // error; with a random one there is no exact solution, so we check the
        // residual instead.
        let check = if use_random_rhs {
            let product = reduced_stiffness.multiply(&result.solution);
            let residual: Vec<f64> = product
                .iter()
                .zip(right_hand_side.iter())
                .map(|(a, b)| a - b)
                .collect();
            norm(&residual) / norm(&right_hand_side)
        } else {
            let u_h =
                fem::scatter_interior_solution(&result.solution, nodes.len(), &interior);
            fem::l2_error(&nodes, &u_h)
        };

        rows.push(CgRow {
            n_elements,
            iterations: result.iterations,
            condition_number: estimate_condition_number(n_elements),
            check,
        });
    }

    rows
}

fn print_cg_table(rows: &[CgRow], check_label: &str) {
    println!(
        "{:>6} {:>9} {:>8} {:>12} {:>11} {:>14}",
        "n", "CG iters", "growth", "cond(A)", "sqrt(cond)", check_label
    );
    println!("{}", "-".repeat(65));

    let mut previous_iterations: Option<usize> = None;
    for row in rows {
        let growth = match previous_iterations {
            None => "   -  ".to_string(),
            Some(previous) => format!("{:.2}", row.iterations as f64 / previous as f64),
        };
        println!(
            "{:>6} {:>9} {:>8} {:>12.3e} {:>11.1} {:>14.3e}",
            row.n_elements,
            row.iterations,
            growth,
            row.condition_number,
            row.condition_number.sqrt(),
            row.check
        );
        previous_iterations = Some(row.iterations);
    }
}

// ---------------------------------------------------------------------------
// CSV output
// ---------------------------------------------------------------------------
fn write_convergence_csv(rows: &[ConvergenceRow], path: &Path) -> std::io::Result<()> {
    let mut file = fs::File::create(path)?;
    writeln!(file, "n,h,l2,l2_rate,h1,h1_rate,nodal")?;
    for row in rows {
        writeln!(
            file,
            "{},{},{},{},{},{},{}",
            row.n_elements,
            row.h,
            row.l2,
            row.l2_rate.map(|v| v.to_string()).unwrap_or_default(),
            row.h1,
            row.h1_rate.map(|v| v.to_string()).unwrap_or_default(),
            row.nodal
        )?;
    }
    Ok(())
}

fn write_cg_csv(rows: &[CgRow], path: &Path) -> std::io::Result<()> {
    let mut file = fs::File::create(path)?;
    writeln!(file, "n,iterations,condition_number,check")?;
    for row in rows {
        writeln!(
            file,
            "{},{},{},{}",
            row.n_elements, row.iterations, row.condition_number, row.check
        )?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
fn main() -> std::io::Result<()> {
    let results_dir = Path::new(RESULTS_DIR);
    fs::create_dir_all(results_dir)?;

    println!("P1 finite elements for -u'' = pi^2 sin(pi x) on (0, 1), u(0) = u(1) = 0");
    println!("Rust implementation, direct solve via the Thomas algorithm.\n");

    let convergence_rows = run_convergence_study();
    print_convergence_table(&convergence_rows);
    write_convergence_csv(&convergence_rows, &results_dir.join("rust_convergence.csv"))?;

    println!("\n\nOptional extra: conjugate gradient");
    println!("\nTable 1 -- the actual load vector b from f(x) = pi^2 sin(pi x)");
    println!("(b is essentially an eigenvector of A, so CG finishes almost at once)\n");
    let physical_rows = run_cg_study(false);
    print_cg_table(&physical_rows, "L2 error");
    write_cg_csv(&physical_rows, &results_dir.join("rust_cg_iterations.csv"))?;

    println!("\nTable 2 -- a generic (pseudo-random) right-hand side, same matrices");
    println!("(this is the one that shows how CG scales with the mesh)\n");
    let random_rows = run_cg_study(true);
    print_cg_table(&random_rows, "rel. residual");
    write_cg_csv(&random_rows, &results_dir.join("rust_cg_iterations_random_rhs.csv"))?;

    println!("\nrelative tolerance = {CG_RELATIVE_TOLERANCE:e}, no preconditioner");
    println!("Wrote rust_*.csv to {RESULTS_DIR}");

    Ok(())
}
