"""
Optional extra: solve the same systems with conjugate gradient instead of a
direct factorisation

Run with:   python run_cg.py

"""

import csv
import os

import numpy as np
from scipy.sparse.linalg import cg

from fem1d import (
    apply_homogeneous_dirichlet,
    assemble_system,
    exact_solution,
    l2_error,
    scatter_interior_solution,
    source_term,
    uniform_mesh,
)

MESH_SIZES = [10, 20, 40, 80, 160, 320, 640]
RELATIVE_TOLERANCE = 1e-10
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def count_cg_iterations(matrix, right_hand_side, rtol=RELATIVE_TOLERANCE):
    """Run CG and count iterations using the callback hook.

    scipy calls the callback once per iteration, so a counter in a closure is
    the simplest honest way to measure the count.
    """
    iterations = {"count": 0}

    def callback(_current_solution):
        iterations["count"] += 1

    solution, info = cg(matrix, right_hand_side, rtol=rtol, callback=callback)
    if info != 0:
        raise RuntimeError(f"CG did not converge cleanly (info = {info}).")

    return solution, iterations["count"]


def estimate_condition_number(n_elements):
    """Closed-form condition number of the interior stiffness matrix.

    Eigenvalues are (4/h) * sin^2(k*pi*h/2) for k = 1 .. n-1, so the extremes
    come from k = 1 and k = n-1. Using the formula rather than a numerical
    eigenvalue solve keeps this cheap even on the finest mesh.
    """
    h = 1.0 / n_elements
    wavenumbers = np.array([1, n_elements - 1])
    eigenvalues = (4.0 / h) * np.sin(wavenumbers * np.pi * h / 2.0) ** 2
    return eigenvalues.max() / eigenvalues.min()


def build_reduced_system(n_elements):
    """Assemble and apply boundary conditions, returning the SPD system."""
    nodes, elements = uniform_mesh(n_elements)
    stiffness, load = assemble_system(nodes, elements, source_term)
    reduced_stiffness, reduced_load, interior = apply_homogeneous_dirichlet(
        stiffness, load
    )
    return nodes, reduced_stiffness, reduced_load, interior


def run_study(mesh_sizes=MESH_SIZES, use_random_rhs=False, seed=0):
    """Measure CG iteration counts across a sequence of meshes.

    With use_random_rhs the matrix is unchanged but the right-hand side is a
    fixed-seed random vector, which removes the accidental eigenvector structure
    of the real load vector (see the module docstring).
    """
    records = []
    for n_elements in mesh_sizes:
        nodes, matrix, load, interior = build_reduced_system(n_elements)

        if use_random_rhs:
            generator = np.random.default_rng(seed)
            right_hand_side = generator.standard_normal(matrix.shape[0])
        else:
            right_hand_side = load

        interior_values, iterations = count_cg_iterations(matrix, right_hand_side)

        record = {
            "n": n_elements,
            "iterations": iterations,
            "condition_number": estimate_condition_number(n_elements),
        }

        if use_random_rhs:
            
            residual = np.linalg.norm(matrix @ interior_values - right_hand_side)
            record["check"] = residual / np.linalg.norm(right_hand_side)
        else:
            
            u_h = scatter_interior_solution(interior_values, len(nodes), interior)
            record["check"] = l2_error(nodes, u_h, exact_solution)

        records.append(record)

    for index, record in enumerate(records):
        if index == 0:
            record["iteration_growth"] = float("nan")
        else:
            record["iteration_growth"] = (
                record["iterations"] / records[index - 1]["iterations"]
            )

    return records


def print_table(records, check_label):
    header = (
        f"{'n':>6} {'CG iters':>9} {'growth':>8} "
        f"{'cond(A)':>12} {'sqrt(cond)':>11} {check_label:>14}"
    )
    print(header)
    print("-" * len(header))
    for record in records:
        growth = (
            "   -  "
            if np.isnan(record["iteration_growth"])
            else f"{record['iteration_growth']:.2f}"
        )
        print(
            f"{record['n']:>6} {record['iterations']:>9} {growth:>8} "
            f"{record['condition_number']:>12.3e} "
            f"{np.sqrt(record['condition_number']):>11.1f} {record['check']:>14.3e}"
        )


def write_csv(records, path):
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "n", "iterations", "iteration_growth", "condition_number", "check"
            ],
        )
        writer.writeheader()
        writer.writerows(records)


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("Table 1 -- the actual load vector b from f(x) = pi^2 sin(pi x)")
    print("(b is essentially an eigenvector of A, so CG finishes almost at once)\n")
    physical_records = run_study(use_random_rhs=False)
    print_table(physical_records, "L2 error")
    write_csv(physical_records, os.path.join(RESULTS_DIR, "cg_iterations.csv"))

    print("\n\nTable 2 -- a fixed-seed random right-hand side, same matrices")
    print("(this is the one that shows how CG scales with the mesh)\n")
    random_records = run_study(use_random_rhs=True)
    print_table(random_records, "rel. residual")
    write_csv(random_records, os.path.join(RESULTS_DIR, "cg_iterations_random_rhs.csv"))

    print(f"\nrelative tolerance = {RELATIVE_TOLERANCE:g}, no preconditioner")
    print(f"Wrote both CSVs to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
