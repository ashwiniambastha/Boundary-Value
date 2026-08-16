"""
Convergence study for the 1D P1 finite element solver.

Runs the solver on a sequence of uniformly refined meshes, measures the error in
two norms, estimates the observed convergence rate, and writes a table, a CSV
and a plot into ../results/.

Run with:   python run_convergence.py
"""

import csv
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")  
import matplotlib.pyplot as plt

from fem1d import (
    exact_derivative,
    exact_solution,
    h1_seminorm_error,
    l2_error,
    max_nodal_error,
    solve_poisson,
    source_term,
)

MESH_SIZES = [10, 20, 40, 80, 160]
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

COLOUR_L2 = "#3D5AC8"
COLOUR_H1 = "#C2571A"
COLOUR_GUIDE = "#9A9A93"


def observed_rate(coarse_error, fine_error):
    """Estimate the convergence rate p from two successive halvings of h.

    If the error behaves like C * h^p then halving h divides the error by 2^p,
    so p = log2(error_coarse / error_fine).
    """
    return np.log2(coarse_error / fine_error)


def run_study(mesh_sizes=MESH_SIZES):
    """Solve on each mesh and collect the error measurements."""
    records = []
    for n_elements in mesh_sizes:
        nodes, u_h = solve_poisson(n_elements, source_term)
        records.append({
            "n": n_elements,
            "h": 1.0 / n_elements,
            "l2": l2_error(nodes, u_h, exact_solution),
            "h1": h1_seminorm_error(nodes, u_h, exact_derivative),
            "nodal": max_nodal_error(nodes, u_h, exact_solution),
        })

    for index, record in enumerate(records):
        if index == 0:
            record["l2_rate"] = float("nan")
            record["h1_rate"] = float("nan")
        else:
            previous = records[index - 1]
            record["l2_rate"] = observed_rate(previous["l2"], record["l2"])
            record["h1_rate"] = observed_rate(previous["h1"], record["h1"])

    return records


def print_table(records):
    header = (
        f"{'n':>5} {'h':>10} {'L2 error':>12} {'rate':>6} "
        f"{'H1 error':>12} {'rate':>6} {'max nodal':>12}"
    )
    print(header)
    print("-" * len(header))
    for record in records:
        l2_rate = "  -  " if np.isnan(record["l2_rate"]) else f"{record['l2_rate']:.2f}"
        h1_rate = "  -  " if np.isnan(record["h1_rate"]) else f"{record['h1_rate']:.2f}"
        print(
            f"{record['n']:>5} {record['h']:>10.5f} {record['l2']:>12.3e} {l2_rate:>6} "
            f"{record['h1']:>12.3e} {h1_rate:>6} {record['nodal']:>12.3e}"
        )


def write_csv(records, path):
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["n", "h", "l2", "l2_rate", "h1", "h1_rate", "nodal"]
        )
        writer.writeheader()
        writer.writerows(records)


def plot_convergence(records, path):
    """Log-log error against mesh size, with reference slopes for comparison.

    On log-log axes a rate of p shows up as a straight line of slope p, so the
    grey dashed guides make it easy to see by eye whether the measured lines are
    parallel to h^1 and h^2.
    """
    h_values = np.array([record["h"] for record in records])
    l2_values = np.array([record["l2"] for record in records])
    h1_values = np.array([record["h1"] for record in records])

    figure, axes = plt.subplots(figsize=(7.0, 5.0))

    # Reference slopes, anchored to the coarsest measured point of each series.
    axes.loglog(
        h_values, l2_values[0] * (h_values / h_values[0]) ** 2,
        linestyle="--", linewidth=1.2, color=COLOUR_GUIDE, zorder=1,
    )
    axes.loglog(
        h_values, h1_values[0] * (h_values / h_values[0]) ** 1,
        linestyle=":", linewidth=1.2, color=COLOUR_GUIDE, zorder=1,
    )

    axes.loglog(
        h_values, l2_values, marker="o", markersize=7, linewidth=2.0,
        color=COLOUR_L2, label="L2 error", zorder=3,
    )
    axes.loglog(
        h_values, h1_values, marker="s", markersize=7, linewidth=2.0,
        color=COLOUR_H1, label="H1 seminorm error", zorder=3,
    )

    axes.annotate(
        "h^1", xy=(h_values[0], h1_values[0]),
        xytext=(10, 0), textcoords="offset points",
        fontsize=9, color=COLOUR_GUIDE, va="center",
    )
    axes.annotate(
        "h^2", xy=(h_values[0], l2_values[0]),
        xytext=(10, 0), textcoords="offset points",
        fontsize=9, color=COLOUR_GUIDE, va="center",
    )
    axes.set_xlim(h_values[-1] * 0.75, h_values[0] * 1.9)

    axes.set_xlabel("element size  h")
    axes.set_ylabel("error")
    axes.set_title("P1 finite elements: error against mesh size", loc="left")
    axes.grid(True, which="both", linewidth=0.5, alpha=0.3)
    axes.legend(frameon=False)
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)

    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def plot_solution(n_elements, path):
    """Sanity picture: the coarse FE solution drawn over the exact solution."""
    nodes, u_h = solve_poisson(n_elements, source_term)
    fine_x = np.linspace(0.0, 1.0, 400)

    figure, axes = plt.subplots(figsize=(7.0, 4.2))
    axes.plot(
        fine_x, exact_solution(fine_x), linewidth=2.0,
        color=COLOUR_GUIDE, label="exact  sin(pi x)",
    )
    axes.plot(
        nodes, u_h, marker="o", markersize=7, linewidth=2.0,
        color=COLOUR_L2, label=f"P1 solution, n = {n_elements}",
    )

    axes.set_xlabel("x")
    axes.set_ylabel("u")
    axes.set_title("Finite element solution on a coarse mesh", loc="left")
    axes.grid(True, linewidth=0.5, alpha=0.3)
    axes.legend(frameon=False)
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)

    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    records = run_study()
    print_table(records)

    write_csv(records, os.path.join(RESULTS_DIR, "convergence.csv"))
    plot_convergence(records, os.path.join(RESULTS_DIR, "convergence.png"))
    plot_solution(10, os.path.join(RESULTS_DIR, "solution_n10.png"))

    print(f"\nWrote convergence.csv, convergence.png and solution_n10.png to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
