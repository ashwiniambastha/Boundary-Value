"""
Tests for the Python finite element code.

These mirror the Rust tests in rust/src/*.rs so that both implementations are
held to the same standard.

Run with pytest:      pytest test_fem1d.py
or without pytest:    python test_fem1d.py
"""

import numpy as np

from fem1d import (
    assemble_system,
    element_load,
    element_stiffness,
    exact_derivative,
    exact_solution,
    gauss_points_on_element,
    h1_seminorm_error,
    l2_error,
    max_nodal_error,
    solve_poisson,
    source_term,
    uniform_mesh,
)


def test_mesh_covers_the_unit_interval():
    nodes, elements = uniform_mesh(5)

    assert len(nodes) == 6
    assert nodes[0] == 0.0
    assert nodes[-1] == 1.0
    assert elements.shape == (5, 2)
    # Every element must join consecutive nodes.
    assert np.all(elements[:, 1] - elements[:, 0] == 1)


def test_quadrature_integrates_a_cubic_exactly():
    """2-point Gauss is exact up to degree 3; this checks the mapping too."""
    points, weights = gauss_points_on_element(1.0, 3.0, 2)
    computed = np.sum(weights * points**3)
    exact = (3.0**4 - 1.0**4) / 4.0

    assert abs(computed - exact) < 1e-12


def test_element_stiffness_rows_sum_to_zero():
    """A constant function has zero derivative, so [1, 1] must be annihilated.

    This is a cheap but sharp check: it catches sign errors and a wrong 1/h.
    """
    matrix = element_stiffness(0.25)

    assert np.allclose(matrix.sum(axis=1), 0.0)
    assert np.allclose(matrix, matrix.T)


def test_element_load_is_exact_for_a_constant_source():
    """For f = 1 the exact local load vector is [h/2, h/2]."""
    load = element_load(0.2, 0.5, lambda x: np.ones_like(x))

    assert np.allclose(load, [0.15, 0.15])


def test_assembled_matrix_has_the_expected_stencil():
    """Interior rows of a uniform mesh should read [-1/h, 2/h, -1/h]."""
    nodes, elements = uniform_mesh(4)
    stiffness, _ = assemble_system(nodes, elements, source_term)
    h = 0.25

    row = stiffness.toarray()[2]
    assert abs(row[1] + 1.0 / h) < 1e-12
    assert abs(row[2] - 2.0 / h) < 1e-12
    assert abs(row[3] + 1.0 / h) < 1e-12
    # Nothing outside the three-point stencil.
    assert abs(row[0]) < 1e-15 and abs(row[4]) < 1e-15


def test_assembled_matrix_is_singular_before_boundary_conditions():
    """Without boundary conditions the constant vector is in the null space.

    Physically: -u'' = f alone does not determine u, since adding a constant
    changes nothing. This is why the boundary conditions are not optional.
    """
    nodes, elements = uniform_mesh(8)
    stiffness, _ = assemble_system(nodes, elements, source_term)
    ones = np.ones(len(nodes))

    assert np.max(np.abs(stiffness @ ones)) < 1e-12


def test_boundary_values_are_exactly_zero():
    _, u_h = solve_poisson(16, source_term)

    assert u_h[0] == 0.0
    assert u_h[-1] == 0.0


def test_l2_error_drops_by_four_when_the_mesh_is_halved():
    """Expected rate is 2, so refining by 2 should divide the error by ~4."""
    nodes_coarse, u_coarse = solve_poisson(20, source_term)
    nodes_fine, u_fine = solve_poisson(40, source_term)

    ratio = (
        l2_error(nodes_coarse, u_coarse, exact_solution)
        / l2_error(nodes_fine, u_fine, exact_solution)
    )
    assert abs(ratio - 4.0) < 0.1


def test_h1_error_drops_by_two_when_the_mesh_is_halved():
    """Expected rate is 1, so refining by 2 should halve the error."""
    nodes_coarse, u_coarse = solve_poisson(20, source_term)
    nodes_fine, u_fine = solve_poisson(40, source_term)

    ratio = (
        h1_seminorm_error(nodes_coarse, u_coarse, exact_derivative)
        / h1_seminorm_error(nodes_fine, u_fine, exact_derivative)
    )
    assert abs(ratio - 2.0) < 0.05


def test_solution_is_nodally_exact_up_to_quadrature():
    """The 1D superconvergence property, pinned down as a test.

    With a more accurate load-vector quadrature the nodal error should fall to
    roughly machine precision, because in 1D the P1 solution is exact AT THE
    NODES. If this ever stops holding, the assembly or the boundary handling has
    a bug.
    """
    nodes, u_h = solve_poisson(20, source_term, n_quadrature_points=5)

    assert max_nodal_error(nodes, u_h, exact_solution) < 1e-13


def _run_all_tests():
    """Tiny runner so the file works without pytest installed."""
    tests = [
        (name, function)
        for name, function in sorted(globals().items())
        if name.startswith("test_") and callable(function)
    ]

    failures = 0
    for name, function in tests:
        try:
            function()
            print(f"PASS  {name}")
        except AssertionError as error:
            failures += 1
            print(f"FAIL  {name}: {error}")

    print(f"\n{len(tests) - failures} passed, {failures} failed")
    return failures


if __name__ == "__main__":
    raise SystemExit(1 if _run_all_tests() else 0)
