"""
1D Poisson problem solved with linear (P1) finite elements.

Problem
-------
    -u''(x) = f(x)   on (0, 1)
     u(0) = u(1) = 0

with f(x) = pi^2 * sin(pi*x), whose exact solution is u(x) = sin(pi*x).


"""

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve


GAUSS_RULES = {
    2: (
        np.array([-1.0 / np.sqrt(3.0), 1.0 / np.sqrt(3.0)]),
        np.array([1.0, 1.0]),
    ),
    3: (
        np.array([-np.sqrt(3.0 / 5.0), 0.0, np.sqrt(3.0 / 5.0)]),
        np.array([5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0]),
    ),
    5: (
        np.array([
            -0.906179845938664,
            -0.538469310105683,
            0.0,
            0.538469310105683,
            0.906179845938664,
        ]),
        np.array([
            0.236926885056189,
            0.478628670499366,
            0.568888888888889,
            0.478628670499366,
            0.236926885056189,
        ]),
    ),
}


def gauss_points_on_element(x_left, x_right, n_points):
    """Map a reference Gauss rule onto the physical element [x_left, x_right].

    Returns the physical quadrature points and the *already scaled* weights, so
    that sum(weights * g(points)) approximates the integral of g over the
    element. The scaling factor h/2 is the Jacobian of the affine map from
    [-1, 1] to [x_left, x_right].
    """
    if n_points not in GAUSS_RULES:
        raise ValueError(f"No Gauss rule stored for {n_points} points.")

    ref_points, ref_weights = GAUSS_RULES[n_points]
    half_length = 0.5 * (x_right - x_left)
    midpoint = 0.5 * (x_left + x_right)

    points = midpoint + half_length * ref_points
    weights = half_length * ref_weights
    return points, weights



def uniform_mesh(n_elements):
    """Split (0, 1) into `n_elements` equal elements.

    Returns
    -------
    nodes : (n_elements + 1,) array of node coordinates, 0 = x_0 < ... < x_n = 1
    elements : (n_elements, 2) int array; row e holds the two node indices of
               element e. Storing connectivity explicitly is overkill for a
               uniform 1D mesh, but it keeps the assembly loop written the way
               it would be written in 2D or 3D, which is the point of the
               exercise.
    """
    if n_elements < 1:
        raise ValueError("Need at least one element.")

    nodes = np.linspace(0.0, 1.0, n_elements + 1)
    elements = np.column_stack([np.arange(n_elements), np.arange(1, n_elements + 1)])
    return nodes, elements



def element_stiffness(h):
    """Local stiffness matrix for one element of length h.

        (1/h) * [[ 1, -1],
                 [-1,  1]]

    Derivation: on an element the two local basis functions are the straight
    lines N_0 = (x_right - x)/h and N_1 = (x - x_left)/h. Their derivatives are
    the constants -1/h and +1/h. The entry (i, j) is the integral of
    N_i' * N_j' over the element, which is (derivative_i * derivative_j) * h
    because the integrand is constant. That gives (+/-1/h)(+/-1/h)*h = +/-1/h.
    """
    return (1.0 / h) * np.array([[1.0, -1.0], [-1.0, 1.0]])


def element_load(x_left, x_right, source_function, n_quadrature_points=2):
    """Local load vector for one element: integral of f * N_i over the element."""
    points, weights = gauss_points_on_element(x_left, x_right, n_quadrature_points)
    h = x_right - x_left

    
    shape_left = (x_right - points) / h
    shape_right = (points - x_left) / h

    f_values = source_function(points)
    return np.array([
        np.sum(weights * f_values * shape_left),
        np.sum(weights * f_values * shape_right),
    ])



def assemble_system(nodes, elements, source_function, n_quadrature_points=2):
    """Build the global stiffness matrix A and load vector b. """
    n_nodes = len(nodes)

    rows = []
    cols = []
    values = []
    b = np.zeros(n_nodes)

    for element_nodes in elements:
        left_index, right_index = int(element_nodes[0]), int(element_nodes[1])
        x_left, x_right = nodes[left_index], nodes[right_index]
        h = x_right - x_left

        local_stiffness = element_stiffness(h)
        local_load = element_load(x_left, x_right, source_function, n_quadrature_points)

        global_indices = (left_index, right_index)
        for i in range(2):
            b[global_indices[i]] += local_load[i]
            for j in range(2):
                rows.append(global_indices[i])
                cols.append(global_indices[j])
                values.append(local_stiffness[i, j])
    stiffness = coo_matrix(
        (values, (rows, cols)), shape=(n_nodes, n_nodes)).tocsr()

    return stiffness, b


def apply_homogeneous_dirichlet(stiffness, load):
    """Impose u(0) = u(1) = 0 by removing the two boundary rows and columns."""
    n_nodes = stiffness.shape[0]
    interior = np.arange(1, n_nodes - 1)

    reduced_stiffness = stiffness[interior, :][:, interior]
    reduced_load = load[interior]
    return reduced_stiffness, reduced_load, interior


def scatter_interior_solution(interior_values, n_nodes, interior_indices):
    """Put the interior unknowns back into a full vector with zeros at the ends."""
    full = np.zeros(n_nodes)
    full[interior_indices] = interior_values
    return full



def solve_poisson(n_elements, source_function, n_quadrature_points=2):
    """Assemble, apply the boundary conditions, and solve directly.

    Returns
    -------
    nodes : node coordinates
    u_h : nodal values of the finite element solution
    """
    nodes, elements = uniform_mesh(n_elements)
    stiffness, load = assemble_system(
        nodes, elements, source_function, n_quadrature_points
    )
    reduced_stiffness, reduced_load, interior = apply_homogeneous_dirichlet(
        stiffness, load
    )

    interior_values = spsolve(reduced_stiffness.tocsc(), reduced_load)
    u_h = scatter_interior_solution(interior_values, len(nodes), interior)
    return nodes, u_h


def l2_error(nodes, u_h, exact_solution, n_quadrature_points=5):
    """L2 norm of (u - u_h), i.e. sqrt(integral of (u - u_h)^2 over (0,1)).."""
    total = 0.0
    for element_index in range(len(nodes) - 1):
        x_left, x_right = nodes[element_index], nodes[element_index + 1]
        h = x_right - x_left
        points, weights = gauss_points_on_element(x_left, x_right, n_quadrature_points)

        shape_left = (x_right - points) / h
        shape_right = (points - x_left) / h
        u_h_values = u_h[element_index] * shape_left + u_h[element_index + 1] * shape_right

        difference = exact_solution(points) - u_h_values
        total += np.sum(weights * difference**2)

    return np.sqrt(total)


def h1_seminorm_error(nodes, u_h, exact_derivative, n_quadrature_points=5):
    """H1 seminorm of the error: sqrt(integral of (u' - u_h')^2 over (0,1))."""
    total = 0.0
    for element_index in range(len(nodes) - 1):
        x_left, x_right = nodes[element_index], nodes[element_index + 1]
        h = x_right - x_left
        points, weights = gauss_points_on_element(x_left, x_right, n_quadrature_points)

        slope = (u_h[element_index + 1] - u_h[element_index]) / h
        difference = exact_derivative(points) - slope
        total += np.sum(weights * difference**2)

    return np.sqrt(total)


def max_nodal_error(nodes, u_h, exact_solution):
    """Largest |u(x_i) - u_h(x_i)| over the nodes. Reported as a curiosity."""
    return float(np.max(np.abs(exact_solution(nodes) - u_h)))
def source_term(x):
    """f(x) = pi^2 * sin(pi*x)"""
    return np.pi**2 * np.sin(np.pi * x)


def exact_solution(x):
    """u(x) = sin(pi*x)"""
    return np.sin(np.pi * x)


def exact_derivative(x):
    """u'(x) = pi * cos(pi*x)"""
    return np.pi * np.cos(np.pi * x)
