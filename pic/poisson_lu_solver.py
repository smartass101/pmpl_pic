import scipy.sparse as sps
import scipy.sparse.linalg as splalg


def create_minus_laplace_matrix(n):
    """Create sparse matrix representing -\\nabla^2 in 2D

    Assumes 2D (n,n) grid.
    Derivatives are replaced by discretizations.

    Useful for solving discretized Poisson equation
    -(phi_ij-1 + phi_ij+1 + phi_i-1j + phi_i+1j) + 4*phi_ij = rho*h**2/epsilon_0


    """
    return sps.diags([-1, -1, 4, -1, -1],
                     [-n, -1, 0, 1, n],
                     (n**2, n**2))


def create_poisson_solver(n):
    """Return a Poisson equation solver for (n,n) rho grid

    Assumes phi=0 at boundaries.
    Uses sparse LU factorization.

    """
    minus_laplace_matrix = create_minus_laplace_matrix(n)
    solve_lu = splalg.factorized(minus_laplace_matrix)
    def solver(rho_normed):
        """Solve Poisson equation for phi with normed rho

        rho_normed = rho*h**2/epsilon_0

        """
        rho_vector = rho_normed.reshape((n**2,))
        phi_vector = solve_lu(rho_vector)
        return phi_vector.reshape((n,n))
    return solver
