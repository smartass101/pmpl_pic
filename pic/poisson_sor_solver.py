"""Poisson equation solver using the successive over-relaxation (SOR) method

Based on http://www.physics.buffalo.edu/phy410-505/2011/topic3/app1/index.html

"""
import numpy as np
import numba


def optimum_sor_omega(points_1d):
    """Optimal SOR relaxation factor for square lattice with a points_1d side"""
    return 2.0 / (1 + np.pi / points_1d)


@numba.jit(nopython=True, cache=True)
def solve_poisson(rho_normed, phi, convergence_ratio, max_iterations,
                  relaxation_factor):
    """Solve the Poisson equation for phi with given normed rho using SOR

    Assumes phi=0 beyond boundaries.
    Uses equation
    phi_ij = 0.25*(rho_normed + phi_ij-1 + phi_ij+1 + phi_i-1j + phi_i+1j)

    Parameters
    ----------
    rho_normed : (n, n) ndarray
        charge density divided by epsilon_0 and multiplied by
        the square of the lattice step h^2
    phi : (n, n) ndarray
        initial guess of phi, could be phi from last PIC iteration (likely close)
    convergence_ratio : float
        ratio of average absolute change over mean absolute value of phi
        at which to stop iteration
    max_iterations : int
        maximum iterations to perform, n^2 is a good guess
    relaxation_factor : float
        relaxation factor omega in SOR

    Returns
    -------
    iterations: int
        number of iterations performed

    """
    # initial values
    iterations = 0
    reverse = False             # iteration direction
    N = phi.shape[0]
    # SOR iteration loop
    while True:
        # initial values
        sum_abs_change = 0.0
        sum_abs_phi = 0.0
        # choose iteration direction
        if reverse:
            start, stop, step = N-1, -1, -1
        else:
            start, stop, step = 0, N, 1
        # perform 1 iteration over field
        for i in range(start, stop, step):
            for j in range(start, stop, step):
                phi_new = rho_normed[i,j] # new phi estimate
                # beyond boundaries phi=0, so only add if within boundary
                if i > 0:
                    phi_new += phi[i-1,j]
                if i < N-1:
                    phi_new += phi[i+1,j]
                if j > 0:
                    phi_new += phi[i,j-1]
                if j < N-1:
                    phi_new += phi[i,j+1]
                phi_new *= 0.25 # 2D diff results in division by 4
                sum_abs_phi += abs(phi_new)
                sum_abs_change += abs(phi_new - phi[i,j])
                # set new SOR estimate
                phi[i,j] = (1 - relaxation_factor)*phi[i,j] + relaxation_factor*phi_new
        reverse = not reverse # alternate iteration direction to mitigate direction bias
        iterations += 1
        # compare means (count divisor is the same) and iterations count
        if sum_abs_change / sum_abs_phi < convergence_ratio or iterations >= max_iterations:
            break
    return iterations
