"""Poisson equation solver using the successive over-relaxation (SOR) method

Based on http://www.physics.buffalo.edu/phy410-505/2011/topic3/app1/index.html
"""
import numpy as np
import numba


@numba.jit('int64(float64[:,:], float64[:,:], float64, int64, float64)')
def solve_poisson(rho_normed, phi, convergence_ratio=1e-2, max_iterations=0,
                  relaxation_factor=0):
    """Solve the Poisson equation for phi with given normed rho using SOR

    Assumes phi=0 beyond boundaries.
    Uses equation
    phi_ij = 0.25*(rho_normed + phi_ij-1 + phi_ij+1 + phi_i-1j + phi_i+1j)

    Parameters
    ----------
    rho_normed: (n, n) ndarray
        charge density divided by epsilon_0 and multiplied by
        the square of the lattice step h^2
    phi: (n, n) ndarray
       initial guess of phi, could be phi from last PIC iteration (likely close)
    convergence_ratio: float, optional
       ratio of average absolute change over mean absolute value of phi
       at which to stop iteration, defaults to 1%
    max_iterations: int, optional
       maximum iterations to perform, defaults to n^2
    relaxation_factor: float, optional
       relaxation factor omega in SOR, defaults to optimum 2/(1+pi/n)

    Returns
    -------
    iterations: int
        number of iterations performed
    """
    sum_abs_change = iterations = 0
    reverse = False
    N = phi.shape[0]
    if max_iterations == 0:
        max_iterations = N**2
    if relaxation_factor == 0:
        relaxation_factor = optimum_sor_omega(N)
    while True:
        sum_abs_change = 0.0
        sum_abs_phi = 0.0
        # choose iteration direction
        if reverse:
            start, stop, step = N-1, -1, -1
        else:
            start, stop, step = 0, N, 1
        # perform iteration
        for i in range(start, stop, step):
            for j in range(start, stop, step):
                phi_new = rho_normed[i,j]
                # beyond boundaries phi=0, so only add if within boundary
                if i > 0:
                    phi_new += phi[i-1,j]
                if i < N-1:
                    phi_new += phi[i+1,j]
                if j > 0:
                    phi_new += phi[i,j-1]
                if j < N-1:
                    phi_new += phi[i,j+1]
                phi_new *= 0.25 # 2D diff
                sum_abs_phi += abs(phi_new)
                sum_abs_change += abs(phi_new - phi[i,j])
                phi[i,j] = (1 - relaxation_factor)*phi[i,j] + relaxation_factor*phi_new
        reverse = not reverse # alternate iteration direction to mitigate bias
        iterations += 1
        # compare means (count divisor is the same) and iterations count
        if sum_abs_change / sum_abs_phi < convergence_ratio or iterations >= max_iterations:
            break
    return iterations


@numba.jit('float64(int64)')
def optimum_sor_omega(points_1d):
    """Optimal SOR relaxation factor for square lattice with a points_1d side"""
    return 2.0 / (1 + np.pi / points_1d)


if __name__ == '__main__':
    n = 100
    size = (n, n)
    rho = np.zeros(size)
    center = (slice(49,51),)*2
    rho[center] = 1
    rho[12:16, 32] = -1
    phi = rho.copy()
    iterations = solve_poisson(rho, phi)
    print iterations, optimum_sor_omega(n)
    import matplotlib.pyplot as plt
    plt.matshow(phi, cmap=plt.cm.viridis)
    plt.show()
