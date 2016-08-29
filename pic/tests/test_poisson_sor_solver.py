from __future__ import print_function
from pic.poisson_sor_solver import solve_poisson, optimum_sor_omega
import pytest
import numpy as np
from time import time


@pytest.fixture
def poisson_solution():
    n = 100
    x, h = np.linspace(0, np.pi, n, retstep=True)
    xx, yy = np.meshgrid(x, x)
    phi = np.sin(xx)*np.sin(yy)
    phi **= 2                   # make smoother boundaries
    # rho = phi.copy() * 2
    rho = -laplace_numerical(phi, h)
    rho *= h**2                 # normed
    return xx, yy, rho, phi, h


def laplace_numerical(field, *dr):
    """Numerical estimation of the application laplace operator on the field

    Uses :func:`numpy.gradient`
    """
    gradients = np.gradient(field, *dr)
    if len(dr) == 1:
        dr = dr*field.ndim
    gradgrad = [np.gradient(grad, dr[i])[i] for i, grad in
                enumerate(gradients)]
    return sum(gradgrad)



def test_possion_solver(poisson_solution):
    xx, yy, rho, phi_s, h = poisson_solution
    phi = rho.copy()
    start_t = time()
    n = phi.shape[0]
    omega = optimum_sor_omega(n)
    iterations = solve_poisson(rho, phi, 1e-10, n**2, omega)
    stop_t = time()
    print("solved in", round(1e3*(stop_t-start_t)), "us in", iterations, "iterations")
    np.testing.assert_allclose(phi, phi_s, atol=1e-3)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from mpl_toolkits import mplot3d
    x, y, r, p_s, h = poisson_solution()
    p = np.zeros_like(r)
    omega = optimum_sor_omega(x.shape[0])
    it = solve_poisson(r, p, 1e-10, x.shape[0]**2, omega)
    ax = plt.subplot(111, projection='3d')
    ax.plot_surface(x, y, p, alpha=0.5)
    ax.plot_surface(x, y, p_s, color='y', alpha=0.5)
    plt.show()
