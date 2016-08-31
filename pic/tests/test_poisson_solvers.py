from __future__ import print_function
from pic.poisson_sor_solver import solve_poisson, optimum_sor_omega
from pic.poisson_lu_solver import create_poisson_solver
import pytest
import numpy as np
from time import time


@pytest.fixture
def poisson_solution():
    n = 100
    x, h = np.linspace(0, np.pi, n, retstep=True)
    xx, yy = np.meshgrid(x, x)
    phi = (np.sin(xx)*np.sin(yy))**4
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


def assert_relative_err_below(actual, desired, rel_err_max=1e-2):
    __tracebackhide__ = True
    rel_err = np.mean(np.abs(actual - desired)) / np.mean(np.abs(desired))
    if rel_err > rel_err_max:
        pytest.fail('relative error {:.2e} above desired {:.2e}'.format(rel_err, rel_err_max))


def test_possion_sor_solver(poisson_solution):
    xx, yy, rho_normed, phi_s, h = poisson_solution
    phi = rho_normed.copy()
    n = phi.shape[0]
    omega = optimum_sor_omega(n)
    start_t = time()
    iterations = solve_poisson(rho_normed, phi, 1e-10, n**2, omega)
    stop_t = time()
    print("solved in", round(1e3*(stop_t-start_t)), "ms in", iterations, "iterations")
    assert_relative_err_below(phi, phi_s, 1e-2)


def test_poisson_lu_solver(poisson_solution):
    xx, yy, rho_normed, phi_s, h = poisson_solution
    n = rho_normed.shape[0]
    solver = create_poisson_solver(n)
    start_t = time()
    phi = solver(rho_normed)
    stop_t = time()
    print("solved in", round(1e3*(stop_t-start_t)), "ms")
    assert_relative_err_below(phi, phi_s, 1e-2)




if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from mpl_toolkits import mplot3d
    x, y, rho_normed, phi_s, h = poisson_solution()
    phi_sor = np.zeros_like(rho_normed)
    omega = optimum_sor_omega(x.shape[0])
    it = solve_poisson(rho_normed, phi_sor, 1e-10, x.shape[0]**2, omega)
    phi_lu = create_poisson_solver(rho_normed.shape[0])(rho_normed)
    ax = plt.subplot(111, projection='3d')
    ax.plot_surface(x, y, phi_sor, alpha=0.5)
    ax.plot_surface(x, y, phi_s, color='y', alpha=0.5)
    ax.plot_surface(x, y, phi_lu, color='r', alpha=0.5)
    plt.show()
