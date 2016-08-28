from pic.poisson_sor_solver import solve_poisson
import pytest
import numpy as np


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
    gradients = np.gradient(field, *dr)
    if len(dr) == 1:
        dr = dr*field.ndim
    gradgrad = [np.gradient(grad, dr[i])[i] for i, grad in
                enumerate(gradients)]
    return sum(gradgrad)



def test_possion_solver(poisson_solution):
    xx, yy, rho, phi_s, h = poisson_solution
    phi = rho.copy()
    iterations = solve_poisson(rho, phi, convergence_ratio=1e-10)
    print iterations, phi_s/phi
    np.testing.assert_allclose(phi, phi_s)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from mpl_toolkits import mplot3d
    x, y, r, p_s, h = poisson_solution()
    p = np.zeros_like(r)
    it = solve_poisson(r, p, convergence_ratio=1e-6)
    ax = plt.subplot(111, projection='3d')
    ax.plot_surface(x, y, p, alpha=0.5)
    ax.plot_surface(x, y, p_s, color='y', alpha=0.5)
    plt.show()
