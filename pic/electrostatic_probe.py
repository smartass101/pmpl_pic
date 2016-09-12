import numpy as np
from pic.poisson_sor_solver import solve_poisson


class ProbeSetup(object):

    def __init__(self, grid_size, probe_size, h):
        self.h = h
        grid_length = grid_size * h
        grid_center = grid_length / 2.0
        self.probe_length = probe_size * h
        half_width = self.probe_length / 2.0
        self.S = self.probe_length**2
        self.min = grid_center - half_width
        self.max = grid_center + half_width
        # pre-calculate potential
        self.grid_shape = (grid_size, grid_size)
        self.phi_1 = np.zeros(self.grid_shape)
        # make sure the potential is given within the probe
        i_min = int(np.ceil(self.min / h))
        i_max = int(np.floor(self.max / h))
        # solve the phi potential
        rho = np.zeros(self.grid_shape)
        for i in range(100):
            self.phi_1[i_min:i_max+1,i_min:i_max+1] = 1.0 # set U_probe
            solve_poisson(rho, self.phi_1, np.nan, 0, 1)  # one iteration

    def get_potential(self, U_probe):
        return self.phi_1 * U_probe


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    probe = ProbeSetup(100, 9, 1e-10)
    phi = probe.get_potential(1)
    plt.matshow(phi)
    plt.colorbar()
    plt.show()

