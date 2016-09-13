import numpy as np
from pic.poisson_sor_solver import solve_poisson
import matplotlib.pyplot as plt


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


def v_a_characteristic(U_pr, i_sat, V_fl, T_e):
    """V-A characteristic of Langmuir probe, T_e is in eV"""
    return i_sat * (1 - np.exp((U_pr - V_fl)/T_e))


def plot_fit_v_a_characteristic(U_pr, j_probe, j_probe_std, T_e, B):
    plt.errorbar(U_pr, j_probe, j_probe_std, fmt='ko', label='simulation')
    plt.title('$T_e=%.2g$ eV, $B_z=%.2g$ T' % (T_e, B))
    plt.ylabel('$j_{probe}$ [A/m^2]')
    plt.xlabel('$U_{probe}$ [V]')
    ylims = plt.ylim()
    plt.grid()
    # try fitting analytic curve
    from scipy.optimize import curve_fit
    V_fl = U_pr[np.abs(j_probe).argmin()]
    fit_sl = slice(None, U_pr.shape[0] // 2)
    try:
        p0 = [j_probe.max(), V_fl, T_e]
        p, err = curve_fit(v_a_characteristic, U_pr[fit_sl], j_probe[fit_sl], p0=p0,
                           sigma=j_probe_std[fit_sl], absolute_sigma=True)
    except RuntimeError:
        p = None
    if p is not None:
        plt.plot(U_pr, v_a_characteristic(U_pr, *p), 'r-',
                 label=('$%.3g\\left(1-\\exp\\left(\\frac{U_{probe}  %+.3g}{%.3g} \\right)\\right)$' % tuple(p)))
        plt.ylim(*ylims)        # restore lims to sim data
    plt.legend(loc='lower left')


if __name__ == '__main__':
    probe = ProbeSetup(100, 9, 1e-10)
    phi = probe.get_potential(1)
    plt.matshow(phi)
    plt.colorbar()
    plt.show()

