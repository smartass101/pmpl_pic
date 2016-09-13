from scipy.stats import norm, uniform
import scipy.constants as spc


def maxwell_velocities(T_e, m, N):
    vel_var = T_e * spc.eV / m # variance of normal dist
    return norm.rvs(scale=vel_var**0.5, size=(N, 3))


def uniform_positions(region_length, N):
    return uniform.rvs(loc=0, scale=region_length, size=(N, 2)) # 2D position


def optimum_dt(h, T_e, m, hops_in_cell=10):
    v = (3*T_e*spc.eV/m)**0.5
    dt = h / v / hops_in_cell
    return dt
