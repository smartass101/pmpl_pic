from scipy.stats import norm, uniform
import scipy.constants as spc


def maxwell_velocities(T_e, m, N):
    vel_var = T_e * spc.eV / m # variance of normal dist
    return norm.rvs(scale=vel_var**0.5, size=(N, 3))


def uniform_positions(region_length, N):
    return uniform.rvs(loc=0, scale=region_length, size=(N, 2)) # 2D position
