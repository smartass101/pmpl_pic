import numpy as np
import numba


@numba.jit(nopython=True)
def find_particle_in_grid(particle_pos, h):
    """Find position (index and offset from lower left node) of particle in grid"""
    j, x = divmod(particle_pos[0], h)
    k, y = divmod(particle_pos[1], h)
    return int(j), int(k), x, y


@numba.jit(nopython=True)
def cic_charge_weighting(particle_pos, particle_charge, active_particles, rho, h):
    """Cloud-in-cell charge weighting of particles to rho grid"""
    S_h = h**2                             # cell surface
    n = rho.shape[0]
    for i in range(active_particles):
        j, k, x, y = find_particle_in_grid(particle_pos[i], h)
        rho[j,k] += particle_charge * (h-x)*(h-y)/S_h
        jj, kk = j+1, k+1
        if jj < n:
            rho[jj,k] += particle_charge * x*(h-y)/S_h
            if kk < n:
                rho[jj,kk] += particle_charge * x*y/S_h
        if kk < n:
            rho[j,kk] += particle_charge * (h-x)*y/S_h


@numba.jit(nopython=True)
def cic_field_weighting(particle_pos, particle_field, active_particles, field, h):
    """Cloud-in-cell field weighting of field grid to particles"""
    S_h = h**2                             # cell surface
    n = field.shape[0]
    for i in range(active_particles):
        j, k, x, y = find_particle_in_grid(particle_pos[i], h)
        particle_field[i] = 0.0
        particle_field[i] += field[j,k] * (h-x)*(h-y)/S_h
        jj, kk = j+1, k+1
        if jj < n:
            particle_field[i] += field[jj,k] * x*(h-y)/S_h
            if kk < n:
                particle_field[i] += field[jj,kk] * x*y/S_h
        if kk < n:
            particle_field[i] += field[j,kk] * (h-x)*y/S_h

