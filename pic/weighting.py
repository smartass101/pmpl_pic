import numba


@numba.jit(nopython=True)
def find_particle_in_grid(particle_pos, h, S_h):
    """Find position (index and offset from lower left node) of particle in grid"""
    j, x = divmod(particle_pos[0], h)
    k, y = divmod(particle_pos[1], h)
    return j, k, x, y


@numba.jit(nopython=True)
def cic_charge_weighting(particle_pos, particle_charge, active_particles, rho, h):
    """Cloud-in-cell charge weighting of particles to rho grid"""
    S_h = h**2                             # cell surface
    rho.fill(0.0)             # TODO optimization: perform in some for loop
    for i in range(active_particles):
        j, k, x, y = find_particle_in_grid(particle_pos[i], h)
        rho[j,k] += partcle_charge * (h-x)*(h-y)/S_h
        rho[j+1,k] += partcle_charge * x*(h-y)/S_h
        rho[j+1,k+1] += partcle_charge * x*y/S_h
        rho[j,k+1] += partcle_charge * (h-x)*y/S_h


@numba.jit(nopython=True)
def cic_field_weighting(particle_pos, particle_field, active_particles, field, h):
    """Cloud-in-cell field weighting of field grid to particles"""
    S_h = h**2                             # cell surface
    for i in range(active_particles):
        j, k, x, y = find_particle_in_grid(particle_pos[i], h)
        particle_field[i] = 0.0
        particle_field[i] += field[j,k] * (h-x)*(h-y)/S_h
        particle_field[i] += field[j+1,k] * x*(h-y)/S_h
        particle_field[i] += field[j+1,k+1] * x*y/S_h
        particle_field[i] += field[j,k+1] * (h-x)*y/S_h

