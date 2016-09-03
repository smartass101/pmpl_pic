import numba


@numba.jit(nopython=True, cache=True)
def cic_charge_weighting(particle_pos, particle_charge, active_particles, rho, h):
    """Cloud-in-cell charge weighting of particles to rho grid"""
    S_h = h**2                             # cell surface
    for i in range(active_particles):
        j, x = divmod(particle_pos[i,0], h)
        k, y = divmod(particle_pos[i,1], h)
        p_charge = particle_charge[i]
        rho[j,k] += p_charge * (h-x)*(h-y)/S_h
        rho[j+1,k] += p_charge * x*(h-y)/S_h
        rho[j+1,k+1] += p_charge * x*y/S_h
        rho[j,k+1] += p_charge * (h-x)*y/S_h


@numba.jit(nopython=True, cache=True)
def cic_field_weighting(particle_pos, particle_field, active_particles, field, h):
    """Cloud-in-cell field weighting of field grid to particles"""
    S_h = h**2                             # cell surface
    for i in range(active_particles):
        j, x = divmod(particle_pos[i,0], h)
        k, y = divmod(particle_pos[i,1], h)
        particle_field[i] += field[j,k] * (h-x)*(h-y)/S_h
        particle_field[i] += field[j+1,k] * x*(h-y)/S_h
        particle_field[i] += field[j+1,k+1] * x*y/S_h
        particle_field[i] += field[j,k+1] * (h-x)*y/S_h

