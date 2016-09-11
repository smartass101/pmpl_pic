import numba

# TODO: magnetic field moving
# TODO OPTIMIZE: merge CIC filed weighting with field usage
@numba.jit(nopython=True)
def electrostatic_mover(particle_pos, particle_vel, particle_charge, particle_mass, particle_E, active_particles, dt):
    for i in range(active_particles):
        # velocity is 3D, E and x is 2D
        particle_vel[i][:-1] += particle_charge / particle_mass * particle_E[i][:] * dt
        particle_pos[i][:] += particle_vel[i][:-1] * dt


@numba.jit(nopython=True)
def kinematic_mover(particle_pos, particle_vel, active_particles, dt):
    for i in range(active_particles):
        # velocity is 3D, E and x is 2D
        particle_pos[i][:] += particle_vel[i][:-1] * dt
