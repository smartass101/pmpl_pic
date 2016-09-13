import numpy as np
import numba

# TODO OPTIMIZE: merge CIC filed weighting with field usage
@numba.jit(nopython=True)
def electrostatic_mover(particle_pos, particle_vel, particle_charge, particle_mass, particle_E, active_particles, dt):
    for i in range(active_particles):
        # velocity is 3D, E and x is 2D
        particle_vel[i][:-1] += particle_charge / particle_mass * particle_E[i][:] * dt
        particle_pos[i][:] += particle_vel[i][:-1] * dt


@numba.jit(nopython=True)
def cross(a, b):
    """Numba optimized cross product of 2 vectors"""
    res = np.empty(3)
    res[0] = a[1]*b[2] - b[1]*a[2]
    res[1] = a[2]*b[0] - b[2]*a[0]
    res[2] = a[0]*b[1] - b[0]*a[1]
    return res


@numba.jit(nopython=True)
def electrostatic_homog_B_mover(particle_pos, particle_vel, particle_charge,
                                particle_mass, B, particle_E, active_particles, dt):
    q_d_m = particle_charge / particle_mass
    dt_half = dt / 2
    T = q_d_m * np.array([0, 0, B]) * dt_half
    S = 2 * T / (1 + np.sum(T**2))
    E_half = np.empty(particle_E.shape[1:])
    v_minus = np.empty(particle_vel.shape[1:])
    for i in range(active_particles):
        # velocity is 3D, E and x is 2D
        E_half[:] = q_d_m * particle_E[i,:] * dt_half
        v_minus[:] = particle_vel[i,:]
        v_minus[:-1] += E_half[:]
        v_prime = v_minus + cross(v_minus, T)
        v_plus = v_minus + cross(v_prime, S)
        v_plus[:-1] += E_half[:]
        particle_vel[i,:] = v_plus[:]
        particle_pos[i][:] += particle_vel[i][:-1] * dt



@numba.jit(nopython=True)
def kinematic_mover(particle_pos, particle_vel, active_particles, dt):
    for i in range(active_particles):
        # velocity is 3D, E and x is 2D
        particle_pos[i][:] += particle_vel[i][:-1] * dt
