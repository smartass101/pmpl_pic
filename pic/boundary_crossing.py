import numpy as np
import numba


@numba.jit(nopython=True)
def boundary_crossings(particle_pos, particle_vel, particle_charge, x_min,
                       x_max, probe_min, probe_max,
                       active_particles):
    """

    Assumes rectangular mesh

    """
    lost_charge = 0.0
    while i < active_particles: # number of active particle will decrease
        beyond_boundary = np.any(particle_pos[i] < x_min) or np.any(x_max < particle_pos[i])
        in_probe = np.all(probe_min <= probe_pos[i]) and np.all(probe_pos[i] <= probe_max)
        if beyond_boundary or in_probe: # lost particle
            active_particles -= 1 # lost a particle, will use as index of last particle
            if in_probe:          # count charge lost in probe
                lost_charge += particle_charge
            # copy last particle over lost particle
            particle_pos[i] = particle_pos[active_particles]
            particle_vel[i] = particle_vel[active_particles]
            particle_charge[i] = particle_charge[i]
            continue            # process copied last particle at this index i
        i += 1                  # next particle if possible
    return active_particles, lost_charge


@numba.jit(nopython=True)
def swap_in_array(arr, i, j, tmp):
    tmp[...] = arr[i][...]
    arr[i][...] = arr[j][...]
    arr[j][...] = tmp[...]


@numba.jit(nopython=True)
def boundary_reflections(particle_pos, particle_vel, x_min, x_max):
    """Detect, select and count particles reflected off boundaries

    Assumes rectangular mesh.

    Returns
    -------
    reflected_particles : int
        Number of reflected particles. They are put at the beginning of the
        particle_* arrays, i.e. particle_pos[:reflected_particles].
    """
    reflected_particles = 0
    # TODO need to return reflected particles, maybe by shifting them to front
    # TODO returning the reflectd ones would gibe the mirror image of particles entering the region
    # for swapping
    tmp_pos = np.empty(particle_pos.shape[1:], dtype=particle_pos.dtype)
    tmp_vel = np.empty(particle_vel.shape[1:], dtype=particle_vel.dtype)
    for i in range(particle_pos.shape[0]): # for all moving particles
        reflected = False                  # assume not reflected
        for j in range(particle_pos.shape[1]): # check each position
            below = particle_pos[i][j] < x_min
            above = x_max < particle_pos[i][j]
            if above or below:
                reflected = True
                particle_vel[i][j] *= -1 # bounce off component reversed
                boundary = x_min if below else x_max
                particle_pos[i][j] = 2*boundary - particle_pos[i][j]
        if reflected:
            # swap with first particle after last reflected particle
            swap_in_array(particle_pos, i, reflected_particles, tmp_pos)
            swap_in_array(particle_vel, i, reflected_particles, tmp_vel)
            reflected_particles += 1
    return reflected_particles
