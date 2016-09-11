from __future__ import print_function
from scipy.stats import norm, uniform
import scipy.constants as spc
import numpy as np
from collections import namedtuple
from pic.electrostatic_probe import ProbeSetup
from pic.boundary_crossing import boundary_crossings, boundary_reflections
from pic.weighting import cic_charge_weighting, cic_field_weighting
from pic.poisson_lu_solver import create_poisson_solver
from pic.poisson_sor_solver import solve_poisson, optimum_sor_omega
from pic.movers import kinematic_mover, electrostatic_mover

# TODO use (dim, N) arrays for C-column order because we vectorize over rows?

def initialize_particle_kinematics(region_length, T_e, m, N):
    # TODO will need separate mass for electrons and ions
    pos = uniform.rvs(loc=0, scale=region_length, size=(N, 2)) # 2D position
    vel_var = T_e * spc.eV / m # variance of normal dist
    vel = norm.rvs(scale=vel_var**0.5, size=(N, 3))
    return pos, vel


def assert_is_not_nan(arr):
    assert not np.any(np.isnan(arr))


Regions = namedtuple('Regions', ('main', 'reservoir'))
Particles = namedtuple('Particles', ('x', 'v', 'n', 'q', 'm'))


# TODO loop over V_fl with alocated arrays, just change initial values?
def simulate_probe_current(probe_setup, U_probe, N, T_e, dt, max_iterations, callback):
    # TODO physical basis for parameters
    grid_shape = probe_setup.grid_shape # 2D rectangular grid
    h = probe_setup.h
    frac_N = N // 2             # each species is fraction of N all particle
    active_particles = int(frac_N*0.75) # leave some space for extra particles
    region_length = grid_shape[0] * h
    # initialize particle kinematics for each species and main/reservoir regions
    regions = Regions._make(
        [Particles._make(initialize_particle_kinematics(region_length, T_e, m, frac_N)
                         +([active_particles], charge, m))
         for (charge, m) in ((spc.e, spc.m_p), (-spc.e, spc.m_e))]
                 for region in Regions._fields)
    # probe potential
    phi_probe = probe_setup.get_potential(U_probe)
    poisson_solver = create_poisson_solver(grid_shape[0])
    # initial iteration values
    iterations = 0
    rho = np.empty(grid_shape)
    particle_E = np.empty((frac_N, 2))
    while iterations < max_iterations:
        # detect boundary interactions
        lost_charge = 0.0
        for particles in regions.main: # TODO possible optimization: weight particles here
            particles.n[0], lost_in_probe = boundary_crossings(
                particles.x, particles.v, 0, region_length, probe_setup.min, probe_setup.max,
                particles.n[0])
            lost_charge += particles.q * lost_in_probe
        j_probe = lost_charge / dt / probe.S # current density
        for s_i, particles in enumerate(regions.reservoir):
            main_particles = regions.main[s_i]
            reflected_particles = boundary_reflections(particles.x, particles.v, 0, region_length)
            # assign reflected as new, make sure enough space is there
            free_particles = frac_N - main_particles.n[0]
            transferable = min(free_particles, reflected_particles)
            main_sl = slice(main_particles.n[0], main_particles.n[0]+transferable)
            reservoir_sl = slice(None, transferable) # [:transferable]
            for d in ('x', 'v'):
                main_d = getattr(main_particles, d)
                reservoir_d = getattr(particles, d)
                main_d[main_sl] = reservoir_d[reservoir_sl]
            main_particles.n[0] += transferable
        # perform CIC charge weighting
        rho[:] = 0.0            # TODO optimization: perform in some for loop
        for particles in regions.main:
            cic_charge_weighting(particles.x, particles.q, particles.n[0], rho, h)
        # solve Poisson equation
        phi = poisson_solver(rho*h**2/spc.epsilon_0*1e3)
        # add probe potential
        phi += phi_probe
        # calculate E from phi
        E = -np.dstack(np.gradient(phi, h, h)) # TODO *-1 in some for loop
        # move particles
        for particles in regions.reservoir:
            kinematic_mover(particles.x, particles.v, frac_N, dt)
        for particles in regions.main:
            # CIC field weighting
            cic_field_weighting(particles.x, particle_E, particles.n[0], E, h)
            electrostatic_mover(particles.x, particles.v, particles.q,
                                particles.m, particle_E, particles.n[0], dt)
        # next iteration
        iterations += 1
        callback(regions, rho, phi, iterations, j_probe)
    return j_probe


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    plt.ion()
    def callback(reg, rho, phi, it, i):
        plt.cla()
        plt.imshow(phi)
        plt.pause(1.0/30)
        print('iteration', it, 'with current', i)
    probe = ProbeSetup(100, 9, 1e-5)
    i_probe = simulate_probe_current(probe, 1e-12, 10000, 50, 1e-10, 100, callback)
    print(i_probe)
