from __future__ import print_function
import scipy.constants as spc
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple
from pic.electrostatic_probe import ProbeSetup, v_a_characteristic
from pic import initialization as init
from pic.boundary_crossing import boundary_crossings, boundary_reflections
from pic.weighting import cic_charge_weighting, cic_field_weighting
from pic.poisson_lu_solver import create_poisson_solver
from pic.poisson_sor_solver import solve_poisson, optimum_sor_omega
from pic.movers import kinematic_mover, electrostatic_mover, electrostatic_homog_B_mover
from pic.collisions import collide_with_neutrals
from pic.running_statistics import update_mean_estimate, std_from_means
from pic.physical_scales import debye_length


Regions = namedtuple('Regions', ('main', 'reservoir'))
Particles = namedtuple('Particles', ('x', 'v', 'n', 'q', 'm'))


def simulate_probe_current(probe_setup, U_probe, N, N_macro, T_e, B, hops_per_cell_e, max_iterations, callback=None):
    grid_shape = probe_setup.grid_shape # 2D rectangular grid
    h = probe_setup.h
    n_e = N*N_macro/(h*grid_shape[0])**3 # rough estimate as this is 2D sim
    l_D = debye_length(T_e, n_e)
    assert h < l_D / 3, "Cell size must be small than lambda_D/3"
    assert l_D *3 < grid_shape[0] * h, "Grid length must be larger than 3*lambda_D"
    active_particles = int(N*0.75) # leave some space for extra particles
    region_length = grid_shape[0] * h
    # initialize particle kinematics for each species and main/reservoir regions
    regions = Regions._make(
        [Particles(init.uniform_positions(region_length, N),
                   init.maxwell_velocities(T_e, m, N),
                   [active_particles], charge, m)
         for (charge, m) in ((spc.e, spc.m_p), (-spc.e, spc.m_e))]
                 for region in Regions._fields)
    dt = init.optimum_dt(h, T_e, spc.m_e, hops_per_cell_e)
    # probe potential
    phi_probe = probe_setup.get_potential(U_probe)
    poisson_solver = create_poisson_solver(grid_shape[0])
    # initial iteration values
    iterations = 0
    rho = np.empty(grid_shape)
    particle_E = np.empty((N, 2))
    j_probe_mean = 0.0
    j_probe_mean_sq = 0.0
    while iterations < max_iterations:
        # detect boundary interactions
        lost_charge = 0.0
        for particles in regions.main: # TODO possible optimization: weight particles here
            particles.n[0], lost_in_probe = boundary_crossings(
                particles.x, particles.v, 0, region_length, probe_setup.min, probe_setup.max,
                particles.n[0])
            lost_charge += particles.q * lost_in_probe
        j_probe = lost_charge / dt / probe_setup.S # current density
        for s_i, particles in enumerate(regions.reservoir):
            main_particles = regions.main[s_i]
            reflected_particles = boundary_reflections(particles.x, particles.v, 0, region_length)
            # assign reflected as new, make sure enough space is there
            free_particles = N - main_particles.n[0]
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
        phi = poisson_solver(rho*h**2/spc.epsilon_0*N_macro)
        # add probe potential
        phi += phi_probe
        # calculate E from phi
        E = -np.dstack(np.gradient(phi, h, h)) # TODO *-1 in some for loop
        # move particles
        for particles in regions.reservoir:
            kinematic_mover(particles.x, particles.v, N, dt)
        for particles in regions.main:
            # CIC field weighting
            cic_field_weighting(particles.x, particle_E, particles.n[0], E, h)
            if B == 0:
                electrostatic_mover(particles.x, particles.v, particles.q,
                                    particles.m, particle_E, particles.n[0],
                                    dt)
            else:
                electrostatic_homog_B_mover(particles.x, particles.v,
                                            particles.q, particles.m, B,
                                            particle_E, particles.n[0], dt)
        # collisions with neutrals
        collided_fraction_e = 0.01 # estimate for electrons
        for particles in regions.main:
            # collision time will be inversely proportional to v, ratios of v^2
            # are inverse to ratios of m at the same T
            collided_fraction = collided_fraction_e * np.sqrt(spc.m_e/particles.m)
            collide_with_neutrals(collided_fraction, T_e, spc.m_p, particles.m,
                                  particles.v, particles.n[0])
        # next iteration
        iterations += 1
        if callback is not None:
            callback(regions, rho, phi, iterations, j_probe)
        # check current estimate stability
        skip_initial_samples = int(0.25 * max_iterations) # wait for partial equilibrium
        if iterations > skip_initial_samples:
            samples_count = iterations - skip_initial_samples
            j_probe_mean = update_mean_estimate(j_probe, j_probe_mean, samples_count)
            j_probe_mean_sq = update_mean_estimate(j_probe**2, j_probe_mean_sq, samples_count)
            j_probe_std = std_from_means(j_probe_mean, j_probe_mean_sq)
    return j_probe_mean, j_probe_std

def callback_anim_rho(reg, rho, phi, it, i):
    plt.cla()
    plt.imshow(rho, cmap=plt.cm.plasma)
    plt.pause(1e-3)
    print('iteration', it, 'with current', i)


def callback_stats(reg, rho, phi, it, i):
    plt.cla()
    for part in reg.main:
        n = part.n[0]
        bins = int(n**0.5)
        plt.hist(np.linalg.norm(part.v[:n], axis=1)**2*part.m, bins=bins, alpha=0.5)
        print('energy', 0.5*part.m*np.mean(np.sum(part.v[:n]**2, axis=1))/spc.eV)
    plt.pause(1e-3)


