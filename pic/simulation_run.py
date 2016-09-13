from __future__ import print_function
import scipy.constants as spc
import numpy as np
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
Particles = namedtuple('Particles', ('x', 'v', 'dt', 'n', 'q', 'm', 'rho'))


def simulate_probe_current(probe_setup, U_probe, N, N_macro, T_e, B, hops_per_cell, max_iterations, callback=None):
    grid_shape = probe_setup.grid_shape # 2D rectangular grid
    h = probe_setup.h
    active_particles = int(N*0.8) # leave some space for extra particles
    region_length = grid_shape[0] * h
    # initialize particle kinematics for each species and main/reservoir regions
    regions = Regions._make(
        [Particles(init.uniform_positions(region_length, N),
                   init.maxwell_velocities(T_e, m, N),
                   init.optimum_dt(h, T_e, m, hops_per_cell), [active_particles], charge, m, np.empty(grid_shape))
         for (charge, m) in ((spc.e, spc.m_p), (-spc.e, spc.m_e))]
                 for region in Regions._fields)
    # probe potential
    phi_probe = probe_setup.get_potential(U_probe)
    poisson_solver = create_poisson_solver(grid_shape[0])
    # initial iteration values
    iterations = 0
    particle_E = np.empty((N, 2))
    j_probe_mean = 0.0
    j_probe_mean_sq = 0.0
    species = 0                 # start moving electrons
    last_species_switch = 0
    probe_incident_charge = 0.0
    while iterations < max_iterations:
        # select species to iterate on
        steps_with_species = iterations - last_species_switch
        if species == 0 and steps_with_species == 10: # perform 10 steps with ions
            species = 1             # switch to electrons
            steps_with_species = iterations
        elif species == 1 and steps_with_species == 100: # perform 100 steps with electrons
            species = 0             # switch to ions
            steps_with_species == iterations
        particles = regions.main[species]
        particles_r = regions.reservoir[species]
        # detect boundary interactions
        # TODO possible optimization: weight particles here
        particles.n[0], lost_in_probe = boundary_crossings(
            particles.x, particles.v, 0, region_length, probe_setup.min, probe_setup.max,
            particles.n[0])
        probe_incident_charge += particles.q * lost_in_probe
        # get reflected particles in reservoir
        reflected_particles = boundary_reflections(particles_r.x, particles_r.v, 0, region_length)
        # assign reflected as new, make sure enough space is there
        free_particles = N - particles.n[0]
        transferable = min(free_particles, reflected_particles)
        main_sl = slice(particles.n[0], particles.n[0]+transferable)
        reservoir_sl = slice(None, transferable) # [:transferable]
        for d in ('x', 'v'):
            main_d = getattr(particles, d)
            reservoir_d = getattr(particles_r, d)
            main_d[main_sl] = reservoir_d[reservoir_sl]
        particles.n[0] += transferable
        # perform CIC charge weighting
        particles.rho[:] = 0.0            # TODO optimization: perform in some for loop
        cic_charge_weighting(particles.x, particles.q, particles.n[0], particles.rho, h)
        rho = sum(particles.rho for particles in regions.main)
        # solve Poisson equation
        phi = poisson_solver(rho*h**2/spc.epsilon_0*N_macro)
        # add probe potential
        phi += phi_probe
        # calculate E from phi
        E = -np.dstack(np.gradient(phi, h, h)) # TODO *-1 in some for loop
        # move particles
        kinematic_mover(particles_r.x, particles_r.v, N, particles_r.dt)
        # CIC field weighting
        cic_field_weighting(particles.x, particle_E, particles.n[0], E, h)
        if B == 0:
            electrostatic_mover(particles.x, particles.v, particles.q,
                                particles.m, particle_E, particles.n[0],
                                particles.dt)
        else:
            electrostatic_homog_B_mover(particles.x, particles.v,
                                        particles.q, particles.m, B,
                                        particle_E, particles.n[0],
                                        particles.dt)
        # collisions with neutrals
        collided_fraction_e = 0.01 # estimate for electrons
        # are inverse to ratios of m at the same T
        collided_fraction = collided_fraction_e * np.sqrt(spc.m_e/particles.m)
        collide_with_neutrals(collided_fraction, T_e, spc.m_p, particles.m,
                                particles.v, particles.n[0])
        # next iteration
        iterations += 1
        if callback is not None:
            callback(regions, rho, phi, iterations, j_probe)
    return probe_incident_charge / max_iterations  / regions.main[0].dt, 0

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


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    plt.ion()
    T_e = 10                    # eV
    B = 0# T
    grid_size = 100
    h = 1e-4
    probe_size = 9
    N = 10000
    N_macro = 10000
    n_e = N*N_macro/(h*grid_size)**3 # rough estimate as this is 2D sim
    l_D = debye_length(T_e, n_e)
    hops_per_cell = 10
    max_iterations = grid_size * hops_per_cell # must be able to go through whole grid
    assert h < l_D / 3, "cell size must be small than lambda_D/3"
    assert l_D *3 < grid_size * h, "Grid length must be larger than 3*lambda_D"
    probe = ProbeSetup(grid_size, probe_size, h)
    U_pr_max = 1e2
    U_pr = np.linspace(-U_pr_max, U_pr_max, 10)
    j_probe = np.empty_like(U_pr)
    j_probe_std = np.empty_like(U_pr)
    plt.gca()
    for i in range(U_pr.shape[0]):
        print('Simulation', i+1, 'of', U_pr.shape[0], 'with', U_pr[i], 'V')
        j_probe[i], j_probe_std[i] = simulate_probe_current(probe, U_pr[i], N, N_macro, T_e, B, hops_per_cell, max_iterations)
    # display results as plot
    plt.errorbar(U_pr, j_probe, j_probe_std, fmt='ko', label='simulation')
    plt.title('$T_e=%.0f$ eV, $B_z=%.0f$ T' % (T_e, B))
    plt.ylabel('$j_{probe}$ [A/m^2]')
    plt.xlabel('$U_{probe}$ [V]')
    ylims = plt.ylim()
    plt.grid()
    # try fitting analytic curve
    from scipy.optimize import curve_fit
    V_fl = U_pr[np.abs(j_probe).argmin()]
    fit_sl = slice(None, U_pr.shape[0]//2)
    try:
        p0 = [j_probe.max(), V_fl, T_e]
        p, err = curve_fit(v_a_characteristic, U_pr[fit_sl], j_probe[fit_sl], p0=p0,
                           sigma=j_probe_std[fit_sl], absolute_sigma=True)
    except RuntimeError:
        p = None
    if p is not None:
        plt.plot(U_pr, v_a_characteristic(U_pr, *p), 'r-',
                 label=('$%.0f\\left(1-\\exp\\left(\\frac{U_{probe}  %+.1f}{%.1f} \\right)\\right)$' % tuple(p)))
        plt.ylim(*ylims)        # restore lims to sim data
    plt.legend(loc='lower left')
