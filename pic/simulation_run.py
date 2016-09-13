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
from pic.movers import kinematic_mover, electrostatic_mover
from pic.collisions import collide_with_neutrals
from pic.running_statistics import update_mean_estimate, std_from_means


def assert_is_not_nan(arr):
    assert not np.any(np.isnan(arr))


Regions = namedtuple('Regions', ('main', 'reservoir'))
Particles = namedtuple('Particles', ('x', 'v', 'n', 'q', 'm'))


# TODO loop over V_fl with alocated arrays, just change initial values?
def simulate_probe_current(probe_setup, U_probe, N, T_e, dt, max_iterations, callback=None):
    # TODO physical basis for parameters
    grid_shape = probe_setup.grid_shape # 2D rectangular grid
    h = probe_setup.h
    frac_N = N // 2             # each species is fraction of N all particle
    active_particles = int(frac_N*0.75) # leave some space for extra particles
    region_length = grid_shape[0] * h
    # initialize particle kinematics for each species and main/reservoir regions
    regions = Regions._make(
        [Particles(init.uniform_positions(region_length, frac_N),
                   init.maxwell_velocities(T_e, m, frac_N),
                   [active_particles], charge, m)
         for (charge, m) in ((spc.e, spc.m_p), (-spc.e, spc.m_e))]
                 for region in Regions._fields)
    # probe potential
    phi_probe = probe_setup.get_potential(U_probe)
    poisson_solver = create_poisson_solver(grid_shape[0])
    # initial iteration values
    iterations = 0
    rho = np.empty(grid_shape)
    particle_E = np.empty((frac_N, 2))
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
        macro_particle = 1e4
        phi = poisson_solver(rho*h**2/spc.epsilon_0*macro_particle)
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
        # collisions with neutrals
        collided_fraction_e = 0.01 # estimate for electrons
        for particles in regions.main:
            # collision time will be proportional to v, ratios of v^2 are
            # inverse to ratios of m
            collided_fraction = collided_fraction_e * np.sqrt(spc.m_e/particles.m)
            collide_with_neutrals(collided_fraction, T_e, spc.m_p, particles.m,
                                  particles.v, particles.n[0])
        # next iteration
        iterations += 1
        if callback is not None:
            callback(regions, rho, phi, iterations, j_probe)
        # check current estimate stability
        skip_initial_samples = 10
        if iterations > skip_initial_samples:
            samples_count = iterations - skip_initial_samples
            j_probe_mean = update_mean_estimate(j_probe, j_probe_mean, samples_count)
            j_probe_mean_sq = update_mean_estimate(j_probe**2, j_probe_mean_sq, samples_count)
            j_probe_std = std_from_means(j_probe_mean, j_probe_mean_sq)
            if samples_count > 10 and np.abs(np.divide(j_probe_std, j_probe_mean)) < 0.05:
                break
    return j_probe_mean, j_probe_std

def callback_anim_rho(reg, rho, phi, it, i):
    plt.cla()
    plt.imshow(rho, cmap=plt.cm.plasma)
    plt.pause(1.0/30)
    print('iteration', it, 'with current', i)


def callback_stats(reg, rho, phi, it, i):
    plt.cla()
    for part in reg.main:
        n = part.n[0]
        bins = int(n**0.5)
        plt.hist(np.linalg.norm(part.v[:n], axis=1)**2*part.m, bins=bins, alpha=0.5)
        print('energy', 0.5*part.m*np.mean(np.sum(part.v[:n]**2, axis=1))/spc.eV)
    plt.pause(1.0/30)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    plt.ion()
    probe = ProbeSetup(100, 9, 1e-6)
    T_e = 10
    U_pr_max = 1e2
    U_pr = np.linspace(-U_pr_max, U_pr_max)
    j_probe = np.empty_like(U_pr)
    j_probe_std = np.empty_like(U_pr)
    plt.gca()
    for i in range(U_pr.shape[0]):
        print('Simulation', i+1, 'of', U_pr.shape[0], 'with', U_pr[i], 'V')
        j_probe[i], j_probe_std[i] = simulate_probe_current(probe, U_pr[i], 10000, T_e, 1e-12, 1000)
    # display results as plot
    plt.errorbar(U_pr, j_probe, j_probe_std, fmt='ko', label='simulation')
    plt.title('$T_e=%.0f$ eV' % T_e)
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
