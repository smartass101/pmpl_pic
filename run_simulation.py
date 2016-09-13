#!/usr/bin/env python

from __future__ import print_function
import argparse
import progressbar
import numpy as np
from pic.simulation_run import simulate_probe_current, callback_anim_rho
from pic.electrostatic_probe import ProbeSetup, plot_fit_v_a_characteristic
import matplotlib.pyplot as plt

def create_parser():
    parser = argparse.ArgumentParser(description='2D3V PIC simulation of a probe in plasma',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--T_e', type=float, default=10, help='electron temperature [eV]')
    parser.add_argument('--B_z', type=float, default=0, help='perpendicular magnetic field [T]')
    parser.add_argument('--grid_size', type=int, default=100, help='points in grid in 1D')
    parser.add_argument('--cell_length', type=float, default=1e-4, help='cell length [m]')
    parser.add_argument('--probe_size', type=int, default=9, help='probe size in 1D grid points')
    parser.add_argument('--max_particles', type=int, default=10000, help='maximum particle per species')
    parser.add_argument('--macro_particle', type=int, default=10000)
    parser.add_argument('--hops_per_cell_e', type=int, default=10, help='how many electron dt to cover 1 cell')
    parser.add_argument('--U_pr_abs_max', type=float, default=1e2, help='max abs(U_pr) applied probe voltage [B]')
    parser.add_argument('--U_pr_steps', type=int, default=10, help='how many steps in (-U_pr_max, U_pr_max)')
    parser.add_argument('--animate_rho', action='store_true')
    return parser


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    max_iterations = args.grid_size * args.hops_per_cell_e # must be able to go through whole grid
    probe = ProbeSetup(args.grid_size, args.probe_size, args.cell_length)
    U_pr = np.linspace(-args.U_pr_abs_max, args.U_pr_abs_max, args.U_pr_steps)
    j_probe = np.empty_like(U_pr)
    j_probe_std = np.empty_like(U_pr)
    plt.gca()
    pbar = progressbar.ProgressBar(max_value=U_pr.shape[0]*max_iterations).start()
    for i in range(U_pr.shape[0]):
        print('Simulation', i+1, 'of', U_pr.shape[0], 'with', U_pr[i], 'V')
        def callback_progress(*callback_args):
            pbar.update(i*max_iterations + callback_args[3])
            if args.animate_rho:
                callback_anim_rho(*callback_args)
        j_probe[i], j_probe_std[i] = simulate_probe_current(probe, U_pr[i], args.max_particles,
                                                            args.macro_particle, args.T_e, args.B_z, args.hops_per_cell_e, max_iterations, callback_progress)
    # display results as plot
    plt.cla()
    plot_fit_v_a_characteristic(U_pr, j_probe, j_probe_std, args.T_e, args.B_z)
    plt.show()

