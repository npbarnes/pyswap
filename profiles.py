#!/usr/bin/env python
import os
from os.path import join
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.constants import value, unit
from HybridReader2 import HybridReader2 as hr
from HybridParams import HybridParams
import matplotlib.pyplot as plt
from matplotlib.figure import figaspect
import NH_tools
import argparse
from argparse import Action

# Constants
eV2J = value('Boltzmann constant in eV/K')
k = value('Boltzmann constant')
assert unit('Boltzmann constant') == 'J K^-1'

# Command line parsing
parser = argparse.ArgumentParser()
parser.add_argument('--save', action='store_true')
parser.add_argument('--mccomas', action='store_true')
parser.add_argument('-v','--variable', dest='var', default='up')

group = parser.add_argument_group('Select which simulations to include (default --shell)')
group_ex = group.add_mutually_exclusive_group()
group_ex.add_argument('--all', dest='which', action='store_const', const='all')
group_ex.add_argument('--shell', dest='which', action='store_const', const='shell')
group_ex.add_argument('--no-shell', dest='which', action='store_const', const='no-shell')
group_ex.add_argument('--other', nargs='?', const=join(os.getcwd(), 'data'))
parser.set_defaults(which='shell')
parser.add_argument('--other-label', default='')

args = parser.parse_args()
if args.other:
    args.which = 'other'
if args.other_label and not args.other:
    parser.error('--other-label requires --other')
### End command line parsing

shell_prefixes = ['/home/nathan/data/2017-Mon-Nov-13/pluto-7/data',
                '/home/nathan/data/2018-Fri-Jan-26/pluto-1/data',
                '/home/nathan/data/2017-Mon-Nov-13/pluto-8/data']
no_shell_prefixes = ['/home/nathan/data/2018-Mon-Jul-09/pluto-1/data',
                    '/home/nathan/data/2018-Thu-Jul-05/pluto-1/data',
                    '/home/nathan/data/2018-Fri-Jun-29/pluto-1/data']
shell_labels = ['0.3nT', '0.19nT', '0.08nT'] 
no_shell_labels = ['No Shell 0.3nT', 'No Shell 0.19nT', 'No Shell 0.08nT']

all_prefixes = shell_prefixes + no_shell_prefixes
all_labels = shell_labels + no_shell_labels

if args.which == 'all':
    prefixes = all_prefixes
    labels = all_labels
elif args.which == 'shell':
    prefixes = shell_prefixes
    labels = shell_labels
elif args.which == 'no-shell':
    prefixes = no_shell_prefixes
    labels = no_shell_labels
elif args.which == 'other':
    prefixes = [args.other]
    labels = [args.other_label]

paras = []
for p in prefixes:
    paras.append(HybridParams(p).para)

grids = []
for p in paras:
    grids.append(p['grid_points'])

def profile(interp_points, grid_points, grid_data):
    rgi = RegularGridInterpolator(points=grid_points, values=grid_data, bounds_error=False)
    return rgi(interp_points)

points, o = NH_tools.trajectory(NH_tools.close_start, NH_tools.close_end, 60.)

profiles = []
for pre,grid in zip(prefixes,grids):
    v = hr(pre,args.var).get_last_timestep()[-1]
    vdata = np.linalg.norm(v,axis=-1)
    profiles.append(profile(points, grid, vdata))

plt.style.use('pluto-paper')
fig, vax = plt.subplots(figsize=figaspect(0.6))
if args.which == 'shell':
    vax.set_title("Simulation Flyby Velocity Profiles\nWith SW shell distribution")
elif args.which == 'no-shell':
    vax.set_title("Simulation Flyby Velocity Profiles\nWithout SW shell distribution")
else:
    vax.set_title("Simulation Flyby Velocity Profiles")

vax.set_xlabel('X ($R_p$)')

if args.mccomas:
    points[:,0] = -points[:,0]

for p,l in zip(profiles, labels):
    vax.plot(points[:,0]/1187., p, label=l)

vax.plot([-12.7,3.8,8.8,13.1,18.9,158.7,175.2],[400.,365.,324.,250.,140.,320.,400.], marker='o', linestyle='None', label='Data')
if not (args.other and not args.other_label):
    vax.legend()


vax.set_ylabel('|v| (km/s)')

if args.mccomas:
    vax.set_xlim([-20,50])
else:
    vax.invert_xaxis()
    vax.set_xlim([-10,-100])

if args.save:
    if args.which == 'shell':
        plt.savefig('sw_slowing_mccomas_comparison', bbox_inches='tight')
    elif args.which == 'no-shell':
        plt.savefig('sw_slowing_mccomas_comparison_noshell', bbox_inches='tight')
    else:
        plt.savefig('sw_slowing_mccomas_comparison_', bbox_inches='tight')
else:
    plt.show()

