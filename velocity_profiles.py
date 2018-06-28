#!/usr/bin/python
from os.path import join
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.constants import value, unit
from HybridReader2 import HybridReader2 as hr
from HybridParams import HybridParams
import matplotlib.pyplot as plt
import NH_tools
import argparse

eV2J = value('Boltzmann constant in eV/K')
k = value('Boltzmann constant')
assert unit('Boltzmann constant') == 'J K^-1'

parser = argparse.ArgumentParser()
parser.add_argument('--save', action='store_true')
parser.add_argument('--mccomas', action='store_true')
args = parser.parse_args()

prefixes = ['/home/nathan/data/2017-Mon-Nov-13/pluto-7/data',
            '/home/nathan/data/2018-Fri-Jan-26/pluto-1/data',
            '/home/nathan/data/2017-Mon-Nov-13/pluto-8/data',
            '/home/nathan/data/2018-Wed-Jun-06/pluto-1/data']
labels = ['0.3nT', '0.19nT', '0.08nT', 'No Shell 0.19nT']

paras = []
for p in prefixes:
    paras.append(HybridParams(p).para)

def pluto_position(p):
    """get the position of pluto in simulation coordinates"""
    return p['qx'][p['nx']/2 + p['pluto_offset']]

def get_grid_points(para):
    qx = para['qx'] - pluto_position(para)
    qy = para['qy'] - para['qy'][-1]/2
    qz = para['qzrange'] - para['qzrange'][-1]/2

    grid_points = qx, qy, qz
    return grid_points

grids = []
for p in paras:
    grids.append(get_grid_points(p))

def profile(interp_points, grid_points, grid_data):
    rgi = RegularGridInterpolator(points=grid_points, values=grid_data, bounds_error=False)
    return rgi(interp_points)

points, o = NH_tools.trajectory(NH_tools.close_start, NH_tools.close_end, 60.)

profiles = []
for pre,grid in zip(prefixes,grids):
    v = hr(pre,'up').get_last_timestep()[-1]
    vdata = np.linalg.norm(v,axis=-1)
    profiles.append(profile(points, grid, vdata))

plt.style.use('pluto-paper')
fig, vax = plt.subplots()
vax.set_title("Simulation Flyby Velocity Profiles")

vax.set_xlabel('X ($R_p$)')

if args.mccomas:
    points[:,0] = -points[:,0]

for p,l in zip(profiles, labels):
    vax.plot(points[:,0]/1187., p, label=l)

vax.plot([-12.7,3.8,8.8,13.1,18.9,158.7,175.2],[400.,365.,324.,250.,140.,320.,400.], marker='o', linestyle='None', label='Data')
vax.legend()

vax.set_ylabel('|v| (km/s)')

if args.mccomas:
    vax.set_xlim([-20,200])
else:
    vax.invert_xaxis()
    vax.set_xlim([-10,-100])

if args.save:
    plt.savefig('sw_slowing_mccomas_comparison', bbox_inches='tight')
else:
    plt.show()

