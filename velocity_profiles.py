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
parser.add_argument('hybrid_folder')
parser.add_argument('--mccomas', action='store_true')
args = parser.parse_args()

args.high = join(args.hybrid_folder,'pluto-7','data')
args.low  = join(args.hybrid_folder,'pluto-8','data')

high_para = HybridParams(args.high).para
low_para = HybridParams(args.low).para


def pluto_position(p):
    """get the position of pluto in simulation coordinates"""
    print("grid_profiles: setting pluto_offset to 30 since hpara is wrong")
    p['pluto_offset'] = 30
    return p['qx'][p['nx']/2 + p['pluto_offset']]

def get_grid_points(para):
    qx = para['qx'] - pluto_position(para)
    qy = para['qy'] - para['qy'][-1]/2
    qz = para['qzrange'] - para['qzrange'][-1]/2

    grid_points = qx, qy, qz
    return grid_points

high_grid = get_grid_points(high_para)
low_grid = get_grid_points(low_para)

def profile(interp_points, grid_points, grid_data):
    rgi = RegularGridInterpolator(points=grid_points, values=grid_data, bounds_error=False)
    return rgi(interp_points)

points, o = NH_tools.trajectory(NH_tools.close_start, NH_tools.close_end, 60.)

v = hr(args.high,'up').get_last_timestep()[-1]
vdata = np.linalg.norm(v, axis=-1)
high_profile = profile(points, high_grid, vdata)

v = hr(args.low,'up').get_last_timestep()[-1]
vdata = np.linalg.norm(v, axis=-1)
low_profile = profile(points, low_grid, vdata)

fig, vax = plt.subplots()
vax.set_title("Simulation Flyby Velocity Profile", fontsize=1.5*15)

vax.set_xlabel('X [$R_p$]')

if args.mccomas:
    points[:,0] = -points[:,0]

#vax.plot(points[:,0]/1187., high_profile, label='0.3nT')
#vax.plot(points[:,0]/1187., low_profile, label='0.08nT')
vax.plot([-12.7,3.8,8.8,13.1,18.9,158.7,175.2],[400.,365.,324.,250.,140.,320.,400.], marker='o', linestyle='None', label='Data', color='C2')
vax.legend()

vax.set_ylabel('<v$_{flow}$> [km/s]')

if args.mccomas:
    vax.set_xlim([-20,200])
else:
    vax.invert_xaxis()
    vax.set_xlim([-10,-100])

plt.show()

