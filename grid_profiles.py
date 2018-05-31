#!/usr/bin/env python
import os
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.constants import value, unit
from HybridReader2 import HybridReader2 as hr
from HybridParams import HybridParams
import matplotlib.pyplot as plt
import spice_tools
import argparse

eV2J = value('Boltzmann constant in eV/K')
k = value('Boltzmann constant')
assert unit('Boltzmann constant') == 'J K^-1'

parser = argparse.ArgumentParser()
parser.add_argument('--mccomas', action='store_true')
parser.add_argument('--hybrid-folder', dest='hybrid_folder', default=None)
parser.add_argument('-b', action='store_true')
parser.add_argument('--prefix', default='data')
args = parser.parse_args()

if args.hybrid_folder is None:
    args.hybrid_folder = os.path.join(os.getcwd(), args.prefix)
else:
    args.hybrid_folder = os.path.join(args.hybrid_folder, args.prefix)

para = HybridParams(args.hybrid_folder).para

def b_pressure(b):
    b = 1.6726219e-27/1.60217662e-19 * b # proton gyrofrequency -> T
    ret = np.linalg.norm(b, axis=-1)**2/(2*4*3.14159e-7)
    return ret*1e12


def pluto_position(p):
    """get the position of pluto in simulation coordinates"""
    return p['qx'][p['nx']/2 + p['pluto_offset']]

qx = para['qx'] - pluto_position(para)
qy = para['qy'] - para['qy'][-1]/2
qz = para['qzrange'] - para['qzrange'][-1]/2

grid_points = qx, qy, qz

def profile(interp_points, grid_points, grid_data):
    rgi = RegularGridInterpolator(points=grid_points, values=grid_data, bounds_error=False)
    return rgi(interp_points)

points, o, times = spice_tools.trajectory(spice_tools.flyby_start, spice_tools.flyby_end, 60.)

v = hr(args.hybrid_folder,'up').get_last_timestep()[-1]
vdata = np.linalg.norm(v, axis=-1)
v_profile = profile(points, grid_points, vdata)

n = hr(args.hybrid_folder, 'np').get_last_timestep()[-1]
ndata = n*1e-15
n_profile = profile(points, grid_points, ndata)

t = hr(args.hybrid_folder,'temp_p').get_last_timestep()[-1]
tdata = t/eV2J
t_profile = profile(points, grid_points, tdata)

pdata = k*ndata*tdata*(100**3)*(1e12) #pressure in pPa
p_profile = profile(points, grid_points, pdata)

b = hr(args.hybrid_folder, 'bt').get_last_timestep()[-1]
bdata = b_pressure(b)
b_profile = profile(points, grid_points, bdata)

#fig, (vax, nax, tax, pax) = plt.subplots(4, sharex=True)
fig, pax = plt.subplots()
#vax.set_title("Simulation Flyby Profiles", fontsize=1.5*15)

pax.set_xlabel('X [$R_p$]')

if args.mccomas:
    points[:,0] = -points[:,0]

#vax.plot(points[:,0]/1187., v_profile)
#nax.plot(points[:,0]/1187., n_profile)
#tax.plot(points[:,0]/1187., t_profile)
pax.plot(points[:,0]/1187., p_profile, label='Thermal Pressure')
pax.plot(points[:,0]/1187., b_profile, label='Magnetic Pressure')

#vax.set_ylabel('<v$_{flow}$> [km/s]')
#nax.set_ylabel('Density [cm$^{-3}$]')
#tax.set_ylabel('Temp [K]')
pax.set_ylabel('nkT [pPa]')

#nax.set_yscale('log')
#tax.set_yscale('log')
pax.set_yscale('log')

pax.set_title("Flyby pressure profile")
pax.legend()

if args.mccomas:
    pax.set_xlim([10,100])
else:
    pax.invert_xaxis()
    pax.set_xlim([-10,-100])

plt.show()

