#!/usr/bin/python
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
args = parser.parse_args()

para = HybridParams(args.hybrid_folder).para


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

points, o = NH_tools.trajectory(NH_tools.close_start, NH_tools.close_end, 60.)

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

fig, (vax, nax, tax, pax) = plt.subplots(4, sharex=True)
vax.invert_xaxis()
vax.set_xlim([-10,-100])
pax.set_xlabel('X [$R_p$]')

vax.plot(points[:,0]/1187., v_profile)
nax.plot(points[:,0]/1187., n_profile)
tax.plot(points[:,0]/1187., t_profile)
pax.plot(points[:,0]/1187., p_profile)

vax.set_ylabel('<v$_{flow}$> [km/s]')
nax.set_ylabel('Density [cm$^{-3}$]')
tax.set_ylabel('Temp [K]')
pax.set_ylabel('nkT [pPa]')

nax.set_yscale('log')
tax.set_yscale('log')
pax.set_yscale('log')

plt.show()
