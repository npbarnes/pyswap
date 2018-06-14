#!/usr/bin/env python
import os
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.constants import value, unit
from HybridReader2 import HybridReader2 as hr
from HybridReader2 import NoSuchVariable
from HybridParams import HybridParams
import matplotlib.pyplot as plt
import spice_tools

# Profile interpolator
def profile(interp_points, grid_points, grid_data):
    rgi = RegularGridInterpolator(points=grid_points, values=grid_data, bounds_error=False)
    return rgi(interp_points)

# Get trajectory points
points, _, _ = spice_tools.trajectory(spice_tools.flyby_start, spice_tools.flyby_end, 60.)


# Define functions for getting the values we want
def vmag(v):
    return np.linalg.norm(v, axis=-1)

def n_km(n):
    return n*1e-15

def t_K(t):
    eV_per_K = value('Boltzmann constant in eV/K')
    return t/eV_per_K

def pressure_pPa(n,t):
    k = value('Boltzmann constant')
    assert unit('Boltzmann constant') == 'J K^-1'
    return k*n*t*(100**3)*(1e12)

def b_pressure(b):
    b = 1.6726219e-27/1.60217662e-19 * b # proton gyrofrequency -> T
    ret = np.linalg.norm(b, axis=-1)**2/(2*4*3.14159e-7)
    return ret*1e12

def density_profile(hf, step=-1):
    hybrid_n = hr(hf, 'np')
    grid_points = hybrid_n.para['grid_points']
    n = hybrid_n.get_timestep(step)[-1]
    ndata = n_km(n)
    return profile(points, grid_points, ndata)

def pressure_profile(hf, step=-1):
    hybrid_n = hr(hf, 'np')
    grid_points = hybrid_n.para['grid_points']
    n = hybrid_n.get_timestep(step)[-1]
    ndata = n_km(n)

    try:
        hybrid_temp = hr(hf, 'temp_tot')
    except NoSuchVariable:
        hybrid_temp = hr(hf, 'temp_p')

    t = hybrid_temp.get_timestep(step)[-1]
    tdata = t_K(t)

    thermal_p_data = pressure_pPa(ndata, tdata)

    hybrid_b = hr(hf, 'bt')
    b = hybrid_b.get_timestep(step)[-1]
    bdata = b_pressure(b)

    p_therm = profile(points, grid_points, thermal_p_data)
    p_b     = profile(points, grid_points, bdata)

    return zip(p_therm, p_b)


def main(profile_func, title='Flyby Profile', labels=[], xlabel='', ylabel=''):
    import argparse

    # Some usage stuff
    def limitType(s):
        if s == 'auto':
            return None
        else:
            return float(s)

    parser = argparse.ArgumentParser()
    parser.add_argument('--mccomas', action='store_true')
    parser.add_argument('--hybrid-folder', dest='hybrid_folder', default=None)
    parser.add_argument('--prefix', default='data')
    parser.add_argument('--step', type=int, default=-1)
    parser.add_argument('--xlim', type=limitType, nargs=2, default=[10,100])
    parser.add_argument('--ylim', type=limitType, nargs=2, default=[0.008,1.1])

    args = parser.parse_args()

    if args.hybrid_folder is None:
        args.hybrid_folder = os.path.join(os.getcwd(), args.prefix)
    else:
        args.hybrid_folder = os.path.join(args.hybrid_folder, args.prefix)

    my_profile = profile_func(args.hybrid_folder, args.step)

    # Make a figure with an axis
    fig, ax = plt.subplots()

    if args.mccomas:
        points[:,0] = -points[:,0]

    lines = ax.plot(points[:,0]/1187., my_profile)
    for line, label in zip(lines, labels):
        line.set_label(label)

    ax.legend(loc='best')
    ax.set_yscale('log')

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if args.mccomas:
        ax.set_xlim(*args.xlim)
    else:
        ax.invert_xaxis()
        ax.set_xlim(*[-x for x in args.xlim])

    ax.set_ylim(*args.ylim)


    plt.savefig(os.path.join(args.hybrid_folder,'../pressure.png'))


if __name__ == '__main__':
    #main(density_profile)
    plt.style.use('pluto-paper')
    main(pressure_profile, title='Pressure Profiles\n0.19nT', labels=('Thermal Pressure','Magnetic Pressure'), xlabel='X ($R_p$)', ylabel='Pressure (pPa)')
