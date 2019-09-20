#!/usr/bin/env python
import os
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.constants import value, unit
from HybridReader2 import HybridReader2 as hr
from HybridReader2 import NoSuchVariable
from HybridParams import HybridParams
import matplotlib.pyplot as plt
from matplotlib.figure import figaspect
import spice_tools
import argparse
from argparse_utils import ExtendAction, ExtendConstAction, copy_items

# Profile interpolator
def profile(interp_points, grid_points, grid_data):
    rgi = RegularGridInterpolator(points=grid_points, values=grid_data, bounds_error=False)
    return rgi(interp_points)

# Get trajectory points
# These must be in internal coordinates since the grid interpolation happens in internal coordinates.
# Plotting may be in either internal or McComas coords
points, _, _ = spice_tools.trajectory(spice_tools.flyby_start, spice_tools.flyby_end, 60., mccomas=False)

# Define modifier functions for converting internal variables to values for plotting
def vector_mag(v):
    return np.linalg.norm(v, axis=-1)

def vector_x(v):
    # in km/s
    return v[:,:,:,0]*1000

def sw_normalized_velocity(v):
    return vector_mag(v)/401.0

def n_km2cm(n):
    """Converts number density (n) units from km^-3 to cm^-3"""
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

def direct_profiler_factory(var, modifier=None):
    def profiler(hf, step=-1):
        hybrid = hr(hf, var)
        grid_points = hybrid.para['grid_points']
        v = hybrid.get_timestep(step)[-1]
        if modifier is not None:
            data = modifier(v)
        else:
            data = v

        return profile(points, grid_points, data)
    return profiler

density_profiler = direct_profiler_factory('np', modifier=n_km2cm)
heavy_profiler = direct_profiler_factory('np_CH4', modifier=n_km2cm)
light_profiler = direct_profiler_factory('np_H', modifier=n_km2cm)
H_profiler = direct_profiler_factory('np_H', modifier=n_km2cm)
He_profiler = direct_profiler_factory('np_He', modifier=n_km2cm)
vnorm_profiler = direct_profiler_factory('up', modifier=sw_normalized_velocity)
vx_profiler = direct_profiler_factory('up', modifier=vector_x)

def mass_density_profiler(hf, step=-1):
    mproton = 1.67e-27 #kg
    H     = mproton*H_profiler(hf, step=step)
    He    = 4*mproton*He_profiler(hf, step=step)
    Heavy = 16*mproton*heavy_profiler(hf, step=step)

    return H+He+Heavy

def pressure_profiler(hf, step=-1):
    try:
        hybrid_n = hr(hf, 'np_tot')
    except NoSuchVariable:
        hybrid_n = hr(hf, 'np')

    grid_points = hybrid_n.para['grid_points']
    n = hybrid_n.get_timestep(step)[-1]
    ndata = n_km2cm(n)

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

def thermal_pressure_profiler(hf, step=-1):
    """In Pa"""
    try:
        hybrid_n = hr(hf, 'np_tot')
    except NoSuchVariable:
        hybrid_n = hr(hf, 'np')

    grid_points = hybrid_n.para['grid_points']
    n = hybrid_n.get_timestep(step)[-1]
    ndata = n_km2cm(n)

    try:
        hybrid_temp = hr(hf, 'temp_tot')
    except NoSuchVariable:
        hybrid_temp = hr(hf, 'temp_p')

    t = hybrid_temp.get_timestep(step)[-1]
    tdata = t_K(t)

    # in Pa not pPa
    thermal_p_data = pressure_pPa(ndata, tdata)*1e-12

    p_therm = profile(points, grid_points, thermal_p_data)

    return p_therm

def magnetic_pressure_profiler(hf, step=-1):
    hybrid_b = hr(hf, 'bt')
    b = hybrid_b.get_timestep(step)[-1]

    # in Pa not pPa
    bdata = b_pressure(b)*1e-12

    grid_points = hybrid_b.para['grid_points']

    p_b = profile(points, grid_points, bdata)

    return p_b

def RH2a_profiler(hf, step=-1):
    """This will make profiles for (rho/P * Ux^2) and plasma beta
    The difference of their sum before and after the shock should
    be zero (at least at the nose).
    """
    rho = mass_density_profiler(hf, step=step)*100**3
    P = thermal_pressure_profiler(hf, step=step)
    ux = vx_profiler(hf, step=step)

    A_profile = rho/P * ux**2
    print A_profile

    b_pressure = magnetic_pressure_profiler(hf, step=step)

    beta_profile = P/b_pressure

    return zip(A_profile, beta_profile)



def plot_profile(ax, profiles, labels=[], mccomas=False):
    x = points[:,0]/1187.
    if mccomas:
        x = -x
    else:
        ax.invert_xaxis()
    lines = ax.plot(x, profiles)
    for line, label in zip(lines, labels):
        line.set_label(label)

def figure_setup(num_subplots, aspect, scale=1):
    #plt.style.use('pluto-paper')
    fig, axs = plt.subplots(nrows=num_subplots, sharex=True, gridspec_kw={'hspace':0}, figsize=[scale*dim for dim in figaspect(num_subplots*aspect)])
    if num_subplots == 1:
        axs = [axs]
    return fig, axs

shell_prefixes = ['/home/nathan/data/pre-2019/2017-Mon-Nov-13/pluto-7/data',
                '/home/nathan/data/pre-2019/2018-Fri-Jan-26/pluto-1/data',
                '/home/nathan/data/pre-2019/2017-Mon-Nov-13/pluto-8/data']
no_shell_prefixes = ['/home/nathan/data/pre-2019/2018-Mon-Jul-09/pluto-1/data',
                    '/home/nathan/data/pre-2019/2018-Thu-Jul-05/pluto-1/data',
                    '/home/nathan/data/pre-2019/2018-Thu-Jul-19/pluto-1/data']
shell_labels = ['IMF: 0.3 nT, with IPUIs', 'IMF: 0.19 nT, with IPUIs', 'IMF: 0.08 nT, with IPUIs'] 
no_shell_labels = ['IMF: 0.3 nT, without IPUIs', 'IMF: 0.19 nT, without IPUIs', 'IMF: 0.08 nT, without IPUIs']
generic_labels = ['IMF: 0.3 nT','IMF: 0.19 nT','IMF: 0.08 nT']

def parse_command_line():
    def limitType(s):
        if s == 'auto':
            return None
        else:
            return float(s)

    class AddSimulationGroup(ExtendConstAction):
        def __init__(self, labels, *args, **kwargs):
            self.new_labels = labels
            super(AddSimulationGroup, self).__init__(*args, **kwargs)


        def __call__(self, parser, namespace, values, option_string=None):
            labels = getattr(namespace, 'line_labels', None)
            labels = copy_items(labels)
            labels.extend(self.new_labels)
            setattr(namespace, 'line_labels', labels)
            super(AddSimulationGroup, self).__call__(parser, namespace, values, option_string)

    parser = argparse.ArgumentParser(description="This tool is for plotting the value of some plasma property (or properties) from along the New Horizons flyby profile.")
    parser.add_argument('--mccomas', action='store_true',
            help='This is used to specify whether to use internal coordinate or McComas coordinates.')
    parser.add_argument('--prefix', default='data',
            help='The folder where data is kept within the --hybrid-folder. Usually the default /data is all that is needed.')
    parser.add_argument('--step', type=int, default=-1,
            help='The timestep number to take from the simulation.')
    parser.add_argument('--xlim', type=limitType, nargs=2, default=['auto','auto'],
            help='xlimits for the profiles')
    parser.add_argument('--ylim', type=limitType, nargs='+',
            help='ylimits for the profiles in the same order they are added')
    parser.add_argument('--save', action='store_true',
            help='Set this flag to save the final plot instead of displaying')
    parser.add_argument('--title', default='Profiles',
            help='Give the plot a descriptive title')
    parser.add_argument('--xlabel', default='X ($R_p$)', help='Label for the X axis')
    parser.add_argument('--ylabels', nargs='+', default=[], help='Labels for the Y axes')

    profile_group = parser.add_argument_group(title="Profiles", 
            description="Select the variables whose profiles you want plotted. One subplot wll be created for each in the order given.\
            If a profiler returns multiple profiles they will go in the same subplot.")
    profile_group.add_argument('--normalized-velocity', dest='profilers', action='append_const', const=vnorm_profiler,
            help='Bulk velocity magnitude normalized by solar wind velocity')
    profile_group.add_argument('--light-ions',          dest='profilers', action='append_const', const=light_profiler,
            help='Number density of hydrogen')
    profile_group.add_argument('--heavy-ions',          dest='profilers', action='append_const', const=heavy_profiler,
            help='Number density of methane')
    profile_group.add_argument('--pressure',            dest='profilers', action='append_const', const=pressure_profiler,
            help='Plot both thermal and magnetic pressure. Line labels don\'t quite work on this one (yet).')

    simulations_group = parser.add_argument_group(title='Simulations', 
            description='Select which simulations to include in each subplot')
    simulations_group.add_argument('--shell', dest='sims', action=AddSimulationGroup, const=shell_prefixes, labels=shell_labels)
    simulations_group.add_argument('--no-shell', dest='sims', action=AddSimulationGroup, const=no_shell_prefixes, labels=no_shell_labels)
    simulations_group.add_argument('--low-no-shell', dest='sims', action=AddSimulationGroup, const=no_shell_prefixes[-1], labels=no_shell_labels[-1])
    simulations_group.add_argument('--low-shell', dest='sims', action=AddSimulationGroup, const=shell_prefixes[-1], labels=shell_labels[-1])
    simulations_group.add_argument('--both-shells', dest='sims', action=AddSimulationGroup, const=shell_prefixes+no_shell_prefixes, labels=shell_labels+no_shell_labels)
    simulations_group.add_argument('--others', dest='sims', action=ExtendAction, nargs='+')
    simulations_group.add_argument('--this', dest='sims', action=ExtendConstAction, const=os.path.join(os.getcwd(),'data'))

    parser.add_argument('--line-labels', nargs='*', help='This overrides the default labels from selecting a simulation group: --shell, --no-shell, or --both-shells')



    args = parser.parse_args()


    return args

if __name__ == '__main__':

    args = parse_command_line()

    if args.sims is None:
        args.sims = shell_prefixes


    fig, axs = figure_setup(len(args.profilers), 1/3.75) 

    for ax, profiler in zip(axs, args.profilers):
        #ax.set_ylabel(y_label)
        for h, line_label in zip(args.sims, args.line_labels):
            pro = profiler(h, step=args.step)
            plot_profile(ax, pro, labels=[line_label], mccomas=args.mccomas)
    axs[-1].set_xlabel(args.xlabel)
    axs[-1].legend()



    if args.save:
        plt.savefig(os.path.join(args.hybrid_folder,args.var + '.png'), bbox_inches='tight')
    else:
        plt.show()


#plt.style.use('pluto-paper')
#main(pressure_profile, labels=('Thermal Pressure','Magnetic Pressure'), xlabel='X ($R_p$)', ylabel='Pressure (pPa)')
#plt.style.use('pluto-paper')
#main(light_profile)
