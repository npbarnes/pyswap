#!/usr/bin/env python
import numpy as np
import spice_tools
from swap import spectrograms_by_species, bin_centers, bin_edges, is_sunview
from HybridReader2 import HybridReader2 as hr
import HybridHelper as hh
from HybridParticleReader import particle_data
from matplotlib import rcParams
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

def get_espec(prefix, n, traj=None, progress=False, times=None, radius=1187.):
    if traj is None:
        points, cmats, times = spice_tools.trajectory(spice_tools.flyby_start, spice_tools.flyby_end, 30.)
    else:
        points, cmats = traj

    para, xp, v, mrat, beta, tags = particle_data(prefix, n)
    xp = xp[(tags != 0)]
    v = v[(tags != 0)]
    beta = beta[(tags != 0)]
    mrat = mrat[(tags != 0)]

    #H, He, CH4 = spectrograms_by_species(xp, v, mrat, beta, points, cmats, radius=0.5*1187., progress=progress)
    H, He, CH4 = spectrograms_by_species(xp, v, mrat, beta, points, cmats, radius=radius, progress=progress)


    return {'H':H, 'He':He, 'CH4':CH4, 'trajectory':points, 'orientation':cmats, 'times':times}

def plot_onespec(fig, ax, espec):
    H, He, CH4 = espec['H'], espec['He'], espec['CH4']
    # The actual spectrogram to be plotted will be the total response of all species
    tot_spec = H + He + CH4


    axis = espec['trajectory'][:,0]/1187.
    ax.pcolormesh(range(len(tot_spec)), bin_centers, tot_spec.T, norm=LogNorm(),
            vmin=2e-3, vmax=2e4)
    ax.set_yscale('log')

    plt.figure()
    bin_widths = bin_edges[1:] - bin_edges[:-1]
    plt.hist(bin_centers, bins=bin_edges, weights=tot_spec[100,:], log=True)
    plt.gca().set_xscale('log')

    plt.figure()
    plt.plot(bin_centers, tot_spec[100,:])
    plt.gca().set_xscale('log')
    plt.gca().set_yscale('log')

    plt.show()

    
def plot_espec(fig, ax, cbar_H, cbar_He, cbar_CH4, espec, mccomas=False, timeaxis=False):

    H, He, CH4 = espec['H'], espec['He'], espec['CH4']
    # The actual spectrogram to be plotted will be the total response of all species
    tot_spec = H + He + CH4

    # but it will be colored by which species contributes the most counts to that bin
    # so we need masks for where each species dominates.
    mH = np.ma.masked_array(tot_spec, mask=(~((H>He) & (H>CH4))))
    mHe = np.ma.masked_array(tot_spec, mask=(~((He>H) & (He>CH4))))
    mCH4 = np.ma.masked_array(tot_spec, mask=(~((CH4>H) & (CH4>He))))
#    print("!!!!!!!!!!!!!!!!!! Changing masks to any CH4 !!!!!!!!!!!!!!!!!!!!!")
#    mH = np.ma.masked_array(tot_spec, mask=((He >= H) | (CH4 !=0)))
#    mHe = np.ma.masked_array(tot_spec, mask=((H > He) | (CH4 !=0)))
#    mCH4 = np.ma.masked_array(tot_spec, mask=(CH4 == 0))

    if timeaxis:
        axis = espec['times']
    else:
        axis = espec['trajectory'][:,0]/1187.

    if mccomas and not timeaxis:
        axis = -axis

    # Plot the masked arrays
    Hhist = ax.pcolormesh(axis, bin_centers, mH.T, norm=LogNorm(), cmap='Blues',
            vmin=2e-3, vmax=2e4)

    Hehist = ax.pcolormesh(axis, bin_centers, mHe.T, norm=LogNorm(), cmap='Greens',
            vmin=2e-3, vmax=2e4)

    CH4hist = ax.pcolormesh(axis, bin_centers, mCH4.T, norm=LogNorm(), cmap='Reds',
            vmin=2e-3, vmax=2e4)

    Hcb = fig.colorbar(Hhist, cax=cbar_H)
    Hecb = fig.colorbar(Hehist, cax=cbar_He, format="")
    CH4cb = fig.colorbar(CH4hist, cax=cbar_CH4, format="")

    # add a title
    Hecb.ax.set_title('SCEM (Hz)', fontdict={'fontsize':'small'})

    if not mccomas:
        ax.invert_xaxis()

    ax.set_yscale('log')

def plot_traj_context(fig, ax_xy, ax_xz, prefix, traj, size, mccomas=False):
    hybrid_np_CH4 = hr(prefix, 'np_CH4')
    np_3d = hybrid_np_CH4.get_timestep(-1)[-1]
    hh.direct_plot(fig, ax_xy, np_3d, hybrid_np_CH4.para, 'xy', norm=LogNorm(), fontsize=size, mccomas=mccomas)
    hh.direct_plot(fig, ax_xz, np_3d, hybrid_np_CH4.para, 'xz', norm=LogNorm(), fontsize=size, mccomas=mccomas)

    traj = traj/1187.

    if mccomas:
        traj[:,0] = -traj[:,0]
        traj[:,1] = -traj[:,1]

    x = traj[:, 0]
    y = traj[:, 1]
    z = traj[:, 2]
    ax_xy.plot(x, y, color='black', linewidth=2, scalex=False, scaley=False)
    ax_xz.plot(x, z, color='black', linewidth=2, scalex=False, scaley=False)

def plot_one_traj_context(fig, ax_xy, prefix, traj, size, mccomas=False):
    hybrid_np_CH4 = hr(prefix, 'np_CH4')
    np_3d = hybrid_np_CH4.get_timestep(-1)[-1]
    hh.direct_plot(fig, ax_xy, np_3d, hybrid_np_CH4.para, 'xy', norm=LogNorm(), fontsize=size, mccomas=mccomas)

    traj = traj/1187.

    if mccomas:
        traj[:,0] = -traj[:,0]
        traj[:,1] = -traj[:,1]

    x = traj[:, 0]
    y = traj[:, 1]
    ax_xy.plot(x, y, color='black', linewidth=2, scalex=False, scaley=False)

def three_colorbars(fig, ax):
    fig.subplots_adjust(right=0.8, hspace=0.05)

    # Setup colorbar axes
    spec_pos = ax.get_position()
    cbar_CH4 = fig.add_axes([spec_pos.x1+.01, spec_pos.y0+(1./6.)*spec_pos.height, 0.1/3, (2./3.)*spec_pos.height])

    cbar_CH4_pos = cbar_CH4.get_position()
    cbar_He = fig.add_axes([cbar_CH4_pos.x1, spec_pos.y0+(1./6.)*spec_pos.height, 0.1/3, (2./3.)*spec_pos.height])

    cbar_He_pos = cbar_He.get_position()
    cbar_H = fig.add_axes([cbar_He_pos.x1, spec_pos.y0+(1./6.)*spec_pos.height, 0.1/3, (2./3.)*spec_pos.height])

    return cbar_H, cbar_He, cbar_CH4

if __name__ == '__main__': # just xy traj context
    fontsize = 20

    fig, ax1 = plt.subplots()

    ax1.set_aspect('equal')

    points, o = NH_tools.trajectory(NH_tools.flyby_start, NH_tools.flyby_end, 60.)
    plot_one_traj_context(fig, ax1, "/home/nathan/data/2017-Mon-Nov-13/pluto-8/data", points, fontsize, mccomas=True)
    ax1.set_xlim([-30, ax1.get_xlim()[1]])
    ax1.set_ylim([-50,50])

    ax1.set_title("Heavy Ion Density", fontsize=1.5*fontsize)

    plt.tight_layout(rect=[0,0.03,1,0.9])
    #plt.tight_layout()
    plt.show()

if __name__ == '__pain__': # xy, xz traj context
    fontsize = 20

    fig, (ax1,ax2) = plt.subplots(ncols=2, sharex=True, figsize=(2.2*rcParams['figure.figsize'][0], rcParams['figure.figsize'][1]))

    ax1.set_aspect('equal')
    ax2.set_aspect('equal')

    points, o = NH_tools.trajectory(NH_tools.flyby_start, NH_tools.flyby_end, 60.)
    plot_traj_context(fig, ax1, ax2, "/home/nathan/data/2017-Mon-Nov-13/pluto-8/data", points, fontsize, mccomas=True)
    ax1.set_xlim([-30, ax1.get_xlim()[1]])

    fig.suptitle("Heavy Ion Density", fontsize=1.5*fontsize)

    plt.tight_layout(rect=[0,0.03,1,0.9])
    plt.show()

if __name__ == '__pain__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('prefix', help='Path to the hybrid data folder')
    parser.add_argument('-n', type=int, default=2, help='Number of z direction slices to use')

    args = parser.parse_args()

    fig, (ax, ax2) = plt.subplots(2)
    cbar_H, cbar_He, cbar_CH4 = three_colorbars(fig, ax)
    ax.set_title('Synthetic SWAP Spectrogram\n0.3nT IMF')

    plot_espec(fig, ax, cbar_H, cbar_He, cbar_CH4, args.prefix, args.n)
    points, o = NH_tools.trajectory(NH_tools.close_start, NH_tools.close_end, 60.)

    ax2.plot(points[:,0], o)


    plt.show()


if __name__ == '__pain__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('prefix', help='Path to the hybrid data folder')
    parser.add_argument('-n', type=int, default=2, help='Number of z direction slices to use')

    args = parser.parse_args()

    fig, ax = plt.subplots()
    cbar_H, cbar_He, cbar_CH4 = three_colorbars(fig, ax)
    ax.set_title('Synthetic SWAP Spectrogram\n0.3nT IMF')

    plot_espec(fig, ax, cbar_H, cbar_He, cbar_CH4, args.prefix, args.n)


    plt.show()

