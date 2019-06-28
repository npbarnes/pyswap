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

def get_espec(prefix, n, traj=None, progress=False, times=None):
    if traj is None:
        points, cmats, times = spice_tools.trajectory(spice_tools.flyby_start, spice_tools.flyby_end, 30.)
    else:
        points, cmats = traj

    para, xp, v, mrat, beta, tags = particle_data(prefix, n)
    print "Number of Dummies: {}".format(len(xp[tags==0]))
    volume = para['dx']**3
    xp = xp[(tags != 0)]
    v = v[(tags != 0)]
    beta = beta[(tags != 0)]
    mrat = mrat[(tags != 0)]

    #H, He, CH4 = spectrograms_by_species(xp, v, mrat, beta, points, cmats, radius=0.5*1187., progress=progress)
    H, He, CH4 = spectrograms_by_species(xp, v, mrat, beta, points, cmats, volume=volume, progress=progress)


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

def one_cmap_espec_pcolormesh(ax, X, espec):
    H, He, CH4 = espec['H'], espec['He'], espec['CH4']
    tot_spec = H + He + CH4

    mappable = ax.pcolormesh(X, bin_centers, tot_spec.T, norm=LogNorm(),
            vmin=1, vmax=3e4)

    ax.format_coord = hh.build_format_coord(X, bin_centers, tot_spec)

    return mappable

def color_coded_espec_pcolormesh(ax, X, espec):
    """ Plot the spectrogram without messing with labels, colorbars etc.
    arguments:
        fig, and ax are the matplotlib figure and axes objects. 
        X gets passed directly into pcolormesh as the points on the X axis
        espec is the object made by the get_espec function

    returns:
        a tuple of the three QuadMesh objects returned by the three calls to pcolormesh,
        one for each species. The norm, cmap, vmin, and vmax all have default 
        values, but these may be changed after the fact with the respective set_
        functions of the QuadMesh objects.
    """
    H, He, CH4 = espec['H'], espec['He'], espec['CH4']

    # The actual spectrogram to be plotted will be the total response of all species
    tot_spec = H + He + CH4

    # but it will be colored by which species contributes the most counts to that bin
    # so we need masks for where each species dominates.
    mH = np.ma.masked_array(tot_spec, mask=(~((H>He) & (H>CH4))))
    mHe = np.ma.masked_array(tot_spec, mask=(~((He>H) & (He>CH4))))
    mCH4 = np.ma.masked_array(tot_spec, mask=(~((CH4>H) & (CH4>He))))

    Hhist = ax.pcolormesh(X, bin_centers, mH.T, norm=LogNorm(), cmap='Blues',
            vmin=1, vmax=3e4)

    Hehist = ax.pcolormesh(X, bin_centers, mHe.T, norm=LogNorm(), cmap='Greens',
            vmin=1, vmax=3e4)

    CH4hist = ax.pcolormesh(X, bin_centers, mCH4.T, norm=LogNorm(), cmap='Reds',
            vmin=1, vmax=3e4)

    ax.format_coord = hh.build_format_coord(X, bin_centers, tot_spec)

    return Hhist, Hehist, CH4hist

def plot_espec(fig, ax, cbar_H, cbar_He, cbar_CH4, espec, mccomas=False, rehearsal=False):
    if rehearsal: # Use time axis instead of position
        X = [spice_tools.et2pydatetime(et) for et in espec['times']]
    else:
        X = espec['trajectory'][:,0]/1187.

    if mccomas and not rehearsal:
        X = -X

    Hhist, Hehist, CH4hist = color_coded_espec_pcolormesh(ax, X, espec)

    Hcb = fig.colorbar(Hhist, cax=cbar_H)
    Hecb = fig.colorbar(Hehist, cax=cbar_He, format="")
    if not rehearsal: # There are no heavies in the rehearsal
        CH4cb = fig.colorbar(CH4hist, cax=cbar_CH4, format="")

    # add a colorbar title
    cb_pos = Hecb.ax.get_position()
    if rehearsal:
        center = cb_pos.xmax
    else:
        center = (cb_pos.xmin + cb_pos.xmax)/2
    fig.text(center+0.005, cb_pos.ymax+0.01, "SCEM (Hz)", horizontalalignment='center', fontsize=15)

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

def three_colorbars(fig, ax, fraction=0.2, pad=0.01):
    return N_colorbars(fig, ax, 3, fraction, pad)

def N_colorbars(fig, ax, N, fraction=0.2, pad=0.01):
    fig.subplots_adjust(right=1-fraction)

    spec_pos = ax.get_position()
    first_cbar = fig.add_axes([spec_pos.xmax+pad, spec_pos.ymin+(1./6.)*spec_pos.height, fraction/2/N, (2./3.)*spec_pos.height])

    cbars = [first_cbar]
    for i in range(1,N):
        prev_pos = cbars[i-1].get_position()
        cbars.append(fig.add_axes([prev_pos.xmax, spec_pos.ymin+(1./6.)*spec_pos.height, fraction/2/N, (2./3.)*spec_pos.height]))

    return cbars


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

