#!/usr/bin/env python
import numpy as np
import NH_tools
from swap import spectrograms_by_species, bin_centers, is_sunview
from HybridParticleReader import particle_data
from matplotlib import rcParams
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

def plot_espec(fig, ax, cbar_H, cbar_He, cbar_CH4, prefix, n):

    print("Reading particle data...")
    para, xp, v, mrat, beta, tags = particle_data(prefix, range(-7,10))

    xp = xp[(tags != 0)]
    v = v[(tags != 0)]
    beta = beta[(tags != 0)]
    mrat = mrat[(tags != 0)]

    print("Loading trajectory information...")
    points, o = NH_tools.trajectory(NH_tools.close_start, NH_tools.close_end, 60.)


    print("Generating spectrograms...")
    H, He, CH4 = spectrograms_by_species(xp, v, mrat, beta, points, o, radius=1187., progress=True)

    # The actual spectrogram to be plotted will be the total response of all species
    tot_spec = H + He + CH4

    # but it will be colored by which species contributes the most counts to that bin
    # so we need masks for where each species dominates.
    mH = np.ma.masked_array(tot_spec, mask=(~((H>He) & (H>CH4))))
    mHe = np.ma.masked_array(tot_spec, mask=(~((He>H) & (He>CH4))))
    mCH4 = np.ma.masked_array(tot_spec, mask=(~((CH4>H) & (CH4>He))))

    # Plot the masked arrays
    Hhist = ax.pcolormesh(points[:,0]/1187., bin_centers, mH.T, norm=LogNorm(), cmap='Blues',
            vmin=2e-3, vmax=2e4)

    Hehist = ax.pcolormesh(points[:,0]/1187., bin_centers, mHe.T, norm=LogNorm(), cmap='Greens',
            vmin=2e-3, vmax=2e4)

    CH4hist = ax.pcolormesh(points[:,0]/1187., bin_centers, mCH4.T, norm=LogNorm(), cmap='Reds',
            vmin=2e-3, vmax=2e4)

    Hcb = fig.colorbar(Hhist, cax=cbar_H)
    Hecb = fig.colorbar(Hehist, cax=cbar_He, format="")
    CH4cb = fig.colorbar(CH4hist, cax=cbar_CH4, format="")

    # add a title
    Hecb.ax.set_title('SCEM (Hz)', fontdict={'fontsize':'small'})

    ax.invert_xaxis()
    ax.set_yscale('log')

def three_colorbars(fig, ax):
    fig.subplots_adjust(right=0.8, hspace=0.05)

    # Setup colorbar axes
    spec_pos = ax.get_position()
    cbar_CH4 = fig.add_axes([spec_pos.x1+.01, spec_pos.y0+.01, 0.1/3, spec_pos.height-.02])

    cbar_CH4_pos = cbar_CH4.get_position()
    cbar_He = fig.add_axes([cbar_CH4_pos.x1, spec_pos.y0+.01, 0.1/3, spec_pos.height-.02])

    cbar_He_pos = cbar_He.get_position()
    cbar_H = fig.add_axes([cbar_He_pos.x1, spec_pos.y0+.01, 0.1/3,spec_pos.height-.02])

    return cbar_H, cbar_He, cbar_CH4


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('prefix', help='Path to the hybrid data folder')
    parser.add_argument('-n', type=int, default=2, help='Number of z direction slices to use')

    args = parser.parse_args()

    fig, ax = plt.subplots()
    cbar_H, cbar_He, cbar_CH4 = three_colorbars(fig, ax)

    plot_espec(fig, ax, cbar_H, cbar_He, cbar_CH4, args.prefix, args.n)

    plt.show()

