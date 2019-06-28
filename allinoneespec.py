#!/usr/bin/env python
import os
import matplotlib.pyplot as plt
from matplotlib.figure import figaspect
from espec import get_espec, three_colorbars, plot_espec
import argparse

def imftype(s):
    if s is None:
        return ''
    else:
        return s + 'nT IMF'

parser = argparse.ArgumentParser()
parser.add_argument('--mccomas', action='store_true')
parser.add_argument('--hybrid-folder', dest='hybrid_folder', default=None)
parser.add_argument('--prefix', default='data')
parser.add_argument('--imf', type=imftype)

args = parser.parse_args()

if args.hybrid_folder is None:
    args.data_folder = os.path.join(os.getcwd(), args.prefix)
else:
    args.data_folder = os.path.join(args.hybrid_folder, args.prefix)

def plot(filename, title, espec, fontsize=15):
    fig, ax = plt.subplots(figsize=figaspect(0.5))
    cbar_H, cbar_He, cbar_CH4 = three_colorbars(fig, ax)
    plot_espec(fig, ax, cbar_H, cbar_He, cbar_CH4, espec, mccomas=args.mccomas)
    ax.set_xlim([-20,150])

    ax.set_title(title)
    ax.set_xlabel('X ($R_p$)')
    ax.set_ylabel('Energy/Q (eV/q)')

    fig.savefig(filename, bbox_inches='tight')


plt.style.use('pluto-paper')
other_espec = get_espec(args.data_folder, range(-7,7))
plot('synth_spec.png', "Synthetic SWAP Spectrogram{}".format('\n'+args.imf), other_espec)
