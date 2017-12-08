#!/usr/bin/env python
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--high', action='store_true', 
        help='Generate the high IMF spectrogram and not the low IMF spectrogram')
parser.add_argument('--low', action='store_true', 
        help='Generate the low IMF spectrogram and not the high IMF spectrogram')
parser.add_argument('--two', action='store_true')

args = parser.parse_args()

if not args.high and not args.low and not args.two:
    args.high = True
    args.low  = True
    args.two  = True

import cPickle
from matplotlib import rcParams
import matplotlib.pyplot as plt
from espec import three_colorbars, plot_espec

def plot(filename, title, espec, fontsize=15):
    fig, ax = plt.subplots(figsize=(rcParams['figure.figsize'][0], 0.7*rcParams['figure.figsize'][1]))
    cbar_H, cbar_He, cbar_CH4 = three_colorbars(fig, ax)
    plot_espec(fig, ax, cbar_H, cbar_He, cbar_CH4, espec, mccomas=True)
    ax.set_xlim([-20,105])

    ax.set_title(title, fontsize=1.5*fontsize)
    ax.set_xlabel('X ($R_p$)', fontsize=fontsize)
    ax.set_ylabel('Energy/Q (eV/q)', fontsize=fontsize)
    ax.tick_params(axis='both', which='major', labelsize=0.7*fontsize)

    fig.savefig(filename, bbox_inches='tight')

if args.high:
    with open('high_IMF_espec.pickle') as f:
        high_IMF_espec = cPickle.load(f)
    plot('high_IMF_synth_spec.png', "Synthetic SWAP Spectrogram\n0.3nT IMF", high_IMF_espec)

if args.low:
    with open('low_IMF_espec.pickle') as f:
        low_IMF_espec = cPickle.load(f)
    plot('low_IMF_synth_spec.png', "Synthetic SWAP Spectrogram\n0.08nT IMF", low_IMF_espec)

if args.two:
    with open('pluto-2-espec.pickle') as f:
        low_IMF_espec = cPickle.load(f)
    plot('synth_spec.png', "Synthetic SWAP Spectrogram\n0.08nT IMF\n 'Pluto-2'", low_IMF_espec)
