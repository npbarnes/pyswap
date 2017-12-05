#!/usr/bin/env python
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--high', action='store_true', 
        help='Generate the high IMF spectrogram and not the low IMF spectrogram')
parser.add_argument('--low', action='store_true', 
        help='Generate the low IMF spectrogram and not the high IMF spectrogram')

args = parser.parse_args()

if not args.high and not args.low:
    args.high = True
    args.low  = True

import cPickle
import matplotlib.pyplot as plt
from espec import three_colorbars, plot_espec
def plot(filename, title, espec):
    fig, ax = plt.subplots()
    cbar_H, cbar_He, cbar_CH4 = three_colorbars(fig, ax)
    ax.set_title(title)
    plot_espec(fig, ax, cbar_H, cbar_He, cbar_CH4, espec)
    ax.set_xlim([20,-180])

    fig.savefig(filename, bbox_inches='tight')

if args.high:
    with open('high_IMF_espec') as f:
        high_IMF_espec = cPickle.load(f)
    plot('high_IMF_syth_spec.png', "Synthetic SWAP Spectrogram\n0.3nT IMF", high_IMF_espec)
if args.low:
    with open('low_IMF_espec') as f:
        low_IMF_espec = cPickle.load(f)
    plot('low_IMF_synth_spec.png', "Synthetic SWAP Spectrogram\n0.08nT IMF", low_IMF_espec)
