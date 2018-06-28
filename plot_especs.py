#!/usr/bin/env python
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--high', action='store_true', 
        help='Generate the high IMF spectrogram and not the others')
parser.add_argument('--medium', action='store_true', 
        help='Generate the medium IMF spectrogram and not the others')
parser.add_argument('--low', action='store_true', 
        help='Generate the low IMF spectrogram and not the others')
parser.add_argument('--rehearsal', action='store_true', 
        help='Generate the rehearsal spectrogram and not the others')
parser.add_argument('--other', action='store_true')
parser.add_argument('--show', action='store_true')

args = parser.parse_args()

if not args.high and not args.medium and not args.low and not args.other and not args.rehearsal:
    args.high = True
    args.medium  = True
    args.low  = True
    args.rehearsal = True

import cPickle
from matplotlib import rcParams
import matplotlib.pyplot as plt
from matplotlib.dates import HourLocator, MinuteLocator, DateFormatter
from matplotlib.figure import figaspect
from matplotlib.ticker import MultipleLocator
from espec import N_colorbars, plot_espec, plot_onespec
import spice_tools

def one_plot(filename, title, espec, fontsize=15):
    fig, ax = plt.subplots()
    plot_onespec(fig, ax, espec)

def plot(filename, title, espec, rehearsal=False):
    plt.style.use('pluto-paper')
    rcParams['figure.autolayout'] = False # autolayout (default True in pluto-paper style) breaks these plots

    fig, ax = plt.subplots(figsize=figaspect(0.3))
    if rehearsal:
        cbar_CH4 = None
        cbar_He, cbar_H = N_colorbars(fig, ax, 2, fraction=0.1)
    else:
        cbar_CH4, cbar_He, cbar_H = N_colorbars(fig, ax, 3, fraction=0.1)

    plot_espec(fig, ax, cbar_H, cbar_He, cbar_CH4, espec, mccomas=True, rehearsal=rehearsal)

    ax.set_ylabel('Energy/Q (eV/q)')

    if rehearsal:
        ax.set_title(title)
        ax.set_xlabel('Time (UTC)')
        ax.xaxis.set_major_locator(HourLocator())
        ax.xaxis.set_major_formatter(DateFormatter('%H:%M'))
        ax.xaxis.set_minor_locator(MinuteLocator(byminute=range(0,60,10)))

    else:
        ax.set_title(title, pad=50)
        ax.set_xlabel('Time (UTC)')
        ax.set_xlabel('X ($R_p$)')
        ax.xaxis.set_major_locator(MultipleLocator(50))
        ax.xaxis.set_minor_locator(MultipleLocator(10))
        ax.set_xlim([-20,105])

        time_axis = ax.twiny()
        time_axis.set_xlim(spice_tools.et2pydatetime(espec['times'][0]), 
                           spice_tools.et2pydatetime(espec['times'][-1]))
        time_axis.set_xlabel('Time (UTC)')
        time_axis.xaxis.set_major_locator(HourLocator())
        time_axis.xaxis.set_major_formatter(DateFormatter('%H:%M'))
        time_axis.xaxis.set_minor_locator(MinuteLocator(byminute=range(0,60,10)))

    if args.show:
        plt.show()
    else:
        fig.savefig(filename, bbox_inches='tight')

if args.high:
    with open('high_IMF_espec.pickle') as f:
        high_IMF_espec = cPickle.load(f)
    plot('high_IMF_synth_spec.png', "Synthetic SWAP Spectrogram\n0.3nT IMF", high_IMF_espec)

if args.medium:
    with open('medium_IMF_espec.pickle') as f:
        medium_IMF_espec = cPickle.load(f)
    plot('medium_IMF_synth_spec.png', "Synthetic SWAP Spectrogram\n0.19nT IMF", medium_IMF_espec)

if args.low:
    with open('low_IMF_espec.pickle') as f:
        low_IMF_espec = cPickle.load(f)
    plot('low_IMF_synth_spec.png', "Synthetic SWAP Spectrogram\n0.08nT IMF", low_IMF_espec)

if args.rehearsal:
    with open('rehearsal_espec.pickle') as f:
        rehearsal_espec = cPickle.load(f)
    plot('rehearsal_synth_spec.png', "Synthetic SWAP Rehearsal Spectrogram", rehearsal_espec, rehearsal=True)

if args.other:
    with open('other_espec.pickle') as f:
        other_espec = cPickle.load(f)
    one_plot('synth_spec.png', "Synthetic SWAP Spectrogram", other_espec)
