#!/usr/bin/env python
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--high', action='store_true', 
        help='Generate the high IMF spectrogram.')
parser.add_argument('--medium', action='store_true', 
        help='Generate the medium IMF spectrogram.')
parser.add_argument('--low', action='store_true', 
        help='Generate the low IMF spectrogram.')
parser.add_argument('--rehearsal', action='store_true', 
        help='Generate the rehearsal spectrogram.')
parser.add_argument('--other', default=None)
parser.add_argument('--onespec', action='store_true')

args = parser.parse_args()

if not any([args.high, args.medium, args.low, args.rehearsal, args.other, args.onespec]):
    args.high = True
    args.medium  = True
    args.low  = True
    args.rehearsal = True

import spice_tools
from espec import get_espec
import cPickle
import numpy as np

if args.high:
    print "Making high IMF spectrogram"
    high_IMF_espec = get_espec("/home/nathan/data/2017-Mon-Nov-13/pluto-7/data", range(-11,11), progress=True)
    with open('high_IMF_espec.pickle', 'w') as f:
        cPickle.dump(high_IMF_espec, f)

if args.medium:
    print "Making medium IMF spectrogram"
    medium_IMF_espec = get_espec("/home/nathan/data/2018-Fri-Jan-26/pluto-1/data", range(-11,11), progress=True)
    with open('medium_IMF_espec.pickle', 'w') as f:
        cPickle.dump(medium_IMF_espec, f)

if args.low:
    print "Making low IMF spectrogram"
    low_IMF_espec  = get_espec("/home/nathan/data/2017-Mon-Nov-13/pluto-8/data", range(-11,11), progress=True)
    with open('low_IMF_espec.pickle', 'w') as f:
        cPickle.dump(low_IMF_espec, f)

if args.rehearsal:
    print "Making rehearsal spectrogram"
    points, cmats, times  = spice_tools.trajectory(spice_tools.rehearsal_start, spice_tools.rehearsal_end, 30.)
    points[:,0] = np.linspace(440.0*1187.0,-440*1187.0,points.shape[0])
    points[:,1] = 0.0
    points[:,2] = 0.0
    #rehearsal_espec = get_espec("/home/nathan/data/2018-Mon-Apr-30/pluto-1/data", [0,1], traj=(points,cmats), times=times, progress=True)
    rehearsal_espec = get_espec("/home/nathan/data/2018-Wed-Aug-15/pluto-4/data", [-1,0,1], traj=(points,cmats), times=times, progress=True)
    with open('rehearsal_espec.pickle', 'w') as f:
        cPickle.dump(rehearsal_espec, f)

if args.onespec:
    print "Making one spectrum"
    points = np.array([[0,0,0]], dtype=np.float64) # but actually only one point
    cmats = np.array([[[0,1,0],[1,0,0],[0,0,1]]], dtype=np.float64) # but actually only one cmat
    onespec = get_espec("/home/nathan/data/2018-Mon-Apr-30/pluto-1/data", [0,1], traj=(points,cmats), times=None, progress=True)
    with open('onespec.pickle', 'w') as f:
        cPickle.dump(onespec, f)

if args.other is not None:
    other_espec = get_espec(args.other, range(-7,7), progress=True)
    with open('other_espec.pickle', 'w') as f:
        cPickle.dump(other_espec, f)
