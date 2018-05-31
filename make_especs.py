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

args = parser.parse_args()

if not args.high and not args.medium and not args.low and not args.other and not args.rehearsal:
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
    high_IMF_espec = get_espec("/home/nathan/data/2017-Mon-Nov-13/pluto-7/data", range(-11,11))
    with open('high_IMF_espec.pickle', 'w') as f:
        cPickle.dump(high_IMF_espec, f)

if args.medium:
    print "Making medium IMF spectrogram"
    medium_IMF_espec = get_espec("/home/nathan/data/2018-Fri-Jan-26/pluto-1/data", range(-11,11))
    with open('medium_IMF_espec.pickle', 'w') as f:
        cPickle.dump(medium_IMF_espec, f)

if args.low:
    print "Making low IMF spectrogram"
    low_IMF_espec  = get_espec("/home/nathan/data/2017-Mon-Nov-13/pluto-8/data", range(-11,11))
    with open('low_IMF_espec.pickle', 'w') as f:
        cPickle.dump(low_IMF_espec, f)

if args.rehearsal:
    print "Making rehearsal spectrogram"
    points, cmats, times  = spice_tools.trajectory(spice_tools.rehearsal_start, spice_tools.rehearsal_end, 30.)
    points[:,0] = np.linspace(300.0*1187.0,-270*1187.0,points.shape[0])
    points[:,1] = 0.0
    points[:,2] = 0.0
    rehearsal_espec = get_espec("/home/nathan/data/2018-Mon-Apr-30/pluto-1/data", [0,1], traj=(points,cmats), radius=1187./4.)
    with open('rehearsal_espec.pickle', 'w') as f:
        cPickle.dump(rehearsal_espec, f)

if args.other is not None:
    other_espec = get_espec(args.other, range(-7,7))
    with open('other_espec.pickle', 'w') as f:
        cPickle.dump(other_espec, f)




