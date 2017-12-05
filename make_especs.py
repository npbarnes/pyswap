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

import NH_tools
from espec import get_espec
import cPickle

points, o = NH_tools.trajectory(NH_tools.flyby_start, NH_tools.flyby_end, 30.)

if args.high:
    high_IMF_espec = get_espec("/home/nathan/data/2017-Mon-Nov-13/pluto-7/data", range(-11,11), traj=(points, o))
    with open('high_IMF_espec', 'w') as f:
        cPickle.dump(high_IMF_espec, f)
if args.low:
    low_IMF_espec  = get_espec("/home/nathan/data/2017-Mon-Nov-13/pluto-8/data", range(-11,11), traj=(points, o))
    with open('low_IMF_espec', 'w') as f:
        cPickle.dump(low_IMF_espec, f)



