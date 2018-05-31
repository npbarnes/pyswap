import numpy as np
from scipy import spatial
import spice_tools
from HybridParticleReader import LastStep

import matplotlib.pyplot as plt

def profile(f):
    def wrapped(traj, particles, radius=1187.):
        kdparts  = spatial.cKDTree(particles.x)
        kdpoints = spatial.cKDTree(traj)
        local = kdpoints.query_ball_tree(kdparts, radius)

        first = f(particles.filter(local[0]), radius=radius)
        ret = np.zeros((len(local),)+first.shape)
        ret[0] = first

        for i, l in enumerate(local[1:]):
            # i+1 since we already did the first entry when getting the dimensions for ret
            ret[i+1] = f(particles.filter(l), radius=radius) 

        return ret

    return wrapped

@profile
def bulk_velocity(p, radius):
    return np.average(p.v, weights=1./p.beta, axis=0)

@profile
def density(p, radius):
    dV = 4./3. * np.pi * radius**3
    return 1e-15*np.sum(1./p.beta)/dV

if __name__ == '__main__':
    # No Shell
    #hybrid_folder = '/home/nathan/data/2018-Thu-Jan-25/pluto-2'
    # large box
    hybrid_folder = '/home/nathan/data/2017-Mon-Nov-13/pluto-8'
    data_folder = 'data'
    n=7
    p = LastStep(hybrid_folder, n, data_folder=data_folder)
    #p.filter(p.tags != 0, out=p)

    traj, cmats, times = spice_tools.trajectory(spice_tools.flyby_start, spice_tools.flyby_end, 60.)

    d = density(traj, p, radius=1187./2.)

    plt.plot(traj[:,0]/1187., d)

    plt.gca().invert_xaxis()
    plt.gca().set_xlim([-10,-100])

    plt.show()
