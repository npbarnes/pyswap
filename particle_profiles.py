import numpy as np
from scipy.spatial.distance import cdist
import NH_tools
from HybridParticleReader import particle_data

import matplotlib.pyplot as plt
#from matplotlib import rcParams
#from matplotlib.colors import LogNorm

def profile(f):
    def wrapped(local, *args, **kwargs):
        ret = np.empty(len(local))

        for i, l in enumerate(local):
            if np.any(l):
                ret[i] = f(l, *args, **kwargs)
            else:
                ret[i] = 0.

        return ret

    return wrapped

@profile
def bulk_velocity(l, v, beta):
    ave_velocity = np.average(v[l,:], weights=1./beta[l], axis=0)
    return np.linalg.norm(ave_velocity)

@profile
def density(l, beta, radius=1187.):
    dV = 4./3. * np.pi * radius**3
    return np.sum(1./beta[l])/dV

@profile
def temperature(l, v, mrat, beta



print("Reading particle data...")
para, xp, v, mrat, beta, tags = particle_data('/home/nathan/data/pluto-2/databig', n=2)
xp = xp[(tags != 0)]
v = v[(tags != 0)]
beta = beta[(tags != 0)]
mrat = mrat[(tags != 0)]

print("Loading trajectory information...")
points, o = NH_tools.trajectory(NH_tools.close_start, NH_tools.close_end, 60.)

radius = 1187.

print("Finding local particles...")
local = cdist(points, xp) < radius

profile = bulk_velocity(local, v, beta)

plt.plot(points[:,0]/1187., profile)

plt.show()



