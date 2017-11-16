from swap import *
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

para, xp, v, mrat, beta, tags = open_stuff('/home/nathan/data/pluto-2/databig')
v = v.astype(np.float64)

#points, o = trajectory(NH_tools.close_start, NH_tools.close_end, 600.)

num = 1
points = np.zeros((num,3),dtype='d')
points[:,0] = 25.*1187. #np.linspace(20,-100,num)*1187.
points[:,2] = np.max(para['qz'])/2

o = np.zeros((num,3),dtype='d')
o[:,2] = 90.

xp = xp[tags != 0]
v = v[tags != 0]
mrat = mrat[tags != 0]
beta = beta[tags != 0]

sp = spectrogram(xp, v, mrat, beta, points, o, radius=2*1187.)

spec = sp[0,:]
center = (bin_edges[1:] + bin_edges[:-1])/2
width = bin_edges[1:] - bin_edges[:-1]
plt.bar(center, spec, width=width)
plt.gca().set_xscale('log')
plt.gca().set_yscale('log')
plt.xlim([100,10000])
plt.title('Synthetic SWAP solar wind spectrum')
plt.xlabel('Energy per charge (eV/q)')
plt.ylabel('SCEM counts (Hz)')
plt.show()
