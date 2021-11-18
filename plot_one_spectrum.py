import numpy as np
import cPickle
import matplotlib.pyplot as plt
from swap import bin_centers, bin_edges

with open('rehearsal_espec.pickle') as f:
    spec = cPickle.load(f)

fig, ax = plt.subplots()

step = -20
counts = np.stack([spec['H'][step],spec['He'][step],spec['CH4'][step]])
multi_centers = np.stack([bin_centers,bin_centers,bin_centers])
H = spec['H'][step]
He = spec['He'][step]

bin_widths = bin_edges[1:] - bin_edges[:-1]

ax.hist([bin_centers,bin_centers], bins=bin_edges, weights=[H,He], histtype='barstacked', stacked=True, log=True)
ax.set_xscale('log')

plt.show()

