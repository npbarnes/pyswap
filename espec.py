import numpy as np
import NH_tools
from swap import spectrograms_by_species, bin_centers
from HybridParticleReader import particle_data
from matplotlib import rcParams
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
rcParams['figure.figsize'] = (rcParams['figure.figsize'][0], 2*rcParams['figure.figsize'][1])


print("Reading particle data...")
para, xp, v, mrat, beta, tags = particle_data('/home/nathan/data/pluto-2/databig', n=2)

print("Loading trajectory information...")
points, o = NH_tools.trajectory(NH_tools.close_start, NH_tools.close_end, 60.)

xp = xp[(tags != 0)]
v = v[(tags != 0)]
beta = beta[(tags != 0)]
mrat = mrat[(tags != 0)]

print("Generating spectrograms...")
H, He, CH4 = spectrograms_by_species(xp, v, mrat, beta, points, o, radius=1187., progress=True)

print("Plotting...")
fig, (ax1, ax2) = plt.subplots(2, sharex=True)

# The actual spectrogram to be plotted will be the total response of all species
tot_spec = H + He + CH4

# but it will be colored by which species contributes the most counts to that bin
# so we need masks for where each species dominates.
mH = np.ma.masked_array(tot_spec, mask=(~((H>He) & (H>CH4))))
mHe = np.ma.masked_array(tot_spec, mask=(~((He>H) & (He>CH4))))
mCH4 = np.ma.masked_array(tot_spec, mask=(~((CH4>H) & (CH4>He))))

# Plot the masked arrays
Hhist = ax1.pcolormesh(points[:,0]/1187., bin_centers, mH.T, norm=LogNorm(), cmap='Blues',
        vmin=2e-3, vmax=2e4)

Hehist = ax1.pcolormesh(points[:,0]/1187., bin_centers, mHe.T, norm=LogNorm(), cmap='Greens',
        vmin=2e-3, vmax=2e4)

CH4hist = ax1.pcolormesh(points[:,0]/1187., bin_centers, mCH4.T, norm=LogNorm(), cmap='Reds',
        vmin=2e-3, vmax=2e4)

ax1.invert_xaxis()
ax1.set_yscale('log')

ax2.plot(points[:,0]/1187., o)

plt.show()

