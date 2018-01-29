import numpy as np
from HybridReader2 import HybridReader2 as hr
from HybridHelper import parser, parse_cmd_line, direct_plot, beta_plot
import matplotlib.pyplot as plt
from matplotlib import colors, rcParams

def beta_data(prefix):
    hn = hr(prefix, 'np')
    para = hn.para
    n = hn.get_timestep(-1)[-1]
    T = hr(prefix, 'temp_p').get_timestep(-1)[-1]
    B = hr(prefix, 'bt').get_timestep(-1)[-1]

    # Convert units
    n = n/(1000.0**3)                    # 1/km^3 -> 1/m^3
    T = 1.60218e-19 * T                  # eV -> J
    B = 1.6726219e-27/1.60217662e-19 * B # proton gyrofrequency -> T

    # Compute B \cdot B
    B2 = np.sum(B**2, axis=-1)

    # Compute plasma beta
    data = n*T/(B2/(2*1.257e-6))

    return para, data



fig, (ax_high, ax_low) = plt.subplots(ncols=2, sharex=True, figsize=(2*rcParams['figure.figsize'][0], 0.8*rcParams['figure.figsize'][1]))

ax_low.set_aspect('equal')
ax_high.set_aspect('equal')

low_para, low_data  = beta_data('/home/nathan/data/2017-Mon-Nov-13/pluto-8/data')
high_para, high_data = beta_data('/home/nathan/data/2017-Mon-Nov-13/pluto-7/data')
beta_plot(fig, ax_low, low_data, low_para, 'xy', fontsize=20, mccomas=True)
beta_plot(fig, ax_high, high_data, high_para, 'xy', fontsize=20, mccomas=True)

fig.suptitle("Plasma Beta", fontsize=1.5*20)
ax_low.set_title("IMF: 0.08nT", fontsize=0.7*20)
ax_high.set_title("IMF: 0.3nT", fontsize=0.7*20)

ax_low.set_xlim([-25, 105])

plt.tight_layout(rect=[0,0.03,1,0.9])
plt.show()

