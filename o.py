import numpy as np
import NH_tools
import matplotlib.pyplot as plt
points, o = NH_tools.trajectory(NH_tools.close_start, NH_tools.close_end, 1.)

interp_error1 = np.empty(o.shape[0], dtype=bool)
interp_error2 = np.empty(o.shape[0], dtype=bool)
interp_error1[:-1] = np.abs(np.diff(o[:,2])) > 100.
interp_error1[-1] = 0.
interp_error2[1:] = np.abs(np.diff(o[:,2])) > 100.
interp_error2[0] = 0.

o[interp_error1,2] = np.nan
o[interp_error2,2] = np.nan

plt.plot(points[:,0]/1187., o[:,2])
plt.gca().invert_xaxis()
plt.ylim([-180,180])
plt.show()
