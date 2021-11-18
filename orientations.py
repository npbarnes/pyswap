import numpy as np
import spice_tools as st
import matplotlib.pyplot as plt

def cmat2tps(cmat):
    # The first column of cmat is the unit vector in the direction
    # of the sun in spacecraft coords.
    # Its lat-long is theta and phi
    t,p = st.vec2tp(np.copy(cmat[:,0]))
    
    # Now take the pluto coords z direction in spacecraft coords
    z = cmat[:,2]
    # and compute spin angle
    s = np.arctan2(z[0],z[2])

    return t,p,s

def get_os(cmats):
    ret = np.empty((cmats.shape[0],3))
    for i,cmat in enumerate(cmats):
        ret[i,:] = cmat2tps(cmat)
    return ret

if __name__ == '__main__':
    #points, cmats, times  = st.trajectory(st.rehearsal_start, st.rehearsal_end, 30.)
    points, cmats, times  = st.trajectory(st.flyby_start, st.flyby_end, 30.)

    os_spice = get_os(cmats)

    fig, (ax1,ax2,ax3) = plt.subplots(3)

    ax1.plot(times, os_spice[:,0])
    ax2.plot(times, os_spice[:,1])
    ax3.plot(times, os_spice[:,2])

    plt.show()
