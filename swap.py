#!/usr/bin/env python
import os
from os.path import join
import numpy as np
from scipy.spatial.distance import cdist
from scipy.constants import m_p, e
from math import sin, cos, radians
import NH_tools
import HybridParams
import FortranFile as ff
import math
from progress import printProgressBar 

# Importing idlpy sets the working directory to the home directory for
# both Python and IDL interpreters.
# These lines are a workaround to unstupid that behavior.
cwd = os.getcwd()
from idlpy import IDL
os.chdir(cwd)
IDL.cd(cwd)

# Load SWAP calibration information
IDL.restore('fin_arr_ebea_ang_eb_bg_corr.sav')
IDL.restore('w_phi.sav')
IDL.restore('prob_scem_parm.sav')

def readbins():
    import csv
    with open('swap_e_bin.dat') as csvfile:
        binreader = csv.reader(csvfile)
        mid, nextlargest, largest = next(binreader)
        bin_centers = [float(mid)]
        bin_edges = [float(largest), float(nextlargest)]
        for row in binreader:
            bin_centers.append(float(row[0]))
            bin_edges.append(float(row[1]))

    # reverse to get bins from low to high
    bin_centers = np.array(bin_centers[::-1])
    bin_edges   = np.array(bin_edges[::-1])
    return bin_centers, bin_edges

bin_centers, bin_edges = readbins()
Ageom = 2.74e-11 # km^2

"""Pasive rotation matricies used to change coordinate systems"""
def Rx(a):
    a = radians(a)
    return np.array([   [1.0,0.0,0.0],
                        [0.0,cos(a),sin(a)],
                        [0.0,-sin(a),cos(a)]])
def Ry(a):
    a = radians(a)
    return np.array([   [cos(a),0.0,-sin(a)],
                        [0.0,1.0,0.0],
                        [sin(a),0.0,cos(a)]])
def Rz(a):
    a = radians(a)
    return np.array([   [cos(a),sin(a),0.0],
                        [-sin(a),cos(a),0.0],
                        [0.0,0.0,1.0]])

def R1(o):
    """Given an orientation o = (theta, phi, spin), R1(o) will be the rotation matrix
    to convert pluto coordinates into spacecraft coordinates.
    """
    return np.linalg.multi_dot([Rz(o[1]),Rx(o[0]),Ry(o[2]),Rz(-90)])

def look_vectors(v,o):
    """Converts velocity vectors to spacecraft look directions"""
    # Negative of the velocity vector in NH coordinates.
    # The einsum just applies the appropriate rotation matrix to each v.
    # We take the negative since it's the look direction; i.e. the
    # direction to look to see that particle coming in.
    return -np.einsum('ij,kj->ki', R1(o), v)

def look_directions(v, o):
    """Computes the SWAP look direction phi and theta for each of a collection of ion velocities"""
    l = look_vectors(v,o)
    ret = np.empty((l.shape[0],2), dtype=np.float64)

    ret[:,0] = np.degrees(np.arctan2(l[:,0],l[:,1]))
    ret[:,1] = np.degrees(np.arctan2(l[:,2],np.sqrt(l[:,0]**2 + l[:,1]**2)))

    return ret

def p(phi):
    """Phi dependance of the SWAP response"""
    return np.interp(phi, IDL.wphi['PHI'], IDL.wphi['W'])

def Eoverq(v,mrat):
    """Compute energy per charge from velocity and mass ratio (q/m)"""
    ret = np.linalg.norm(v, axis=1)
    np.square(ret, out=ret)
    np.multiply(m_p/e, ret, out=ret)
    np.divide(ret, mrat, out=ret)

    # Convert to eV per electron charge. Same as J/C.
    np.multiply(0.5*1000.**2, ret, out=ret)

    return ret

def E(Eoverq, mrat):
    """Compute energy in eV given energy per charge in eV per electron charge"""
    # multiply by 2 for He++ to give energy in eV for all particles
    He_mask = (mrat == 1./2.)
    ret = np.empty(Eoverq.shape, dtype=np.float64)
    np.multiply(Eoverq, 2., where=He_mask, out=ret)
    ret[~He_mask] = Eoverq[~He_mask]

    return ret


azimuth_lab = IDL.s4['X']
theta_lab = -azimuth_lab
y = IDL.s4['Y'].transpose() # transpose for row-major/column-major conversion
arr = IDL.s4['ARR'].transpose() # transpose for row-major/column-major conversion
ecen = IDL.s4['ECEN']
def w(ee,theta):
    """Velocity and theta dependance of the SWAP response expressed in terms of energy
    instead of velocity.
    """
    # Find the index of the bin center closest to our value for energy and theta
    iee    = np.abs(ee[:,np.newaxis]-ecen[np.newaxis,:]).argmin(axis=1)
    itheta = np.abs(theta[:,np.newaxis]-theta_lab[np.newaxis,:]).argmin(axis=1)

    bin_energies=y[iee,:]*ee[:,np.newaxis]

    ret = np.empty(ee.shape[0], dtype=np.float64)
    for i in xrange(ret.shape[0]):
        ret[i] = np.interp(ee[i], bin_energies[i], arr[:,itheta[i],iee[i]])

    return ret

def Aeff(ee, mrat, scem_voltage=2400.):
    """Computes the detector efficiency as a function of energy.
    Uses the Valek method. Ignores time dependance.
    ee: particle energy in eV
    """
    He_mask  = (mrat==1./2.)
    CH4_mask = (mrat==1./16.)
    N2_mask  = (mrat==1./28.)
    H_mask   = ~(He_mask | CH4_mask | N2_mask)

    # Build up the complicated formula for efficiency (Valek scem_eff(E)) in a vectorized way without wasting memory.
    # Helium has a charge of 2 while all the others have a charge of 1.
    ret = np.sqrt(2*(ee + scem_voltage)/1000.0, where=(~He_mask))
    np.sqrt(2*(ee + 2.*scem_voltage)/1000.0, where=He_mask, out=ret)

    # Divide by sqrt of amu for all species (leave H alone since sqrt(1) = 1)
    np.divide(ret, np.sqrt(4),  where=He_mask,  out=ret)
    np.divide(ret, np.sqrt(16), where=CH4_mask, out=ret)
    np.divide(ret, np.sqrt(28), where=N2_mask, out=ret)

    np.power(ret, IDL.h_parms[1],  where=(H_mask   | He_mask), out=ret)
    np.power(ret, IDL.n2_parms[1], where=(CH4_mask | N2_mask), out=ret)

    np.multiply(IDL.h_parms[0],  ret, where=(H_mask   | He_mask), out=ret)
    np.multiply(IDL.n2_parms[0], ret, where=(CH4_mask | N2_mask), out=ret)

    np.power(1.0 - IDL.h_parms[2],  ret, where=(H_mask   | He_mask), out=ret)
    np.power(1.0 - IDL.n2_parms[2], ret, where=(CH4_mask | N2_mask), out=ret)

    np.subtract(1.0, ret, out=ret)

    # Helium has double the efficiency of Hydrogen
    np.multiply(2.0, ret, where=He_mask, out=ret)

    return ret

def swap_resp(ee, l, mrat, orientation):
    """Compute swap response"""
    A = Ageom*Aeff(ee, mrat)
    ww = w(ee, l[:,1])
    pp = p(l[:,0])

    return A*ww*pp

def spectrum(v, mrat, n, o):
    """Generate a spectrum for the given particles and orientation"""
    eq = Eoverq(v,mrat)
    ee = E(eq, mrat)
    l  = look_directions(v, o)
    resp = swap_resp(ee, l, mrat, o)
    nv = n * np.linalg.norm(v, axis=1)

    # Counts due to each individual macro-particle
    counts = nv*resp

    return np.histogram(eq, bin_edges, weights=counts)[0]

def spectrogram(x, v, mrat, beta, points, orientations, radius=1187., progress=False):
    """The spectrogram is built out of a spectrum for each given point in space"""
    ret = np.empty((points.shape[0], bin_edges.shape[0]-1), dtype=np.float64)

    if progress:
        printProgressBar(0,1)

    local = cdist(points, x) < radius
    dV = (4./3.)*np.pi*radius**3


    for i, l in enumerate(local):
        if progress:
            printProgressBar(len(local)+i, 2*len(local))
        ret[i, :] = spectrum(v[l], mrat[l], 1./(dV*beta[l]), orientations[i])

    if progress:
        printProgressBar(1,1)

    return ret


def particle_data(hybrid_folder, n=0):
    para = HybridParams.HybridParams(hybrid_folder).para

    if para['num_proc'] % 2 != 1:
        raise NotImplemented("Tell Nathan he needs to write the code for concatenating an odd number of processors")
    else:
        center = int(math.ceil(para['num_proc']/2.0))

        pp = pluto_position(para)

        x_list = []
        v_list = []
        mrat_list = []
        beta_list = []
        tags_list = []
        for offset in (range(-n,n+1) if isinstance(n,int) else n):
            cur_rank = center+offset
            from_bottom = para['num_proc'] - (cur_rank+1)

            _, x, v, mrat, beta, tags = read_particle_files(hybrid_folder, cur_rank)

            # Convert processor local coordinate to pluto coordinates
            x[:,0] -= pp
            x[:,1] -= np.max(para['qy'])/2
            x[:,2] += np.max(para['qz'])*from_bottom - from_bottom*para['delz'] - np.max(para['qzrange'])/2

            x_list.append(x)
            v_list.append(v)
            mrat_list.append(mrat)
            beta_list.append(beta)
            tags_list.append(tags)

        ret_x = np.concatenate(x_list)
        ret_v = np.concatenate(v_list)
        ret_mrat = np.concatenate(mrat_list)
        ret_beta = np.concatenate(beta_list)
        ret_tags = np.concatenate(tags_list)

    return para, ret_x, ret_v, ret_mrat, ret_beta, ret_tags


def read_particle_files(hybrid_folder, procnum):
    """Read datafiles"""
    para = HybridParams.HybridParams(hybrid_folder).para

    xp_file = ff.FortranFile(join(hybrid_folder,'particle','c.xp_{}.dat'.format(procnum)))
    vp_file = ff.FortranFile(join(hybrid_folder,'particle','c.vp_{}.dat'.format(procnum)))
    mrat_file = ff.FortranFile(join(hybrid_folder,'particle','c.mrat_{}.dat'.format(procnum)))
    beta_p_file = ff.FortranFile(join(hybrid_folder,'particle','c.beta_p_{}.dat'.format(procnum)))
    tags_file = ff.FortranFile(join(hybrid_folder,'particle','c.tags_{}.dat'.format(procnum)))

    xp_file.seek(0,os.SEEK_END)
    vp_file.seek(0,os.SEEK_END)
    mrat_file.seek(0,os.SEEK_END)
    beta_p_file.seek(0,os.SEEK_END)
    tags_file.seek(0,os.SEEK_END)

    xp = xp_file.readBackReals().reshape((-1, 3), order='F')
    vp = vp_file.readBackReals().reshape((-1, 3), order='F')
    mrat = mrat_file.readBackReals()
    beta_p = beta_p_file.readBackReals()
    tags = tags_file.readBackReals()

    xp_file.close()
    vp_file.close()
    mrat_file.close()
    beta_p_file.close()
    tags_file.close()

    return (para, xp.astype(np.float64), vp.astype(np.float64), mrat.astype(np.float64),
                    para['beta']*beta_p.astype(np.float64), tags)

def pluto_position(p):
    """get the position of pluto in simulation coordinates"""
    return p['qx'][p['nx']/2 + p['pluto_offset']]

def trajectory(t1, t2, dt):
    """Read in New Horizons trajectory and orientation data"""
    times = np.arange(t1,t2,dt)

    pos = np.empty((len(times), 3), dtype=np.float64)
    o   = np.empty((len(times), 3), dtype=np.float64)

    for i,t in enumerate(times):
        pos[i,:] = NH_tools.coordinate_at_time(t)
        o[i,:]   = NH_tools.orientation_at_time(t)

    return pos, o
        

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    para, xp, v, mrat, beta, tags = open_stuff('/home/nathan/data/pluto-2/databig')
    v = v.astype(np.float64)

    #points, o = trajectory(NH_tools.close_start, NH_tools.close_end, 600.)

    points = np.zeros((50,3),dtype='d')
    points[:,0] = np.linspace(20,-100,50)*1187.
    points[:,2] = np.max(para['qz'])/2

    o = np.zeros((50,3),dtype='d')
    o[:,2] = 90.

    xp = xp[tags != 0]
    v = v[tags != 0]
    mrat = mrat[tags != 0]
    beta = beta[tags != 0]

    sp = spectrogram(xp, v, mrat, beta, points, o)
    plt.pcolormesh(sp.T, norm=LogNorm())
    plt.gca().set_yscale('log')
    plt.colorbar()
    plt.show()