#!/usr/bin/env python
import os
import numpy as np
from scipy.spatial.distance import cdist
from scipy.constants import m_p, e
from math import sin, cos, radians
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
    """Computes the SWAP look direction phi and theta for each of a collection of ion velocities
    Returns array of [theta, phi] pairs.
    """
    l = look_vectors(v,o)
    ret = np.empty((l.shape[0],2), dtype=np.float64)

    # Theta
    ret[:,0] = np.degrees(np.arctan2(l[:,2],np.sqrt(l[:,0]**2 + l[:,1]**2)))
    # Phi
    ret[:,1] = np.degrees(np.arctan2(l[:,0],l[:,1]))

    return ret

sun_dir = np.array([[1.,0.,0.]])
swap_fov = (10., 276.) # theta, phi
def is_sunview(o, tolerance=5.):
    """Determines if the sun is within 5 degrees of any part of the field of view
    at each orientation. You can change the tolerance with the keyword arg.
    """
    theta, phi = look_directions(sun_dir, o).T

    return (np.abs(theta) < swap_fov[0]/2+tolerance) & (np.abs(phi) < swap_fov[1]/2+tolerance)

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
    ww = w(ee, l[:,0])
    pp = p(l[:,1])

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

def spectrograms_by_species(x,v,mrat,beta,points,orientations, radius=1187., progress=False):
    H_xp = x[mrat == 1.]
    H_v = v[mrat == 1.]
    H_beta = beta[mrat == 1.]
    H_mrat = mrat[mrat == 1.]

    He_xp = x[mrat == 1./2.]
    He_v = v[mrat == 1./2.]
    He_beta = beta[mrat == 1./2.]
    He_mrat = mrat[mrat == 1./2.]

    CH4_xp = x[mrat == 1./16.]
    CH4_v = v[mrat == 1./16.]
    CH4_beta = beta[mrat == 1./16.]
    CH4_mrat = mrat[mrat == 1./16.]

    H_spec = spectrogram(H_xp, H_v, H_mrat, H_beta, points, orientations, radius, progress)
    He_spec = spectrogram(He_xp, He_v, He_mrat, He_beta, points, orientations, radius, progress)
    CH4_spec = spectrogram(CH4_xp, CH4_v, CH4_mrat, CH4_beta, points, orientations, radius, progress)

    return [H_spec, He_spec, CH4_spec]
