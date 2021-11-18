#!/usr/bin/env python
import os
import numpy as np
from scipy import spatial
from scipy.constants import m_p, e
from spice_tools import look_directions, et2pydatetime
from progress import printProgressBar
from bisect import bisect_right
from astropy.io import fits
from datetime import datetime

calib = np.load('/home/nathan/Code/swap/swap_calibration_data.npz')

encounter_folder = '/home/nathan/Code/swap/nh-p-swap-3-pluto-v3.0/data'
cruise_folder = '/home/nathan/Code/swap/nh-x-swap-3-plutocruise-v3.0/data'

def readbins():
    import csv
    with open('/home/nathan/Code/swap/swap_e_bin.dat') as csvfile:
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

def extract_spectrogram(fit_file, preprocess='simple_spectra'):
    if preprocess == 'start_stop':
        pcem = np.zeros((fit_file['pcem_spect_hz'].data.shape[0], 2*fit_file['pcem_spect_hz'].data.shape[1]))
        scem = np.zeros_like(pcem)
        coin = np.zeros_like(pcem)
        pcem[:,::2] = fit_file['pcem_spect_hz'].data
        scem[:,::2] = fit_file['scem_spect_hz'].data
        coin[:,::2] = fit_file['coin_spect_hz'].data

        start_times = fit_file['time_label_spect'].data['start_et']
        stop_times  = fit_file['time_label_spect'].data['stop_et']
        times = np.zeros(2*start_times.shape[0])
        times[::2] = start_times
        times[1::2] = stop_times

    elif preprocess == 'simple_spectra':
        pcem = fit_file['pcem_spect_hz'].data
        scem = fit_file['scem_spect_hz'].data
        coin = fit_file['coin_spect_hz'].data
        times = fit_file['time_label_spect'].data['middle_et']

    elif preprocess == 'None':
        pcem = fit_file['pcem_spect_hz'].data
        scem = fit_file['scem_spect_hz'].data
        coin = fit_file['coin_spect_hz'].data
        times = fit_file['time_label_spect'].data

    else:
        raise ValueError("The argument preprocess must be one of: 'simple_spectra', 'start_stop', or 'None'")

    energies = fit_file['energy_label_spect'].data[0][2:]
    return pcem, scem, coin, times, energies

def _extract_date(item):
    # extract the date; i.e. first eight digits of the folder name
    # Since it's in the order YYYYMMDD we can compare them as integers
    return int(os.path.basename(item)[:8])

def _find_nearest_before(folders, target_date):
    b = [_extract_date(item) for item in folders]
    target = int(target_date.strftime('%Y%m%d'))
    i = bisect_right(b,target)
    if i:
        return folders[i-1]
    raise ValueError("No folder comes before or on that date")

def find_fit_file(start_et):
    start_date = et2pydatetime(start_et)

    if start_date < datetime(year=2015, month=1, day=15):
        mission_phase = cruise_folder
    else:
        mission_phase = encounter_folder

    date_folders = [os.path.join(mission_phase, d) for d in os.listdir(mission_phase) if os.path.isdir(os.path.join(mission_phase, d))]
    date_folders.sort(key=_extract_date)

    found = _find_nearest_before(date_folders, start_date)

    data_file, = [os.path.join(found,f) for f in os.listdir(found) if f.endswith('.fit')]

    return fits.open(data_file)

def find_spectrogram(start_et, preprocess='simple_spectra'):
    return extract_spectrogram( find_fit_file(start_et), preprocess=preprocess)


bin_centers, bin_edges = readbins()
Ageom = 2.74e-11 # km^2

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
    return np.interp(phi, calib['PHI'], calib['W'], left=0., right=0.)

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


azimuth_lab = calib['X']
theta_lab = -azimuth_lab
y = calib['Y'].transpose() # transpose for row-major/column-major conversion
arr = calib['ARR'].transpose() # transpose for row-major/column-major conversion
ecen = calib['ECEN']
def w(ee,theta):
    """Velocity and theta dependance of the SWAP response expressed in terms of energy
    instead of velocity.
    """
    # Find the index of the bin center closest to our value for energy and theta
    iee    = np.abs(ee[:,np.newaxis]-ecen[np.newaxis,:]).argmin(axis=1)
    itheta = np.abs(theta[:,np.newaxis]-theta_lab[np.newaxis,:]).argmin(axis=1)

    bin_energies=y[iee,:]*ee[:,np.newaxis]

    ret = np.empty(ee.shape[0], dtype=np.float64)
    for i in range(ret.shape[0]):
        ret[i] = np.interp(ee[i], bin_energies[i], arr[:,itheta[i],iee[i]], left=0., right=0.)

    return ret

def Aeff(ee, mrat, scem_voltage=2400.):
    r"""Computes the detector efficiency as a function of energy.
    Uses the Valek method. Ignores time dependance.
    ee: particle energy in eV

    Valek scem efficiency formula:
    \frac{\sqrt{2 \frac{(ee + q V)}{1000}}}{\sqrt{m}}
    """
    He_mask  = (mrat==1./2.)
    CH4_mask = (mrat==1./16.)
    N2_mask  = (mrat==1./28.)
    H_mask   = ~(He_mask | CH4_mask | N2_mask)

    pe = 0.34889024
    k_h = 0.40146017
    k_ch4 = 2.44629
    alpha_h = 0.98122561
    alpha_ch4 = 0.94425982

    # Build up the complicated formula for efficiency (Valek scem_eff(E), see above) in a vectorized way without wasting memory.
    # Helium has a charge of 2 while all the others have a charge of 1.
    ret = np.sqrt(2*(ee + scem_voltage)/1000.0, where=(~He_mask))
    np.sqrt(2*(ee + 2.*scem_voltage)/1000.0, where=He_mask, out=ret)

    # Divide by sqrt of amu for all species (leave H alone since sqrt(1) = 1)
    np.divide(ret, np.sqrt(4),  where=He_mask,  out=ret)
    np.divide(ret, np.sqrt(16), where=CH4_mask, out=ret)
    np.divide(ret, np.sqrt(28), where=N2_mask, out=ret)

    np.power(ret, alpha_h,  where=(H_mask   | He_mask), out=ret)
    np.power(ret, alpha_ch4, where=(CH4_mask | N2_mask), out=ret)

    np.multiply(k_h,  ret, where=(H_mask   | He_mask), out=ret)
    np.multiply(k_ch4, ret, where=(CH4_mask | N2_mask), out=ret)

    np.power(1.0 - pe,  ret, out=ret)

    np.subtract(1.0, ret, out=ret)

    # Helium has double the efficiency of Hydrogen
    np.multiply(2.0, ret, where=He_mask, out=ret)

    return ret

def transmission(ee, l):
    ww = w(ee, l[:,0])
    pp = p(l[:,1])
    return ww*pp

def swap_resp(ee, l, mrat):
    """Compute swap response"""
    A = Ageom*Aeff(ee, mrat)
    T = transmission(ee, l)

    return A*T

def spectrum(v, mrat, n, o):
    """Generate a spectrum for the given particles and orientation"""
    eq = Eoverq(v,mrat)
    ee = E(eq, mrat)
    l  = look_directions(v, o)
    resp = swap_resp(ee, l, mrat)
    nv = n * np.linalg.norm(v, axis=1)

    # Counts due to each individual macro-particle
    counts = nv*resp

    return np.histogram(eq, bin_edges, weights=counts)[0]

def spectrogram(x, v, mrat, beta, points, orientations, radius=None, volume=None, progress=False):
    """The spectrogram is built out of a spectrum for each given point in space"""
    mutually_exclusive_args = [radius, volume]
    if mutually_exclusive_args.count(None) != 1:
        raise TypeError("Only provide one of radius or volume")
    if radius is not None:
        print('Warining: radius is depreciated')
        volume = (4./3.)*np.pi*radius**3



    ret = np.empty((points.shape[0], bin_edges.shape[0]-1), dtype=np.float64)

    if len(x) == 0:
        ret[:,:] = 0
        return ret

    if progress:
        printProgressBar(0,1)

    kdparts  = spatial.cKDTree(x)
    kdpoints = spatial.cKDTree(points)
    effective_radius = volume**(1./3.)/2.
    radius_enhancement = 4
    enhanced_radius = effective_radius*radius_enhancement
    enhanced_volume = (2.0*enhanced_radius)**3.0
    local = kdpoints.query_ball_tree(kdparts, enhanced_radius, p=np.inf)

    # Since we ignore particle weighting we need to correct counts by a certain factor
    # assuming particles are uniformly distributed within the cell volume.
    # We also assume the same cell volume throughout.
    effective_volume = enhanced_volume/8.0

    for i, l in enumerate(local):
        if progress:
            printProgressBar(len(local)+i, 2*len(local))
        ret[i, :] = spectrum(v[l], mrat[l], 1./(effective_volume*beta[l]), orientations[i])

    if progress:
        printProgressBar(1,1)

    return ret

def spectrograms_by_species(x,v,mrat,beta,points,orientations, radius=None, volume=(1187./2)**3, progress=False):
    if radius is not None:
        raise TypeError("Don't use radius, use volume instead")
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

    H_spec = spectrogram(H_xp, H_v, H_mrat, H_beta, points, orientations, volume=volume, progress=progress)
    He_spec = spectrogram(He_xp, He_v, He_mrat, He_beta, points, orientations, volume=volume, progress=progress)
    CH4_spec = spectrogram(CH4_xp, CH4_v, CH4_mrat, CH4_beta, points, orientations, volume=volume, progress=progress)

    return [H_spec, He_spec, CH4_spec]

def single_spectrum_by_tag(v, mrat, beta, tags, orientation, volume):

    tag_spectrums = {}
    for tag in np.unique(tags):
        tagged_v = v[tags == tag]
        tagged_beta = beta[tags == tag]
        tagged_mrat = mrat[tags == tag]

        # Since we ignore particle weighting we need to correct counts by a certain factor
        # assuming particles are uniformly distributed within the cell volume
        effective_volume = volume/8.0

        tag_spectrums[tag] = spectrum(tagged_v, tagged_mrat, 1./(effective_volume*tagged_beta), orientation)

    return tag_spectrums

def spectrograms_by_tag(x,v,mrat,beta,points,tags,orientations, radius=1187., progress=False):
    tag_spectrograms = {}
    for tag in np.unique(tags):
        tagged_x = x[tags == tag]
        tagged_v = v[tags == tag]
        tagged_beta = beta[tags == tag]
        tagged_mrat = mrat[tags == tag]
        tag_spectrograms[tag] = spectrogram(tagged_x, tagged_v, tagged_mrat, tagged_beta, points, orientations, radius, progress)

    return tag_spectrograms
