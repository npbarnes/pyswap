#!/usr/bin/python
from sys import argv
import numpy as np
from scipy.io import readsav
from scipy.interpolate import RegularGridInterpolator
from HybridReader2 import HybridReader2 as hr
from pprint import pprint
import matplotlib.pyplot as plt

k = 1.38064852e-23 # Boltzmann constant in SI units [m^2 kg s^-2 K^-1] == [J/K]

p = hr(argv[1],'up')
traj = readsav('traj.sav')['traj']
traj = np.dstack((traj.x[0],traj.y[0]))[0]

qx = p.para['qx']
qy = p.para['qy']
cz = p.para['zrange']/2

try:
    po = p.para['pluto_offset']
except KeyError:
    po = 30

Rp = 1187

traj[:,0] = (traj[:,0] - qx[len(qx)/2 + po])/Rp
traj[:,1] = (traj[:,1] - qy[len(qy)/2])/Rp
qx = (qx - qx[len(qx)/2 + po])/Rp
qy = (qy - qy[len(qy)/2])/Rp

def vprofile():
    v = hr(argv[1],'up')
    vdata = v.get_last_timestep()[-1]
    vmag = np.sqrt(vdata[:,:,cz,0]**2 + vdata[:,:,cz,1]**2 + vdata[:,:,cz,2]**2)

    rgi = RegularGridInterpolator(points=[qx,qy], values=vmag, bounds_error=False)
    vinterp = rgi(traj)

    return vinterp

def nprofile():
    n = hr(argv[1],'np')
    ndata = n.get_last_timestep()[-1]
    # Convert to cm^-3
    ndata = ndata*1e-15

    rgi = RegularGridInterpolator(points=[qx,qy], values=ndata[:,:,cz], bounds_error=False)
    ninterp = rgi(traj)

    return ninterp

def tprofile():
    t = hr(argv[1],'temp')
    tdata = t.get_last_timestep()[-1]
    # Convert from eV to K
    tdata = tdata/(k*6.242e18)

    rgi = RegularGridInterpolator(points=[qx,qy], values=tdata[:,:,cz], bounds_error=False)
    tinterp = rgi(traj)

    return tinterp


v = vprofile()
n = nprofile()
t = tprofile()
p = k*n*t*(100**3)*(1e12) #pressure in pPa

f, (av,an,at,ap) = plt.subplots(4, sharex=True)
# Settings for the shared x axis
av.invert_xaxis()
av.set_xlim([-10,-100])
ap.set_xlabel('X [R$_p$]')

av.plot(qx[::2],v)
an.plot(qx[::2],n)
at.plot(qx[::2],t)
ap.plot(qx[::2],p)

av.set_ylabel('<v$_{flow}$> [km/s]')
an.set_ylabel('Density [cm$^{-3}$]')
at.set_ylabel('Temp [K]')
ap.set_ylabel('nkT [pPa]')

an.set_yscale('log')
at.set_yscale('log')
ap.set_yscale('log')

f.savefig('profiles.png', format='png', bbox_inches='tight')
f.clear()
