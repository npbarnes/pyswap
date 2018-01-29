from os.path import join
import numpy as np
import matplotlib.pyplot as plt
import argparse
from progress import printProgressBar 
import FortranFile as ff
import HybridParticleReader as hpr
from HybridParams import HybridParams
from scipy.constants import value, unit
kg_per_amu = value('atomic mass constant')
assert unit('atomic mass constant') == 'kg'

def marginal_outflow(p):
    part_out = np.zeros(2*p.para['nt'])
    mass_out = np.zeros(2*p.para['nt'])
    # for each processor
    for n in range(p.para['num_proc']):
        f = ff.FortranFile(join(p.particle,"c.outflowing_"+str(n+1)+".dat"))

        # for each half timestep
        for i in range(2*p.para['nt']):
            try:
                mrat = f.readReals()
            except ff.NoMoreRecords:
                break
            beta_p = f.readReals()
            tags = f.readReals()

            # Each of the arrays must have the same length
            assert len(mrat) == len(beta_p) and len(mrat) == len(tags)
            # If that length is zero than there was no outflow
            if len(mrat) == 0:
                continue

            # for each macro particle
            for m,b,t in zip(mrat, beta_p, tags):
                if t != 0:
                    part_out[i] += 1/(b*p.para['beta'])
                    mass_out[i] += kg_per_amu/m * 1/(p.para['beta']*b)
        printProgressBar(n+1, p.para['num_proc'], prefix='Outflow Files')

    return part_out, mass_out

def cumulative_outflow(p):
    part, mass = marginal_outflow(p)
    return np.cumsum(part), np.cumsum(mass)

def total_outflow(p):
    part, mass = marginal_outflow(p)
    return np.sum(part), np.sum(mass)

parser = argparse.ArgumentParser()
parser.add_argument('prefix')

args = parser.parse_args()
p = HybridParams(args.prefix)

part_out, _ = marginal_outflow(p)

print(p.para['dt'])

plt.plot(part_out/(p.para['dt']/2))

plt.show()
