import numpy as np
import pylab as pl
import scipy.stats as stats

ncell = 64
nbin = 100

alphaH = 4.e-19 # m^3 s^-1
nH = 1.e8 # m^-3
Q = 4.26e49 # s^-1

# compute the Stromgren radius
Rs = (0.75 * Q / (np.pi * nH**2 * alphaH))**(1. / 3.)

# output distances in pc
pc = 3.086e16 # m

Rs /= pc

data = np.memmap("intensities.dat", dtype = np.float64, shape = (ncell**3, 4),
                 mode = 'r')
pos = data[:, 0:3]
r = np.sqrt(pos[:, 0]**2 + pos[:, 1]**2 + pos[:, 2]**2)

rbin_edges = np.linspace(r.min(), r.max(), nbin + 1)
rbin_mid = 0.5 * (rbin_edges[1:] + rbin_edges[:-1])

Ibin, _, _ = stats.binned_statistic(r, data[:, 3], statistic = "mean",
                                    bins = rbin_edges)
I2bin, _, _ = stats.binned_statistic(r, data[:, 3]**2, statistic = "mean",
                                     bins = rbin_edges)
Isigma = np.sqrt(I2bin - Ibin**2)

pl.plot(rbin_mid / pc, Ibin)
pl.fill_between(rbin_mid / pc, Ibin - Isigma, Ibin + Isigma, alpha = 0.5)
pl.gca().axvline(x = Rs, linestyle = "--", color = "k", linewidth = 0.8)
pl.xlabel("radius (pc)")
pl.ylabel("neutral fraction")
pl.tight_layout()
pl.show()
