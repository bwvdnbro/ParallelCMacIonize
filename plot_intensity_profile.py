import numpy as np
import pylab as pl
import scipy.stats as stats

data = np.memmap("intensities.dat", dtype = np.float64, shape = (128**3, 4),
                 mode = 'r')
pos = data[:, 0:3] - 0.5
r = np.sqrt(pos[:, 0]**2 + pos[:, 1]**2 + pos[:, 2]**2)

rbin_edges = np.linspace(r.min(), r.max(), 101)
rbin_mid = 0.5 * (rbin_edges[1:] + rbin_edges[:-1])

Ibin, _, _ = stats.binned_statistic(r, data[:, 3], statistic = "mean",
                                    bins = rbin_edges)
I2bin, _, _ = stats.binned_statistic(r, data[:, 3]**2, statistic = "mean",
                                     bins = rbin_edges)
Isigma = np.sqrt(I2bin - Ibin**2)

pl.plot(rbin_mid, Ibin)
pl.fill_between(rbin_mid, Ibin - Isigma, Ibin + Isigma, alpha = 0.5)
pl.gca().set_yscale("log", nonposy = "clip")
pl.show()
