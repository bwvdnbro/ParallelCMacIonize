################################################################################
# This file is part of CMacIonize
# Copyright (C) 2017 Bert Vandenbroucke (bert.vandenbroucke@gmail.com)
#
# CMacIonize is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# CMacIonize is distributed in the hope that it will be useful,
# but WITOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with CMacIonize. If not, see <http://www.gnu.org/licenses/>.
################################################################################

##
# @file plot_intensity_profile.py
#
# @brief Script to plot the neutral fraction profile of the Stromgren test.
#
# @author Bert Vandenbroucke (bv7@st-andrews.ac.uk)
##

# import modules
import numpy as np
import matplotlib
matplotlib.use("Agg")
import pylab as pl
import scipy.stats as stats
import sys
import os

## run parameters
# number of cells in one coordinate dimension
# should be synced with the values used in the unit test
# we guess based on the file size
ncell = int(np.cbrt(os.path.getsize("intensities.dat")>>5))
# number of bins to use to bin the results
nbin = 100

# physical parameters, used to compute the analytic Stromgren radius
alphaH = 4.e-19 # m^3 s^-1
nH = 1.e8 # m^-3
Q = 4.26e49 # s^-1
PR = 0.
if len(sys.argv) > 1:
  PR = float(sys.argv[1])

# compute the Stromgren radius (in m)
Rs = (0.75 * Q / (1. - PR) / (np.pi * nH**2 * alphaH))**(1. / 3.)

# output distances in pc
pc = 3.086e16 # m
Rs /= pc

# memory-map the binary output file to a numpy array
data = np.memmap("intensities.dat", dtype = np.float64, shape = (ncell**3, 4),
                 mode = 'r')
# compute the radii
pos = data[:, 0:3]
r = np.sqrt(pos[:, 0]**2 + pos[:, 1]**2 + pos[:, 2]**2)

# set up the radial bins
rbin_edges = np.linspace(r.min(), r.max(), nbin + 1)
rbin_mid = 0.5 * (rbin_edges[1:] + rbin_edges[:-1])

# now bin the neutral fractions and compute the scatter within the bins
Ibin, _, _ = stats.binned_statistic(r, data[:, 3], statistic = "mean",
                                    bins = rbin_edges)
I2bin, _, _ = stats.binned_statistic(r, data[:, 3]**2, statistic = "mean",
                                     bins = rbin_edges)
Isigma = np.sqrt(I2bin - Ibin**2)

# plot the means and scatter regions
pl.plot(rbin_mid / pc, Ibin)
pl.fill_between(rbin_mid / pc, Ibin - Isigma, Ibin + Isigma, alpha = 0.5)
# add the analytic Stromgren radius
pl.gca().axvline(x = Rs, linestyle = "--", color = "k", linewidth = 0.8)
# switch to an analytic scale and clip the negative scatter boundaries
pl.gca().set_yscale("log", nonposy = "clip")
# set axis labels
pl.xlabel("radius (pc)")
pl.ylabel("neutral fraction")
# clean up margins and show the plot
pl.tight_layout()
pl.savefig("stromgren.png")
