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
# @file plot_neutral_fraction_profile.py
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
import argparse

argparser = argparse.ArgumentParser("Plot a hydro snapshot.")

argparser.add_argument("-x", "--ncell_x", action = "store", type = int)
argparser.add_argument("-y", "--ncell_y", action = "store", type = int)
argparser.add_argument("-z", "--ncell_z", action = "store", type = int)

args = argparser.parse_args()

## run parameters
# number of cells in one coordinate dimension
# should be synced with the values used in the unit test
# we guess based on the file size
ncell = args.ncell_x * args.ncell_y * args.ncell_z

# memory-map the binary output file to a numpy array
data = np.memmap("hydro_result.dat", dtype = np.float64,
                 shape = (ncell, 10), mode = 'r')
pos = data[:, 0:3]
rho = data[:, 5]

# plot the means and scatter regions
pl.gca().axvline(x = 0., linestyle = "--", linewidth = 0.8, color = "k")
pl.plot(pos[:,0], rho, ".")
# set axis labels
pl.xlabel("position (m))")
pl.ylabel("density (kg m^-3)")
# clean up margins and show the plot
pl.tight_layout()
pl.savefig("hydro.png")
