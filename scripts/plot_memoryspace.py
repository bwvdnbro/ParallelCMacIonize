################################################################################
# This file is part of CMacIonize
# Copyright (C) 2018 Bert Vandenbroucke (bert.vandenbroucke@gmail.com)
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
# @file plot_memoryspace.py
#
# @brief Script to plot an overview of the photon buffer load per iteration for
# the different processes.
#
# Optionally, the script can compute the number of buffers that are present by
# default (if the number of subgrids is passed on as command line argument) and
# subtract this from the plot. This only gives sensible results for non-MPI
# runs.
#
# @author Bert Vandenbroucke (bv7@st-andrews.ac.uk)
##

# import modules
import numpy as np
import matplotlib
matplotlib.use("Agg")
import pylab as pl
import glob
import argparse

# parse the command line arguments
argparser = argparse.ArgumentParser(description = 
  "Plot the number of actively used photon buffers per iteration.")

argparser.add_argument("-x", "--nbx", action = "store", type = int, default = 0)
argparser.add_argument("-y", "--nby", action = "store", type = int, default = 0)
argparser.add_argument("-z", "--nbz", action = "store", type = int, default = 0)

args = argparser.parse_args()

nbx = args.nbx
nby = args.nby
nbz = args.nbz

# if requested: compute the default number of buffers
numsubgrid = nbx * nby * nbz
minnum = 0
if numsubgrid > 0:
  for ix in range(nbx):
    for iy in range(nby):
      for iz in range(nbz):
        for ox in range(-1, 2):
          for oy in range(-1, 2):
            for oz in range(-1, 2):
              cx = ix + ox
              cy = iy + oy
              cz = iz + oz
              if cx >= 0 and cx < nbx and cy >= 0 and cy < nby and cz >= 0 \
                 and cz < nbz:
                minnum += 1

# get a list of all files that are present
files = sorted(glob.glob("memoryspace_??.txt"))
# collect the data
alldata = []
for file in files:
  data = np.loadtxt(file)
  if len(data.shape) > 1:
    data = np.array(sorted(data, key = lambda line: line[0]))
  else:
    data = [data]
  alldata.append(data)

alldata = np.array(alldata)

# prepare data for plotting
nproc = int(alldata[:,:,0].max()) + 2
alldata[:,:,0] = alldata[:,:,0] / nproc
steps = np.arange(len(files))

# plot the contributions for the different processes
for i in range(nproc - 1):
 pl.bar(alldata[:,i,0] + steps, alldata[:,i,1] - minnum, 1. / nproc)

# plot layout
pl.xticks([])
pl.ylabel("Number of used photon buffers")

# save the plot
pl.tight_layout()
pl.savefig("memoryspace_stats.png")
