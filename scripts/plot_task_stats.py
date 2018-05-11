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
# @file plot_task_stats.py
#
# @brief Script to plot task statistics per iteration.
#
# @author Bert Vandenbroucke (bv7@st-andrews.ac.uk)
##

# import modules
import numpy as np
import matplotlib
matplotlib.use("Agg")
import pylab as pl
import glob

# get a list of all files that are present
files = sorted(glob.glob("task_stats_??.txt"))
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
  pl.bar(alldata[:,i,0] + steps, alldata[:,i,1], 1. / nproc)

# plot layout
pl.xticks([])
pl.ylabel("Number of tasks")

# save the plot
pl.tight_layout()
pl.savefig("task_stats.png")
