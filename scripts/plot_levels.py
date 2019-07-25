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
# @file plot_levels.py
#
# @brief Script to plot a copy level histogram.
#
# @author Bert Vandenbroucke (bv7@st-andrews.ac.uk)
##

# import modules
import numpy as np
import matplotlib

matplotlib.use("Agg")
import pylab as pl
from matplotlib.ticker import MaxNLocator

# load the data
data = np.loadtxt("copy_levels.txt", dtype=np.int32)

# find the minimum and maximum
minlevel = data[:, 1].min()
maxlevel = data[:, 1].max()

# compute the levels and histogram
levels = np.linspace(minlevel, maxlevel, maxlevel - minlevel + 1)
hist = np.bincount(data[:, 1])

# make the plot
pl.bar(levels, hist)
# add text labels
for i in range(len(levels)):
    pl.text(levels[i], hist[i], "{0}".format(hist[i]), ha="center")

# set a logarithmic y axis
pl.gca().set_yscale("log")
# force integer x ticks
pl.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

# set labels
pl.xlabel("Copy level")
pl.ylabel("Number of subgrids")

# finalize and save plot
pl.tight_layout()
pl.savefig("copy_levels.png")
