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
# @file plot_costs.py
#
# @brief Script to plot the cost distribution for a given cost output file.
#
# @author Bert Vandenbroucke (bv7@st-andrews.ac.uk)
##

# import modules
import numpy as np
import matplotlib
matplotlib.use("Agg")
import pylab as pl
import argparse

# parse command line arguments
argparser = argparse.ArgumentParser(
  description = "Plot cost plot based on a given cost output file.")

argparser.add_argument("-n", "--name", action = "store", required = True)
argparser.add_argument("-l", "--labels", action = "store_true")

args = argparser.parse_args()

name = args.name

# change the default matplotlib settings
pl.rcParams["text.usetex"] = True
pl.rcParams["figure.figsize"] = (12, 10)
pl.rcParams["font.size"] = 14

# load the cost file
print "Plotting costs for", name, "..."
data = np.loadtxt(name, dtype = np.uint64)

# gather information about the system
nproc = int(data[:,4].max()) + 1

print "number of processes:", nproc

# get the total and average costs
totcompcost = data[:,1].sum()
avgcompcost = totcompcost / nproc
totphotcost = data[:,2].sum()
avgphotcost = totphotcost / nproc
totsrccost = data[:,3].sum()
avgsrccost = totsrccost  / nproc

print "average computational cost:", avgcompcost
print "average photon cost:", avgphotcost
print "average source cost:", avgsrccost
print "number of subgrids:", len(data)

## make the cost plots

# we will alternate the blocks in these two colours
colors = [[(1, 0, 0, 1), (0, 0, 1, 1)],
          [(1, 0, 0, 0.2), (0, 0, 1, 0.2)]]

fig, ax = pl.subplots(3, 1, sharex = True)

barmax = 0.
# loop over the processes
for iproc in range(nproc):
  # filter out the data for this process
  process = data[data[:,4] == iproc]

  # sort and plot the computational cost
  process = sorted(process, key = lambda line: line[1], reverse = True)
  bars = [(0, process[0][1] * 1. / avgcompcost)]
  bar_colors = [colors[process[0][5]][0]]
  for i in range(1, len(process)):
    bars.append((bars[-1][0] + bars[-1][1], process[i][1] * 1. / avgcompcost))
    bar_colors.append(colors[process[i][5]][i%2])
  barmax = max(bars[-1][0] + bars[-1][1], barmax)
  ax[0].broken_barh(bars, (iproc + 0.1, 0.8), facecolors = bar_colors,
                    edgecolor = "none")
  if args.labels:
    totfraction = bars[-1][0] + bars[-1][1]
    label = ""
    if nproc > 1:
      label += "rank {0} - ".format(iproc)
    label += "{0:.2f} \% load".format(totfraction * 100.)
    ax[0].text(0.5, iproc + 0.5, label, ha = "center",
               bbox = dict(facecolor = "white", alpha = 0.9))

  # sort and plot the photon cost
  process = sorted(process, key = lambda line: line[2], reverse = True)
  bars = [(0, process[0][2] * 1. / avgphotcost)]
  bar_colors = [colors[process[0][5]][0]]
  for i in range(1, len(process)):
    bars.append((bars[-1][0] + bars[-1][1], process[i][2] * 1. / avgphotcost))
    bar_colors.append(colors[process[i][5]][i%2])
  barmax = max(bars[-1][0] + bars[-1][1], barmax)
  ax[1].broken_barh(bars, (iproc + 0.1, 0.8), facecolors = bar_colors,
                    edgecolor = "none")
  if args.labels:
    totfraction = bars[-1][0] + bars[-1][1]
    label = ""
    if nproc > 1:
      label += "rank {0} - ".format(iproc)
    label += "{0:.2f} \% load".format(totfraction * 100.)
    ax[1].text(0.5, iproc + 0.5, label, ha = "center",
               bbox = dict(facecolor = "white", alpha = 0.9))

  # sort and plot the source cost
  process = sorted(process, key = lambda line: line[3], reverse = True)
  bars = [(0, process[0][3] * 1. / avgsrccost)]
  bar_colors = [colors[process[0][5]][0]]
  for i in range(1, len(process)):
    bars.append((bars[-1][0] + bars[-1][1], process[i][3] * 1. / avgsrccost))
    bar_colors.append(colors[process[i][5]][i%2])
  barmax = max(bars[-1][0] + bars[-1][1], barmax)
  ax[2].broken_barh(bars, (iproc + 0.1, 0.8), facecolors = bar_colors,
                    edgecolor = "none")
  if args.labels:
    totfraction = bars[-1][0] + bars[-1][1]
    label = ""
    if nproc > 1:
      label += "rank {0} - ".format(iproc)
    label += "{0:.2f} \% load".format(totfraction * 100.)
    ax[2].text(0.5, iproc + 0.5, label, ha = "center",
               bbox = dict(facecolor = "white", alpha = 0.9))

# add axis labels
ax[0].set_title("Computational cost")
ax[1].set_title("Photon cost")
ax[2].set_title("Source cost")

# add the perfect load-balancing reference line
for a in ax:
  a.axvline(x = 1.)
  a.set_yticks([])
  a.set_ylim(-0.05 * nproc, 1.05 * nproc)
  a.set_xlim(-0.05 * barmax, 1.05 * barmax)

ax[-1].set_xlabel("Fractional cost")

# finalize and save the plot
pl.tight_layout()
pl.savefig("{0}.png".format(name[:-4]))
