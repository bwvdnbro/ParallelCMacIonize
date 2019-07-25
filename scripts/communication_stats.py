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
# @file communication_stats.py
#
# @brief Script to plot the communication load per node for a given message
# output file.
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
    description="Plot communication stats based on a given message file."
)

argparser.add_argument("-n", "--name", action="store", required=True)
argparser.add_argument("-l", "--labels", action="store_true")

args = argparser.parse_args()

# load the file
data = np.loadtxt(args.name)

# colours used for plotting
colors = ["y", "g"]

# find the number of processes
nproc = int(data[:, 0].max()) + 1

# create the plot
fig, ax = pl.subplots(1, 1, sharex=True)

# loop over the processes
totnum = 0
for iproc in range(nproc):
    # count the sends and receives on this process
    numsend = len(data[(data[:, 0] == iproc) & (data[:, 2] == 0)])
    numrecv = len(data[(data[:, 0] == iproc) & (data[:, 2] == 1)])

    # add to the total
    totnum += numsend + numrecv

    # plot the contributions from this process
    ax.broken_barh(
        [(0, numsend), (numsend, numrecv)],
        (iproc + 0.1, 0.8),
        facecolors=colors,
        edgecolor="none",
    )

    # optionally add labels
    if args.labels:
        ax.text(
            0.5 * (numsend + numrecv),
            iproc + 0.65,
            "rank {0}".format(iproc),
            ha="center",
            bbox=dict(facecolor="white", alpha=0.9),
        )
        ax.text(
            0.5 * numsend,
            iproc + 0.3,
            "{0} sends".format(numsend),
            ha="center",
            bbox=dict(facecolor="white", alpha=0.9),
        )
        ax.text(
            numsend + 0.5 * numrecv,
            iproc + 0.3,
            "{0} receives".format(numrecv),
            ha="center",
            bbox=dict(facecolor="white", alpha=0.9),
        )

# clean up axis
ax.set_title("total number of communications: {0}".format(totnum))
ax.set_yticks([])
ax.set_xlabel("number of communications")

# finalize and save plot
pl.tight_layout()
pl.savefig("{0}.png".format(args.name[:-4]))
