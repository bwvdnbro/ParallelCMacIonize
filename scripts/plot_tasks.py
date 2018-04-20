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
# @file plot_tasks.py
#
# @brief Script to plot the task plot for a given file with task output.
#
# @author Bert Vandenbroucke (bv7@st-andrews.ac.uk)
##

# import modules
import numpy as np
import matplotlib
matplotlib.use("Agg")
import pylab as pl
import sys
import argparse

# parse the command line arguments
argparser = argparse.ArgumentParser(
  description = "Plot task plot based on a given task output file.")

argparser.add_argument("-n", "--name", action = "store", required = True)
argparser.add_argument("-m", "--messagefile", action = "store")
argparser.add_argument("-l", "--labels", action = "store_true")

args = argparser.parse_args(sys.argv[1:])

name = args.name

# change the default matplotlib settings to get nicer plots
pl.rcParams["text.usetex"] = True
pl.rcParams["figure.figsize"] = (12, 10)
pl.rcParams["font.size"] = 14

# name labels and colours for the various task types
# add extra colours and labels here if new tasks are created
task_colors = ["b", "r", "c", "y", "g"]
task_names = ["source photon", "photon traversal", "reemission", "send",
              "receive"]

# load the data
print "Plotting tasks for", name, "..."
data = np.loadtxt(name)

task_flags = [len(data[data[:,4] == task]) > 0 \
              for task in range(len(task_names))]

# get information about the system
nthread = int(data[:,1].max()) + 1
nproc = int(data[:,0].max()) + 1

# get the minimum and maximum time stamp and compute the time to fraction
# conversion factor for each node
tmin = np.zeros(nproc)
tmax = np.zeros(nproc)
tconv = np.zeros(nproc)
for iproc in range(nproc):
  procline = data[(data[:,0] == iproc) & (data[:,4] == -1)]
  if len(procline) > 1:
    print "Too many node information lines!"
    exit()
  tmin[iproc] = procline[0,2]
  tmax[iproc] = procline[0,3]
  tconv[iproc] = 1. / (tmax[iproc] - tmin[iproc])

## make the plot

fig, ax = pl.subplots(1, 1, sharex = True)

# first plot the MPI communications (if requested)
if args.messagefile:
  mname = args.messagefile

  # load the MPI communication data dump
  mdata = np.loadtxt(mname)

  # sort on time
  mdata = mdata[mdata[:, 5].argsort()]

  # convert time stamps to fractional time
  for iproc in range(nproc):
    mdata[mdata[:, 0] == iproc, 5] = \
      (mdata[mdata[:, 0] == iproc, 5] - tmin[iproc]) * tconv[iproc]

  # get the number of tags
  ntag = int(mdata[:, 4].max()) + 1

  nkey = 0
  # iterate over the rank combinations...
  for iproc in range(nproc):
    for jproc in range(iproc) + range(iproc + 1, nproc):
      # ...and tags
      for itag in range(ntag):
        # send events
        sends = (mdata[(mdata[:, 2] == 0)     & # send events
                       (mdata[:, 0] == iproc) & # from rank iproc
                       (mdata[:, 3] == jproc) & # to rank jproc
                       (mdata[:, 4] == itag)])  # with tag itag
        recvs = (mdata[(mdata[:, 2] == 1)     & # receive events
                       (mdata[:, 3] == iproc) & # from rank iproc
                       (mdata[:, 0] == jproc) & # on rank jproc
                       (mdata[:, 4] == itag)])  # with tag itag

        if not len(sends) == len(recvs):
          print "Sizes do not match (sends: {0}, receives: {1})!".format(
            len(sends), len(recvs))
          exit()

        # make a quiver plot with lines that start where the communication is
        # started, and end where it is received
        x = sends[:, 5]
        y = sends[:, 0] * nthread + sends[:, 1]
        u = recvs[:, 5] - sends[:, 5]
        v = (recvs[:, 0] - sends[:, 0]) * nthread + recvs[:, 1] - sends[:, 1]

        q = ax.quiver(x, y, u, v, angles = "xy", scale_units = "xy",
                      scale = 1., width = 0.0001)

# now plot the tasks
alltime = 0
# loop over the processes
for iproc in range(nproc):
  # filter out the data for this process
  process = data[data[:,0] == iproc]

  # loop over the threads
  for i in range(nthread):
    # filter out the data for this thread
    thread = process[(process[:,1] == i) & (process[:,4] != -1)][:,1:]

    # create the task plot
    bar = [((task[1] - tmin[iproc]) * tconv[iproc],
            (task[2] - task[1]) * tconv[iproc]) \
             for task in thread]
    tottime = np.array([line[1] for line in bar]).sum()
    alltime += tottime
    colors = [task_colors[int(task[3])] for task in thread]
    ax.broken_barh(bar, (iproc * nthread + i - 0.4, 0.8), facecolors = colors,
                   edgecolor = "none")
    # optionally add labels
    if args.labels:
      # status text
      label = ""
      if nproc > 1:
        label += "rank {0} - ".format(iproc)
      if nthread > 1:
        label += "thread {0} - ".format(i)
      label += "{0:.2f} \% load".format(tottime * 100.)
      ax.text(0.5, iproc * nthread + i + 0.2, label, ha = "center",
              bbox = dict(facecolor='white', alpha=0.9))
      # per task fraction text
      label = ""
      for itask in range(len(task_colors)):
        if task_flags[itask]:
          tottime = np.array([(task[2] - task[1]) * tconv[iproc]
                              for task in thread if task[3] == itask]).sum()
          label += "{0}: {1:.2f} \% - ".format(
            task_names[itask], tottime * 100.)
      ax.text(0.5, iproc * nthread + i - 0.2, label[:-2], ha = "center",
              bbox = dict(facecolor='white', alpha=0.9))

# add empty blocks for the legend
for i in range(len(task_colors)):
  if task_flags[i]:
    ax.plot([], [], color = task_colors[i], label = task_names[i])

# add the legend and clean up the axes
ax.legend(loc = "upper center", ncol = len(task_colors))
ax.set_ylim(-1., nproc * nthread * 1.1)
ax.set_yticks([])
ax.set_xlim(-0.05, 1.05)

# get the total idle time and add information to the title
alltime /= (nthread * nproc)
ax.set_title("Total empty fraction: {0:.2f} \%".format((1. - alltime) * 100.))
ax.set_xlabel("fraction of iteration time")

for iproc in range(1, nproc):
  ax.axhline(y = iproc * nthread - 0.5, linestyle = "-", linewidth = 0.8,
             color = "k")

# finalize and save the plot
pl.tight_layout()
pl.savefig("{0}.png".format(name[:-4]))
