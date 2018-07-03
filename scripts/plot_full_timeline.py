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
# @file plot_full_timeline.py
#
# @brief Script to plot the full task plot timeline for the entire simulation.
#
# @author Bert Vandenbroucke (bv7@st-andrews.ac.uk)
##

# import modules
import numpy as np
import matplotlib
matplotlib.use("Agg")
import pylab as pl
from matplotlib.ticker import AutoMinorLocator
import glob
import argparse

argparser = argparse.ArgumentParser(
  description = "Plot the full task plot timeline for the current directory")

argparser.add_argument("-l", "--labels", action = "store_true")
argparser.add_argument("-p", "--prefix", action = "store", default = "tasks")

args = argparser.parse_args()

prefix = args.prefix

# change the default matplotlib settings for nicer plots
pl.rcParams["text.usetex"] = True
pl.rcParams["figure.figsize"] = (12, 10)
pl.rcParams["font.size"] = 14

# colours and labels for task types
# change these if a new task type is created
task_names = ["source photon", "photon traversal", "reemission", "send",
              "receive", "gradsweep internal", "gradsweep neighbour",
              "gradsweep boundary", "slope limiter", "predict primitives",
              "fluxsweep internal", "fluxsweep neighbour", "fluxsweep boundary",
              "update conserved", "update primitives"]
task_colors = \
  pl.cm.ScalarMappable(cmap = "gist_ncar").to_rgba(
    np.linspace(0., 1., len(task_names)))

# get the start and end CPU cycle count for each process to normalize the
# time line
ptime = np.loadtxt("program_time.txt")
if len(ptime.shape) > 1:
  ptime = sorted(ptime, key = lambda line: line[0])
  tmin = [line[1] for line in ptime]
  tconv = [1. / (line[2] - line[1]) for line in ptime]
  total_time = np.array([line[3] for line in ptime]).mean()
else:
  tmin = [ptime[1]]
  tconv = [1. / (ptime[2] - ptime[1])]
  total_time = ptime[3]

##
# @brief Add the tasks from the file with the given name to the plot.
#
# @param name Name of a task output file.
##
def plot_file(name):
  print "Reading tasks from", name, "..."

  # load the data
  data = np.loadtxt(name)

  # check which tasks are present in this file
  task_flags = [len(data[data[:,4] == task]) > 0 \
                for task in range(len(task_names))]

  # get system information
  nthread = int(data[:,1].max()) + 1
  nproc = int(data[:,0].max()) + 1
  # loop over the processes
  for iproc in range(nproc):
    # filter out data per process
    process = data[(data[:,0] == iproc) & (data[:,4] != -1)]
    # loop over the threads
    for i in range(nthread):
      # filter out data per thread
      thread = process[process[:,1] == i][:,1:]
      # plot the tasks
      bar = [((task[1] - tmin[iproc]) * tconv[iproc] * total_time,
              (task[2] - task[1]) * tconv[iproc] * total_time)
             for task in thread]
      colors = [task_colors[int(task[3])] for task in thread]
      pl.broken_barh(bar, (iproc * nthread + i-0.4, 0.8), facecolors = colors,
                     edgecolor = "none")
  # return information required for the global layout of the task plot
  return nproc, nthread, task_flags

# variables to store global task plot information in
nthread = 0
nproc = 0
task_flags = np.zeros(len(task_colors), dtype = bool)
# loop over the task files in the current working directory
files = sorted(glob.glob(prefix + "_??.txt"))
for name in files:
  # plot the file
  nproc_this, nthread_this, this_task_flags = plot_file(name)
  # update the global information
  nthread = max(nthread, nthread_this)
  nproc = max(nproc, nproc_this)
  for i in range(len(task_flags)):
    task_flags[i] = task_flags[i] or this_task_flags[i]

# add dummy tasks for the legend
for i in range(len(task_colors)):
  if task_flags[i]:
    pl.plot([], [], color = task_colors[i], label = task_names[i])

# add node and thread labels
if args.labels and (nproc > 1 or nthread > 1):
  for iproc in range(nproc):
    for ithread in range(nthread):
      label = ""
      if nproc > 1:
        label += "rank {0} - ".format(iproc)
      if nthread > 1:
        label += "thread {0} - ".format(ithread)
      xlim = pl.gca().get_xlim()
      xpos = 0.5 * (xlim[0] + xlim[1])
      pl.text(xpos, iproc * nthread + ithread, label[:-2], ha = "center",
              bbox = dict(facecolor = "white", alpha = 0.9))

# plot start and end time for overhead reference
pl.gca().axvline(x = 0., linestyle = "--", linewidth = 0.8, color = "k")
pl.gca().axvline(x = total_time, linestyle = "--", linewidth = 0.8, color = "k")
pl.gca().xaxis.set_minor_locator(AutoMinorLocator())

pl.xlim(-0.05 * total_time, 1.05 * total_time)

# separate nodes with a horizontal line
for iproc in range(1, nproc):
  pl.gca().axhline(y = iproc * nthread - 0.5, linestyle = "-", linewidth = 0.8,
                   color = "k")

# add legend and clean up axes
pl.legend(loc = "upper center", ncol = min(3, len(task_colors)))
pl.ylim(-1., nproc * nthread * 1.1)
pl.gca().set_yticks([])
pl.xlabel("Simulation time (s)")

# finalize and save plot
pl.tight_layout()
pl.savefig("full_timeline.png")
