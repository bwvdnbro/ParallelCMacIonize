import numpy as np
import matplotlib
matplotlib.use("Agg")
import pylab as pl
import glob

pl.rcParams["text.usetex"] = True
pl.rcParams["figure.figsize"] = (12, 10)
pl.rcParams["font.size"] = 14

task_colors = ["b", "r", "c", "y", "g"]
task_names = ["source photon", "photon traversal", "reemission", "send",
              "receive"]

def plot_file(name):
  print "Reading tasks from", name, "..."

  data = np.loadtxt(name)

  task_flags = [len(data[data[:,4] == task]) > 0 \
                for task in range(len(task_names))]

  nthread = int(data[:,1].max()) + 1
  nproc = int(data[:,0].max()) + 1
  for iproc in range(nproc):
    process = data[data[:,0] == iproc]
    for i in range(nthread):
      thread = process[process[:,1] == i][:,1:]
      bar = [(task[1], task[2] - task[1]) for task in thread]
      colors = [task_colors[int(task[3])] for task in thread]
      pl.broken_barh(bar, (iproc * nthread + i-0.4, 0.8), facecolors = colors,
                     edgecolor = "none")
  return nproc, nthread, task_flags

nthread = 0
nproc = 0
task_flags = np.zeros(len(task_colors), dtype = bool)
for name in sorted(glob.glob("tasks_??.txt")):
  nproc_this, nthread_this, this_task_flags = plot_file(name)
  nthread = max(nthread, nthread_this)
  nproc = max(nproc, nproc_this)
  for i in range(len(task_flags)):
    task_flags[i] = task_flags[i] or this_task_flags[i]
for i in range(len(task_colors)):
  if task_flags[i]:
    pl.plot([], [], color = task_colors[i], label = task_names[i])
if nproc > 1 or nthread > 1:
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
pl.legend(loc = "upper center", ncol = len(task_colors))
pl.ylim(-1., nproc * nthread * 1.1)
pl.gca().set_yticks([])
pl.xlabel("CPU cycle")
pl.tight_layout()
pl.savefig("full_timeline.png")
