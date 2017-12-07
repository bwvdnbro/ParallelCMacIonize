import numpy as np
import matplotlib
matplotlib.use("Agg")
import pylab as pl
import glob

pl.rcParams["text.usetex"] = True
pl.rcParams["figure.figsize"] = (12, 10)
pl.rcParams["font.size"] = 14

task_colors = ["b", "r"]
task_names = ["source photon", "photon traversal"]

def plot_file(name):
  print "Plotting tasks for", name, "..."

  data = np.loadtxt(name)

  nthread = int(data[:,0].max()) + 1
  for i in range(nthread):
    thread = data[data[:,0] == i]
    bar = [(task[1], task[2] - task[1]) for task in thread]
    colors = [task_colors[int(task[3])] for task in thread]
    pl.broken_barh(bar, (i-0.4, 0.8), facecolors = colors, edgecolor = "none")
  return nthread

nthread = 0
for name in sorted(glob.glob("tasks_??.txt")):
  nthread = max(nthread, plot_file(name))
for i in range(len(task_colors)):
  pl.plot([], [], color = task_colors[i], label = task_names[i])
pl.legend(loc = "upper center", ncol = len(task_colors))
pl.ylim(-1., nthread * 1.1)
pl.gca().set_yticks([])
pl.xlabel("CPU cycle")
pl.tight_layout()
pl.savefig("full_timeline.png")
