import numpy as np
import matplotlib
matplotlib.use("Agg")
import pylab as pl
import sys

pl.rcParams["text.usetex"] = True
pl.rcParams["figure.figsize"] = (12, 10)
pl.rcParams["font.size"] = 14

name = sys.argv[1]

task_colors = ["b", "r", "g"]
task_names = ["source photon", "photon traversal", "reemission"]

print "Plotting tasks for", name, "..."

data = np.loadtxt(name)

tmin = data[:,2].min()
tmax = data[:,3].max()
tconv = 1. / (tmax - tmin)
nthread = int(data[:,1].max()) + 1
alltime = 0
for i in range(nthread):
  thread = data[data[:,1] == i][:,1:]
  bar = [((task[1] - tmin) * tconv, (task[2] - task[1]) * tconv) \
           for task in thread]
  tottime = np.array([line[1] for line in bar]).sum()
  alltime += tottime
  colors = [task_colors[int(task[3])] for task in thread]
  pl.broken_barh(bar, (i-0.4, 0.8), facecolors = colors, edgecolor = "none")
  pl.text(0.5, i, "{0:.2f} \%".format(tottime * 100.),
          bbox = dict(facecolor='white', alpha=0.9))
for i in range(len(task_colors)):
  pl.plot([], [], color = task_colors[i], label = task_names[i])
pl.legend(loc = "upper center", ncol = len(task_colors))
pl.ylim(-1., nthread * 1.1)
alltime /= nthread
pl.title("Total empty fraction: {0:.2f} \%".format((1. - alltime) * 100.))
pl.gca().set_yticks([])
pl.xlabel("fraction of iteration time")
pl.tight_layout()
pl.savefig("{0}.png".format(name[:-4]))
