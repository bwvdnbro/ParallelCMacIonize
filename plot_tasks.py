import numpy as np
import matplotlib
matplotlib.use("Agg")
import pylab as pl
import sys

pl.rcParams["text.usetex"] = True
pl.rcParams["figure.figsize"] = (12, 10)

name = sys.argv[1]

print "Plotting tasks for", name, "..."

data = np.loadtxt(name)

tmin = data[:,1].min()
tmax = data[:,2].max()
tconv = 1. / (tmax - tmin)
nthread = int(data[:,0].max()) + 1
alltime = 0
for i in range(nthread):
  thread = data[data[:,0] == i]
  bar = [((task[1] - tmin) * tconv, (task[2] - task[1]) * tconv) \
           for task in thread]
  tottime = np.array([line[1] for line in bar]).sum()
  alltime += tottime
  colors = ["b" for task in thread]
  pl.broken_barh(bar, (i-0.4, 0.8), facecolors = colors, edgecolor = "none")
  pl.text(0.5, i, "{0:.2f} \%".format(tottime * 100.),
          bbox = dict(facecolor='white', alpha=0.9))
alltime /= nthread
pl.title("Total empty fraction: {0:.2f} \%".format((1. - alltime) * 100.))
pl.gca().set_yticks([])
pl.xlabel("fraction of iteration time")
pl.tight_layout()
pl.savefig("{0}.png".format(name[:-4]))
