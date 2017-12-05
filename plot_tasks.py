import numpy as np
import matplotlib
matplotlib.use("Agg")
import pylab as pl
import sys

name = sys.argv[1]

data = np.loadtxt(name)

for i in range(16):
  thread = data[data[:,0] == i]
  bar = [(task[1], task[2] - task[1]) for task in thread]
  colors = ["b" for task in thread]
  pl.broken_barh(bar, (i-0.4, 0.8), facecolors = colors)
pl.savefig("{0}.png".format(name[:-4]))
