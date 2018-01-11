import numpy as np
import matplotlib
matplotlib.use("Agg")
import pylab as pl
import sys

pl.rcParams["text.usetex"] = True
pl.rcParams["figure.figsize"] = (12, 10)
pl.rcParams["font.size"] = 14

name = "cost_test.txt"
if len(sys.argv) > 1:
  name = sys.argv[1]

print "Plotting", name, "..."
data = np.loadtxt(name, dtype = np.uint64)

nproc = int(data[:,2].max()) + 1
nthread = int(data[:,3].max()) + 1

print "nproc:", nproc
print "nthread:", nthread

totcost = data[:,1].sum()
avgcost = totcost / (nproc * nthread)

print "avgcost:", avgcost

colors = ["r", "b"]

fig, ax = pl.subplots(nproc, 1, sharex = True)

if nproc == 1:
  ax = [ax]
maxtext = 0.1
for iproc in range(nproc):
  process = data[data[:,2] == iproc]
  for ithread in range(nthread):
    thread = process[process[:,3] == ithread]
    thread = sorted(thread, key = lambda line: line[1], reverse = True)
    bars = [(0, thread[0][1] * 1. / avgcost)]
    bar_colors = [colors[0]]
    textpos = [0.05]
    texts = ["{0} ({1:.2f})".format(thread[0][0], thread[0][1] * 1. / avgcost)]
    for i in range(1, len(thread)):
      bars.append((bars[-1][0] + bars[-1][1], thread[i][1] * 1. / avgcost))
      bar_colors.append(colors[i%2])
      textpos.append(textpos[-1] + 0.1)
      maxtext = max(maxtext, textpos[-1] + 0.05)
      texts.append("{0} ({1:.2f})".format(
        thread[i][0], thread[i][1] * 1. / avgcost))
    ax[iproc].broken_barh(bars, (ithread - 0.5, 0.4), facecolors = bar_colors)
#    for i in range(len(texts)):
#      ax[iproc].text(textpos[i], ithread, texts[i],
#                     horizontalalignment = "center",
#                     bbox = dict(facecolor = bar_colors[i], edgecolor = "none"))

for a in ax:
  a.axvline(x = 1.)
  a.set_yticks([])
ax[-1].set_xlabel("Fractional cost")
#ax[-1].set_xlim(0., max(ax[-1].get_xlim()[1], maxtext))
pl.tight_layout()
pl.savefig("{0}.png".format(name[:-4]))
