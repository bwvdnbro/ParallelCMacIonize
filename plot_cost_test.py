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

nproc = int(data[:,4].max()) + 1
nthread = int(data[:,5].max()) + 1

print "nproc:", nproc
print "nthread:", nthread

totcompcost = data[:,1].sum()
avgcompcost = totcompcost / (nproc * nthread)
totphotcost = data[:,2].sum()
avgphotcost = totphotcost / (nproc * nthread)
totsrccost = data[:,3].sum()
avgsrccost = totsrccost  / nproc

print "avgcompcost:", avgcompcost
print "avgphotcost:", avgphotcost
print "avgsrccost:", avgsrccost

colors = ["r", "b"]

fig, ax = pl.subplots(3, 1, sharex = True)

for iproc in range(nproc):
  process = data[data[:,4] == iproc]
  for ithread in range(nthread):
    thread = process[process[:,5] == ithread]

    # computational cost
    thread = sorted(thread, key = lambda line: line[1], reverse = True)
    bars = [(0, thread[0][1] * 1. / avgcompcost)]
    bar_colors = [colors[0]]
    for i in range(1, len(thread)):
      bars.append((bars[-1][0] + bars[-1][1], thread[i][1] * 1. / avgcompcost))
      bar_colors.append(colors[i%2])
    ax[0].broken_barh(bars, (iproc * nthread + ithread - 0.4, 0.8),
                      facecolors = bar_colors)
    totfraction = bars[-1][0] + bars[-1][1]
    label = ""
    if nproc > 1:
      label += "rank {0} - ".format(iproc)
    if nthread > 1:
      label += "thread {0} - ".format(ithread)
    label += "{0:.2f} \% load".format(totfraction * 100.)
    ax[0].text(0.5, iproc * nthread + ithread, label, ha = "center",
               bbox = dict(facecolor = "white", alpha = 0.9))

    # photon cost
    thread = sorted(thread, key = lambda line: line[2], reverse = True)
    bars = [(0, thread[0][2] * 1. / avgphotcost)]
    bar_colors = [colors[0]]
    for i in range(1, len(thread)):
      bars.append((bars[-1][0] + bars[-1][1], thread[i][2] * 1. / avgphotcost))
      bar_colors.append(colors[i%2])
    ax[1].broken_barh(bars, (iproc * nthread + ithread - 0.4, 0.8),
                      facecolors = bar_colors)
    totfraction = bars[-1][0] + bars[-1][1]
    label = ""
    if nproc > 1:
      label += "rank {0} - ".format(iproc)
    if nthread > 1:
      label += "thread {0} - ".format(ithread)
    label += "{0:.2f} \% load".format(totfraction * 100.)
    ax[1].text(0.5, iproc * nthread + ithread, label, ha = "center",
               bbox = dict(facecolor = "white", alpha = 0.9))

  # source cost
  process = sorted(process, key = lambda line: line[3], reverse = True)
  bars = [(0, process[0][3] * 1. / avgsrccost)]
  bar_colors = [colors[0]]
  for i in range(1, len(process)):
    bars.append((bars[-1][0] + bars[-1][1], process[i][3] * 1. / avgsrccost))
    bar_colors.append(colors[i%2])
  ax[2].broken_barh(bars, (iproc - 0.4, 0.8), facecolors = bar_colors)
  totfraction = bars[-1][0] + bars[-1][1]
  label = ""
  if nproc > 1:
    label += "rank {0} - ".format(iproc)
  label += "{0:.2f} \% load".format(totfraction * 100.)
  ax[2].text(0.5, iproc, label, ha = "center",
             bbox = dict(facecolor = "white", alpha = 0.9))

ax[0].set_title("Computational cost")
ax[1].set_title("Photon cost")
ax[2].set_title("Source cost")

for a in ax:
  a.axvline(x = 1.)
  a.set_yticks([])
ax[-1].set_xlabel("Fractional cost")
pl.tight_layout()
pl.savefig("{0}.png".format(name[:-4]))
