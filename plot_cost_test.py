import numpy as np
import pylab as pl

data = np.loadtxt("cost_test.txt", dtype = np.uint64)

nproc = int(data[:,2].max()) + 1
nthread = int(data[:,3].max()) + 1

print "nproc:", nproc
print "nthread:", nthread

totcost = data[:,1].sum()
avgcost = totcost / (nproc * nthread)

print "avgcost:", avgcost

colors = ["r", "b"]

fig, ax = pl.subplots(nproc, 1, sharex = True)

for a in ax:
  a.axvline(x = avgcost)
for iproc in range(nproc):
  process = data[data[:,2] == iproc]
  for ithread in range(nthread):
    thread = process[process[:,3] == ithread]
    thread = sorted(thread, key = lambda line: line[1], reverse = True)
    bars = [(0, thread[0][1])]
    bar_colors = [colors[0]]
    texts = ["{0} ({1:.2f})".format(thread[0][0], thread[0][1] * 1. / avgcost)]
    for i in range(1, len(thread)):
      bars.append((bars[-1][0] + bars[-1][1], thread[i][1]))
      bar_colors.append(colors[i%2])
      texts.append("{0} ({1:.2f})".format(
        thread[i][0], thread[i][1] * 1. / avgcost))
    ax[iproc].broken_barh(bars, (ithread - 0.4, 0.8), facecolors = bar_colors)
    for i in range(len(texts)):
      ax[iproc].text(bars[i][0], ithread, texts[i])
pl.tight_layout()
pl.show()
