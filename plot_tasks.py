import numpy as np
import matplotlib
matplotlib.use("Agg")
import pylab as pl
import sys

pl.rcParams["text.usetex"] = True
pl.rcParams["figure.figsize"] = (12, 10)
pl.rcParams["font.size"] = 14

name = sys.argv[1]

task_colors = ["b", "r", "g", "y"]
task_names = ["source photon", "photon traversal", "reemission", "send"]

print "Plotting tasks for", name, "..."

data = np.loadtxt(name)

task_flags = [len(data[data[:,4] == task]) > 0 \
              for task in range(len(task_names))]

tmin = data[:,2].min()
tmax = data[:,3].max()
tconv = 1. / (tmax - tmin)

nthread = int(data[:,1].max()) + 1
nproc = int(data[:,0].max()) + 1

fig, ax = pl.subplots(1, 1, sharex = True)

if len(sys.argv) > 2:
  mname = sys.argv[2]

  mdata = np.loadtxt(mname)

  # sort on time
  mdata = mdata[mdata[:, 5].argsort()]

  # convert time stamps to fractional time
  mdata[:, 5] = (mdata[:, 5] - tmin) * tconv

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

        x = sends[:, 5]
        y = sends[:, 0] * nthread + sends[:, 1]
        u = recvs[:, 5] - sends[:, 5]
        v = (recvs[:, 0] - sends[:, 0]) * nthread + recvs[:, 1] - sends[:, 1]

        q = ax.quiver(x, y, u, v, angles = "xy", scale_units = "xy",
                      scale = 1., width = 0.0001, alpha = 0.2)

alltime = 0
for iproc in range(nproc):
  process = data[data[:,0] == iproc]
  for i in range(nthread):
    thread = process[process[:,1] == i][:,1:]
    bar = [((task[1] - tmin) * tconv, (task[2] - task[1]) * tconv) \
             for task in thread]
    tottime = np.array([line[1] for line in bar]).sum()
    alltime += tottime
    colors = [task_colors[int(task[3])] for task in thread]
    ax.broken_barh(bar, (iproc * nthread + i - 0.4, 0.8), facecolors = colors,
                   edgecolor = "none")
    label = ""
    if nproc > 1:
      label += "rank {0} - ".format(iproc)
    if nthread > 1:
      label += "thread {0} -".format(i)
    label += "{0:.2f} \% load".format(tottime * 100.)
    ax.text(0.5, iproc * nthread + i, label, ha = "center",
            bbox = dict(facecolor='white', alpha=0.9))

for i in range(len(task_colors)):
  if task_flags[i]:
    ax.plot([], [], color = task_colors[i], label = task_names[i])

ax.legend(loc = "upper center", ncol = len(task_colors))
ax.set_ylim(-1., nproc * nthread * 1.1)
ax.set_yticks([])
ax.set_xlim(-0.05, 1.05)
alltime /= (nthread * nproc)
ax.set_title("Total empty fraction: {0:.2f} \%".format((1. - alltime) * 100.))
ax.set_xlabel("fraction of iteration time")

pl.tight_layout()
pl.savefig("{0}.png".format(name[:-4]))
