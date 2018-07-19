import numpy as np
import pylab as pl
import scipy.special as special

pl.rcParams["figure.figsize"] = (8, 5)

data = np.memmap("image.dat", dtype = np.float64, shape = (100, 100),
                 mode = "r")

fig, ax = pl.subplots(1, 2)

ax[0].imshow(np.log10(data))

data = data.reshape(-1)
xy = np.array(np.meshgrid(np.linspace(-0.5, 0.5, 100),
                          np.linspace(-0.5, 0.5, 100))).transpose()
xy = xy.reshape((-1,2))
radius = np.sqrt(xy[:,0]**2 + xy[:,1]**2)
r = np.linspace(1.e-4, 1., 1000)

ax[1].semilogy(r, special.exp1(r * 10.), "-")
ax[1].semilogy(radius, data, ".")

pl.show()
