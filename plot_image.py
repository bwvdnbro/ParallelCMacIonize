import numpy as np
import pylab as pl
import scipy.special as special
import scipy.integrate as integ
import argparse

argparser = argparse.ArgumentParser("Plot the scattering image.")
argparser.add_argument("-f", "--file", action = "store", required = True)
argparser.add_argument("-d", "--direct_light", action = "store_true")
argparser.add_argument("-x", "--nx", action = "store", type = int,
                       default = 101)
argparser.add_argument("-y", "--ny", action = "store", type = int,
                       default = 101)
args = argparser.parse_args()

def integrand(z, kapparho, x2y2):
  return np.exp(-kapparho * (np.sqrt(x2y2 + z**2) - z)) / (x2y2 + z**2)

def analytic(kapparho, x, y, R, dx, dy):
  x2y2 = x**2 + y**2
  if x2y2 < R**2 and x2y2 > 0.:
    weight = (0.25 / np.pi)**2
    zlim = np.sqrt(R**2 - x2y2)
    integral = integ.quad(integrand, -zlim, zlim, args = (kapparho, x2y2))[0]
    return weight * np.exp(-kapparho * zlim) * integral * kapparho * dx * dy
  else:
    if x2y2 == 0. and args.direct_light:
      return 0.25 * np.exp(-kapparho * R) / np.pi
    else:
      return 0.

pl.rcParams["figure.figsize"] = (8, 6)

data = np.memmap(args.file, dtype = np.float64, shape = (args.nx, args.ny),
                 mode = "r")

dataax = pl.subplot2grid((2, 2), (0, 0))
simax = pl.subplot2grid((2, 2), (1, 0))
profax = pl.subplot2grid((2, 2), (0, 1), rowspan = 2)

dataax.imshow(np.log10(data))

data = data.reshape(-1)
xy = np.array(np.meshgrid(np.linspace(-1., 1., args.nx),
                          np.linspace(-1., 1., args.ny))).transpose()
xy = xy.reshape((-1,2))
radius = np.sqrt(xy[:,0]**2 + xy[:,1]**2)

dx = 2. / args.nx
dy = 2. / args.ny

analytic_solution = np.array([analytic(0.1, xyval[0], xyval[1], 1., dx, dy)
                              for xyval in xy])

simax.imshow(np.log10(analytic_solution.reshape((args.nx, args.ny))))

profax.semilogy(radius, data, ".")
profax.semilogy(radius, analytic_solution, "-")

pl.savefig("image.png")
