import numpy as np
import pylab as pl
import scipy.special as special
import scipy.integrate as integ

def integrand(x, tau, rho):
  return np.exp(-tau * (np.sqrt(rho**2 + x**2) - x)) / (rho**2 + x**2)

def analytic(tau, rho, R):
  if rho < R:
    zlim = np.sqrt(R**2 - rho**2)
    integral = integ.quad(integrand, -zlim, zlim, args = (tau, rho))[0]
    return np.exp(-tau * zlim) * integral * 5.e4
  else:
    return 0.

pl.rcParams["figure.figsize"] = (8, 6)

data = np.memmap("image.dat", dtype = np.float64, shape = (100, 100),
                 mode = "r")

dataax = pl.subplot2grid((2, 2), (0, 0))
simax = pl.subplot2grid((2, 2), (1, 0))
profax = pl.subplot2grid((2, 2), (0, 1), rowspan = 2)

dataax.imshow(np.log10(data))

data = data.reshape(-1)
xy = np.array(np.meshgrid(np.linspace(-1., 1., 100),
                          np.linspace(-1., 1., 100))).transpose()
xy = xy.reshape((-1,2))
radius = np.sqrt(xy[:,0]**2 + xy[:,1]**2)

analytic_solution = np.array([analytic(10., rho, 1.) for rho in radius])

simax.imshow(np.log10(analytic_solution.reshape((100, 100))))

profax.semilogy(radius, data, ".")
profax.semilogy(radius, analytic_solution, "-")

pl.savefig("image.png")
