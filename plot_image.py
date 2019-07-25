import numpy as np
import pylab as pl
import scipy.integrate as integ
import argparse

# parse command line arguments
argparser = argparse.ArgumentParser("Plot the scattering image.")
argparser.add_argument("-f", "--file", action="store", required=True)
argparser.add_argument("-d", "--direct_light", action="store_true")
argparser.add_argument("-x", "--nx", action="store", type=int, default=101)
argparser.add_argument("-y", "--ny", action="store", type=int, default=101)
args = argparser.parse_args()

# helper functions
def integrand(z, kapparho, x2y2):
    return np.exp(-kapparho * (np.sqrt(x2y2 + z ** 2) - z)) / (x2y2 + z ** 2)


def analytic(kapparho, r, R, dx, dy):
    x2y2 = r ** 2
    if x2y2 < R ** 2 and x2y2 > 0.0:
        weight = (0.25 / np.pi) ** 2
        zlim = np.sqrt(R ** 2 - x2y2)
        integral = integ.quad(integrand, -zlim, zlim, args=(kapparho, x2y2))[0]
        return weight * np.exp(-kapparho * zlim) * integral * kapparho * dx * dy
    else:
        if x2y2 == 0.0 and args.direct_light:
            return 0.25 * np.exp(-kapparho * R) / np.pi
        else:
            return 0.0


# open the image file
data = np.memmap(
    args.file, dtype=np.float64, shape=(args.nx, args.ny), mode="r"
)

profile_data = data.reshape(-1)
xy = np.array(
    np.meshgrid(
        np.linspace(-1.0, 1.0, args.nx), np.linspace(-1.0, 1.0, args.ny)
    )
).transpose()
xy = xy.reshape((-1, 2))
profile_radius = np.sqrt(xy[:, 0] ** 2 + xy[:, 1] ** 2)

image_xy = xy.reshape((args.nx, args.ny, 2))
image_data = data.reshape((args.nx, args.ny))

ra = np.linspace(0.0, 1.0, 100)
dx = 2.0 / args.nx
dy = 2.0 / args.ny
profile_analytic = np.array([analytic(0.01, r, 1.0, dx, dy) for r in ra])
image_analytic = np.repeat(profile_analytic.reshape((1, -1)), 100, axis=0)
phia = np.linspace(0.0, 2.0 * np.pi, 100)
analytic_xy = np.zeros((100, 100, 3))
for i in range(100):
    for j in range(100):
        analytic_xy[i, j, 0] = ra[j] * np.cos(phia[i])
        analytic_xy[i, j, 1] = ra[j] * np.sin(phia[i])

# make the plot

pl.rcParams["figure.figsize"] = (7, 6)
pl.rcParams["text.usetex"] = True

dataax = pl.subplot2grid((2, 2), (0, 0))
simax = pl.subplot2grid((2, 2), (1, 0))
profax = pl.subplot2grid((2, 2), (0, 1), rowspan=2)

data_min = profile_data[profile_data > 0.0].min()
data_max = profile_data[profile_data > 0.0].max()
analytic_min = profile_analytic[profile_analytic > 0.0].min()
analytic_max = profile_analytic[profile_analytic > 0.0].max()

vmin = np.log10(min(data_min, analytic_min))
vmax = np.log10(max(data_max, analytic_max))

dataax.contourf(
    image_xy[:, :, 0],
    image_xy[:, :, 1],
    np.log10(image_data),
    500,
    vmin=vmin,
    vmax=vmax,
    extend="both",
)

simax.contourf(
    analytic_xy[:, :, 0],
    analytic_xy[:, :, 1],
    np.log10(image_analytic),
    500,
    vmin=vmin,
    vmax=vmax,
    extend="both",
)
simax.contourf

profax.semilogy(profile_radius, profile_data, ".", label="MC")
profax.semilogy(ra, profile_analytic, "-", label="analytic")

dataax.set_title("MC")
simax.set_title("analytic")
profax.legend(loc="best")

dataax.set_aspect("equal")
simax.set_aspect("equal")

dataax.set_ylabel("$y$")
simax.set_xlabel("$x$")
simax.set_ylabel("$y$")
profax.set_xlabel("$\sqrt{x^2+y^2}$")
profax.set_ylabel("pixel count")

pl.tight_layout()
pl.savefig("image.png")
