################################################################################
# This file is part of CMacIonize
# Copyright (C) 2018 Bert Vandenbroucke (bert.vandenbroucke@gmail.com)
#
# CMacIonize is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# CMacIonize is distributed in the hope that it will be useful,
# but WITOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with CMacIonize. If not, see <http://www.gnu.org/licenses/>.
################################################################################

##
# @file mpi_stats.py
#
# @brief Script to output statistics about the MPI send and receive tasks.
#
# @author Bert Vandenbroucke (bv7@st-andrews.ac.uk)
##

import numpy as np
import glob

##
# @brief Print the relative standard deviation, minimum and maximum for the
# given array w.r.t. to its mean value.
#
# @param x Array.
##
def stats(x):
    meanx = x.mean()
    stdx = x.std()
    minx = x.min()
    maxx = x.max()

    print("std:", stdx / meanx)
    print("min:", minx / meanx)
    print("max:", maxx / meanx)


# loop over all task files
for f in sorted(glob.glob("tasks_??.txt")):
    print("MPI statistics for", f)

    # load the data
    data = np.loadtxt(f)

    # filter out send and receive tasks
    send = data[data[:, 4] == 3]
    recv = data[data[:, 4] == 4]

    # convert task data to task durations in CPU cycles
    send = send[:, 3] - send[:, 2]
    recv = recv[:, 3] - recv[:, 2]

    # print statistics
    print("send:")
    stats(send)
    print("recv:")
    stats(recv)
