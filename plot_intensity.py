################################################################################
# This file is part of CMacIonize
# Copyright (C) 2017 Bert Vandenbroucke (bert.vandenbroucke@gmail.com)
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
# @file plot_intensity.py
#
# @brief VisIt plot script to plot a Pseudocolor plot of the neutral fractions.
#
# Run with visit -cli -s plot_intensity.py
#
# @author Bert Vandenbroucke (bv7@st-andrews.ac.uk)
#

# First open the file without setting options
# We need to do this to force a metadata server (mdserver) start
# If we don't do this, the SetDefaultFileOpenOptions command below does not work
OpenDatabase("intensities.txt")
# Now set the file open options: we want the first 3 columns of the ASCII file
# to be used as mesh coordinates
fop = {"Lines to skip at beginning of file": 0,
       "Column for X coordinate (or -1 for none)": 0,
       "Column for Y coordinate (or -1 for none)": 1,
       "Column for Z coordinate (or -1 for none)": 2,
       "Data layout": 0,
       "First row has variable names": 0}
SetDefaultFileOpenOptions("PlainText", fop)
# Now open the file again
ReOpenDatabase("intensities.txt")

# Add the Pseudocolor plot of variable 3: the neutral fraction
AddPlot("Pseudocolor", "var03")

# Switch to a logarithmic scale
p = PseudocolorAttributes()
p.scaling = p.Log
SetPlotOptions(p)

# Show the plot
DrawPlots()
