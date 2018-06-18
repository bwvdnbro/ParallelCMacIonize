/*******************************************************************************
 * This file is part of CMacIonize
 * Copyright (C) 2018 Bert Vandenbroucke (bert.vandenbroucke@gmail.com)
 *
 * CMacIonize is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * CMacIonize is distributed in the hope that it will be useful,
 * but WITOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU Affero General Public License for more details.
 *
 * You should have received a copy of the GNU Affero General Public License
 * along with CMacIonize. If not, see <http://www.gnu.org/licenses/>.
 ******************************************************************************/

/**
 * @file testHydro.cpp
 *
 * @brief Test for the hydro.
 *
 * @author Bert Vandenbroucke (bv7@st-andrews.ac.uk)
 */

#include "DensitySubGrid.hpp"
#include "Hydro.hpp"

/**
 * @brief Test for the hydro.
 *
 * @param argc Number of command line arguments.
 * @param argv Command line arguments.
 * @return Exit code: 0 on success.
 */
int main(int argc, char **argv) {

  const double box[6] = {-0.5, -0.5, -0.5, 1., 1., 1.};
  const int ncell[3] = {100, 1, 1};
  const unsigned int tot_ncell = ncell[0] * ncell[1] * ncell[2];
  DensitySubGrid test_grid(box, ncell);

  for (unsigned int i = 0; i < tot_ncell; ++i) {
    double midpoint[3];
    test_grid.get_cell_midpoint(i, midpoint);
    double density, velocity[3], pressure;
    if (midpoint[0] <= 0.) {
      density = 1.;
      velocity[0] = 0.;
      velocity[1] = 0.;
      velocity[2] = 0.;
      pressure = 1.;
    } else {
      density = 0.125;
      velocity[0] = 0.;
      velocity[1] = 0.;
      velocity[2] = 0.;
      pressure = 0.1;
    }
    test_grid.set_primitive_variables(i, density, velocity, pressure);
  }

  Hydro hydro;

  test_grid.initialize_conserved_variables(hydro);

  MemoryMap file("hydro.dat", test_grid.get_output_size());
  test_grid.output_intensities(0, file);

  return 0;
}
