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

/*! @brief Ascii output of the result. */
//#define ASCII_OUTPUT

#include "DensitySubGrid.hpp"
#include "Hydro.hpp"
#include "Timer.hpp"

#include <fstream>

// global variables, as we need them in the log macro
int MPI_rank, MPI_size;

/**
 * @brief Test for the hydro.
 *
 * @param argc Number of command line arguments.
 * @param argv Command line arguments.
 * @return Exit code: 0 on success.
 */
int main(int argc, char **argv) {

  /// Hydro only
  {
    Hydro hydro;

    double conserved[500];
    double delta_conserved[500];
    double primitives[500];
    double gradients[1500];
    double limiters[1000];

    for (unsigned int i = 0; i < 100; ++i) {
      if (i < 50) {
        primitives[5 * i] = 1.;
        primitives[5 * i + 4] = 1.;
      } else {
        primitives[5 * i] = 0.125;
        primitives[5 * i + 4] = 0.1;
      }
      primitives[5 * i + 1] = 0.;
      primitives[5 * i + 2] = 0.;
      primitives[5 * i + 3] = 0.;

      delta_conserved[5 * i] = 0.;
      delta_conserved[5 * i + 1] = 0.;
      delta_conserved[5 * i + 2] = 0.;
      delta_conserved[5 * i + 3] = 0.;
      delta_conserved[5 * i + 4] = 0.;
      for (unsigned int j = 0; j < 15; ++j) {
        gradients[15 * i + j] = 0.;
      }
      for (unsigned int j = 0; j < 5; ++j) {
        limiters[10 * i + 2 * j] = DBL_MAX;
        limiters[10 * i + 2 * j + 1] = -DBL_MAX;
      }
    }

    for (unsigned int i = 0; i < 100; ++i) {
      hydro.get_conserved_variables(
          primitives[5 * i], &primitives[5 * i + 1], primitives[5 * i + 4],
          0.01, conserved[5 * i], &conserved[5 * i + 1], conserved[5 * i + 4]);
    }

    const double dt = 0.001;
    double dx[3] = {0.01, 0., 0.};
    for (unsigned int istep = 0; istep < 100; ++istep) {
      for (unsigned int i = 0; i < 100; ++i) {
        unsigned int inext = (i + 1) % 100;
        hydro.do_gradient_calculation(
            0, &primitives[5 * i], &primitives[5 * inext], 100.,
            &gradients[15 * i], &limiters[10 * i], &gradients[15 * inext],
            &limiters[10 * inext]);
      }
      for (unsigned int i = 0; i < 100; ++i) {
        hydro.apply_slope_limiter(&primitives[5 * i], &gradients[15 * i],
                                  &limiters[10 * i], dx);
      }
      for (unsigned int i = 0; i < 100; ++i) {
        unsigned int inext = (i + 1) % 100;
        hydro.do_flux_calculation(
            0, &primitives[5 * i], &gradients[15 * i], &primitives[5 * inext],
            &gradients[15 * inext], 0.01, 1., &delta_conserved[5 * i],
            &delta_conserved[5 * inext]);
      }
      for (unsigned int i = 0; i < 500; ++i) {
        conserved[i] += delta_conserved[i] * dt;
      }
      for (unsigned int i = 0; i < 100; ++i) {
        hydro.get_primitive_variables(
            conserved[5 * i], &conserved[5 * i + 1], conserved[5 * i + 4], 100.,
            primitives[5 * i], &primitives[5 * i + 1], primitives[5 * i + 4]);
      }

      for (unsigned int i = 0; i < 100; ++i) {
        delta_conserved[5 * i] = 0.;
        delta_conserved[5 * i + 1] = 0.;
        delta_conserved[5 * i + 2] = 0.;
        delta_conserved[5 * i + 3] = 0.;
        delta_conserved[5 * i + 4] = 0.;
        for (unsigned int j = 0; j < 15; ++j) {
          gradients[15 * i + j] = 0.;
        }
        for (unsigned int j = 0; j < 5; ++j) {
          limiters[10 * i + 2 * j] = DBL_MAX;
          limiters[10 * i + 2 * j + 1] = -DBL_MAX;
        }
      }
    }

    std::ofstream ofile("hydro_only.txt");
    for (unsigned int i = 0; i < 100; ++i) {
      ofile << (i + 0.5) * 0.01 << "\t" << primitives[5 * i] << "\n";
    }
  }

  /// DensitySubGrid hydro
  {
    const double box[6] = {-0.5, -0.5, -0.5, 1., 1., 1.};
    const int ncell[3] = {100, 50, 50};
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

    const double dt = 0.001;
    Hydro hydro;

    test_grid.initialize_conserved_variables(hydro);
    Timer hydro_timer;
    hydro_timer.start();
    for (unsigned int istep = 0; istep < 100; ++istep) {
      // gradient calculations
      test_grid.inner_gradient_sweep(hydro);
      test_grid.outer_gradient_sweep(TRAVELDIRECTION_FACE_X_P, hydro,
                                     test_grid);
      test_grid.outer_gradient_sweep(TRAVELDIRECTION_FACE_Y_P, hydro,
                                     test_grid);
      test_grid.outer_gradient_sweep(TRAVELDIRECTION_FACE_Z_P, hydro,
                                     test_grid);

      // slope limiting
      test_grid.apply_slope_limiter(hydro);

      // flux exchanges
      test_grid.inner_flux_sweep(hydro);
      test_grid.outer_flux_sweep(TRAVELDIRECTION_FACE_X_P, hydro, test_grid);
      test_grid.outer_flux_sweep(TRAVELDIRECTION_FACE_Y_P, hydro, test_grid);
      test_grid.outer_flux_sweep(TRAVELDIRECTION_FACE_Z_P, hydro, test_grid);

      // conserved variable update
      test_grid.update_conserved_variables(dt);

      // primitive variable update
      test_grid.update_primitive_variables(hydro);
    }
    hydro_timer.stop();

    cmac_status("Total time: %g s", hydro_timer.value());

#ifdef ASCII_OUTPUT
    std::ofstream ofile("hydro.txt");
    test_grid.print_intensities(ofile);
#endif
    MemoryMap file("hydro.dat", test_grid.get_output_size());
    test_grid.output_intensities(0, file);
  }

  return 0;
}
