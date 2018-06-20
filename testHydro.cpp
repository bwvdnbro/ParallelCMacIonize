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

/*! @brief Second order hydro scheme. */
#define SECOND_ORDER

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
    const double box1[6] = {-0.5, -0.25, -0.25, 0.5, 0.5, 0.5};
    const double box2[6] = {0., -0.25, -0.25, 0.5, 0.5, 0.5};
    const int ncell[3] = {50, 50, 50};
    const unsigned int tot_ncell = ncell[0] * ncell[1] * ncell[2];
    DensitySubGrid test_grid1(box1, ncell);
    DensitySubGrid test_grid2(box2, ncell);

    for (unsigned int i = 0; i < tot_ncell; ++i) {
      double density, velocity[3], pressure;
      density = 1.;
      velocity[0] = 0.;
      velocity[1] = 0.;
      velocity[2] = 0.;
      pressure = 1.;
      test_grid1.set_primitive_variables(i, density, velocity, pressure);

      density = 0.125;
      velocity[0] = 0.;
      velocity[1] = 0.;
      velocity[2] = 0.;
      pressure = 0.1;
      test_grid2.set_primitive_variables(i, density, velocity, pressure);
    }

    const double dt = 0.001;
    Hydro hydro;
    HydroBoundary boundary;

    test_grid1.initialize_conserved_variables(hydro);
    test_grid2.initialize_conserved_variables(hydro);
    Timer hydro_timer;
    hydro_timer.start();
    for (unsigned int istep = 0; istep < 100; ++istep) {

#ifdef SECOND_ORDER
      // gradient calculations
      test_grid1.inner_gradient_sweep(hydro);
      test_grid1.outer_ghost_gradient_sweep(TRAVELDIRECTION_FACE_X_N, hydro,
                                            boundary);
      test_grid1.outer_gradient_sweep(TRAVELDIRECTION_FACE_X_P, hydro,
                                      test_grid2);
      test_grid1.outer_gradient_sweep(TRAVELDIRECTION_FACE_Y_P, hydro,
                                      test_grid1);
      test_grid1.outer_gradient_sweep(TRAVELDIRECTION_FACE_Z_P, hydro,
                                      test_grid1);
      test_grid2.inner_gradient_sweep(hydro);
      test_grid2.outer_ghost_gradient_sweep(TRAVELDIRECTION_FACE_X_P, hydro,
                                            boundary);
      test_grid2.outer_gradient_sweep(TRAVELDIRECTION_FACE_Y_P, hydro,
                                      test_grid2);
      test_grid2.outer_gradient_sweep(TRAVELDIRECTION_FACE_Z_P, hydro,
                                      test_grid2);

      // slope limiting
      test_grid1.apply_slope_limiter(hydro);
      test_grid2.apply_slope_limiter(hydro);

      // second order time prediction
      test_grid1.predict_primitive_variables(hydro, 0.5 * dt);
      test_grid2.predict_primitive_variables(hydro, 0.5 * dt);
#endif

      // flux exchanges
      test_grid1.inner_flux_sweep(hydro);
      test_grid1.outer_ghost_flux_sweep(TRAVELDIRECTION_FACE_X_N, hydro,
                                        boundary);
      test_grid1.outer_flux_sweep(TRAVELDIRECTION_FACE_X_P, hydro, test_grid2);
      test_grid1.outer_flux_sweep(TRAVELDIRECTION_FACE_Y_P, hydro, test_grid1);
      test_grid1.outer_flux_sweep(TRAVELDIRECTION_FACE_Z_P, hydro, test_grid1);
      test_grid2.inner_flux_sweep(hydro);
      test_grid2.outer_ghost_flux_sweep(TRAVELDIRECTION_FACE_X_P, hydro,
                                        boundary);
      test_grid2.outer_flux_sweep(TRAVELDIRECTION_FACE_Y_P, hydro, test_grid2);
      test_grid2.outer_flux_sweep(TRAVELDIRECTION_FACE_Z_P, hydro, test_grid2);

      // conserved variable update
      test_grid1.update_conserved_variables(dt);
      test_grid2.update_conserved_variables(dt);

      // primitive variable update
      test_grid1.update_primitive_variables(hydro);
      test_grid2.update_primitive_variables(hydro);
    }
    hydro_timer.stop();

    cmac_status("Total time: %g s", hydro_timer.value());

#ifdef ASCII_OUTPUT
    std::ofstream ofile("hydro.txt");
    test_grid1.print_intensities(ofile);
    test_grid2.print_intensities(ofile);
#endif
    MemoryMap file("hydro.dat", 2 * test_grid1.get_output_size());
    test_grid1.output_intensities(0, file);
    test_grid2.output_intensities(test_grid1.get_output_size(), file);
  }

  return 0;
}
