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
 * @file testDensitySubGridDirections.cpp
 *
 * @brief Unit test for DensitySubGrid directional routines.
 *
 * @author Bert Vandenbroucke (bv7@st-andrews.ac.uk)
 */

/*! @brief Output log level. The higher the value, the more stuff is printed to
 *  the stderr. Comment to disable logging altogether. */
#define LOG_OUTPUT 1

/*! @brief Uncomment this to enable run time assertions. */
#define DO_ASSERTS

int MPI_rank, MPI_size;

#include "Assert.hpp"
#include "DensitySubGrid.hpp"
#include "Log.hpp"

/**
 * @brief Check the get_output_direction method of the given grid with the given
 * input parameters and expected output.
 *
 * @param grid DensitySubGrid to test on.
 * @param ix x index.
 * @param iy y index.
 * @param iz z index.
 * @param ref Expected output.
 */
#define check_direction(grid, ix, iy, iz, ref)                                 \
  {                                                                            \
    const int three_index[3] = {ix, iy, iz};                                   \
    const int result = grid.get_output_direction(three_index);                 \
    if (result != ref) {                                                       \
      std::cerr << "Wrong output direction!" << std::endl;                     \
      std::cerr << "Three index: " << ix << " " << iy << " " << iz             \
                << std::endl;                                                  \
      std::cerr << "Expected: " << ref << ", got: " << result << std::endl;    \
      return 1;                                                                \
    }                                                                          \
  }

/**
 * @brief Check the get_start_index function of the given grid againts the given
 * reference output.
 *
 * @param grid DensitySubGrid to test.
 * @param input_direction Input TravelDirection.
 * @param cx Input x coordinate (in m).
 * @param cy Input y coordinate (in m).
 * @param cz Input z coordinate (in m).
 * @param rx Expected x index output.
 * @param ry Expected y index output.
 * @param rz Expected z index output.
 */
#define check_input(grid, input_direction, cx, cy, cz, rx, ry, rz)             \
  {                                                                            \
    int three_index[3];                                                        \
    const double position[3] = {cx, cy, cz};                                   \
    grid.get_start_index(position, input_direction, three_index);              \
    if (!(three_index[0] == rx && three_index[1] == ry &&                      \
          three_index[2] == rz)) {                                             \
      std::cerr << "Wrong input three index!" << std::endl;                    \
      std::cerr << "Input direction: " << input_direction << std::endl;        \
      std::cerr << "Expected: " << rx << " " << ry << " " << rz << std::endl;  \
      std::cerr << "Got: " << three_index[0] << " " << three_index[1] << " "   \
                << three_index[2] << std::endl;                                \
      return 1;                                                                \
    }                                                                          \
  }

/**
 * @brief Unit test for DensitySubGrid directional routines.
 *
 * @param argc Number of command line arguments.
 * @param argv Command line arguments.
 * @return Exit code: 0 on success.
 */
int main(int argc, char **argv) {

  const double testbox[6] = {0., 0., 0., 1., 1., 1.};
  const int testncell[3] = {10, 10, 10};
  DensitySubGrid test_grid(testbox, testncell);

  check_direction(test_grid, 4, 5, 3, TRAVELDIRECTION_INSIDE);

  check_direction(test_grid, -1, 5, 3, TRAVELDIRECTION_FACE_X_N);
  check_direction(test_grid, 10, 5, 3, TRAVELDIRECTION_FACE_X_P);
  check_direction(test_grid, 4, -1, 3, TRAVELDIRECTION_FACE_Y_N);
  check_direction(test_grid, 4, 10, 3, TRAVELDIRECTION_FACE_Y_P);
  check_direction(test_grid, 4, 5, -1, TRAVELDIRECTION_FACE_Z_N);
  check_direction(test_grid, 4, 5, 10, TRAVELDIRECTION_FACE_Z_P);

  check_direction(test_grid, -1, -1, 3, TRAVELDIRECTION_EDGE_Z_NN);
  check_direction(test_grid, -1, 10, 3, TRAVELDIRECTION_EDGE_Z_NP);
  check_direction(test_grid, 10, -1, 3, TRAVELDIRECTION_EDGE_Z_PN);
  check_direction(test_grid, 10, 10, 3, TRAVELDIRECTION_EDGE_Z_PP);
  check_direction(test_grid, -1, 5, -1, TRAVELDIRECTION_EDGE_Y_NN);
  check_direction(test_grid, -1, 5, 10, TRAVELDIRECTION_EDGE_Y_NP);
  check_direction(test_grid, 10, 5, -1, TRAVELDIRECTION_EDGE_Y_PN);
  check_direction(test_grid, 10, 5, 10, TRAVELDIRECTION_EDGE_Y_PP);
  check_direction(test_grid, 4, -1, -1, TRAVELDIRECTION_EDGE_X_NN);
  check_direction(test_grid, 4, -1, 10, TRAVELDIRECTION_EDGE_X_NP);
  check_direction(test_grid, 4, 10, -1, TRAVELDIRECTION_EDGE_X_PN);
  check_direction(test_grid, 4, 10, 10, TRAVELDIRECTION_EDGE_X_PP);

  check_direction(test_grid, -1, -1, -1, TRAVELDIRECTION_CORNER_NNN);
  check_direction(test_grid, -1, -1, 10, TRAVELDIRECTION_CORNER_NNP);
  check_direction(test_grid, -1, 10, -1, TRAVELDIRECTION_CORNER_NPN);
  check_direction(test_grid, -1, 10, 10, TRAVELDIRECTION_CORNER_NPP);
  check_direction(test_grid, 10, -1, -1, TRAVELDIRECTION_CORNER_PNN);
  check_direction(test_grid, 10, -1, 10, TRAVELDIRECTION_CORNER_PNP);
  check_direction(test_grid, 10, 10, -1, TRAVELDIRECTION_CORNER_PPN);
  check_direction(test_grid, 10, 10, 10, TRAVELDIRECTION_CORNER_PPP);

  logmessage("get_output_direction() test successful.", 0);

  check_input(test_grid, TRAVELDIRECTION_INSIDE, 0.5, 0.5, 0.5, 5, 5, 5);

  check_input(test_grid, TRAVELDIRECTION_FACE_X_N, 0., 0.5, 0.5, 0, 5, 5);
  check_input(test_grid, TRAVELDIRECTION_FACE_X_P, 1., 0.5, 0.5, 9, 5, 5);
  check_input(test_grid, TRAVELDIRECTION_FACE_Y_N, 0.5, 0., 0.5, 5, 0, 5);
  check_input(test_grid, TRAVELDIRECTION_FACE_Y_P, 0.5, 1., 0.5, 5, 9, 5);
  check_input(test_grid, TRAVELDIRECTION_FACE_Z_N, 0.5, 0.5, 0., 5, 5, 0);
  check_input(test_grid, TRAVELDIRECTION_FACE_Z_P, 0.5, 0.5, 1., 5, 5, 9);

  check_input(test_grid, TRAVELDIRECTION_EDGE_X_NN, 0.5, 0., 0., 5, 0, 0);
  check_input(test_grid, TRAVELDIRECTION_EDGE_X_NP, 0.5, 0., 1., 5, 0, 9);
  check_input(test_grid, TRAVELDIRECTION_EDGE_X_PN, 0.5, 1., 0., 5, 9, 0);
  check_input(test_grid, TRAVELDIRECTION_EDGE_X_PP, 0.5, 1., 1., 5, 9, 9);
  check_input(test_grid, TRAVELDIRECTION_EDGE_Y_NN, 0., 0.5, 0., 0, 5, 0);
  check_input(test_grid, TRAVELDIRECTION_EDGE_Y_NP, 0., 0.5, 1., 0, 5, 9);
  check_input(test_grid, TRAVELDIRECTION_EDGE_Y_PN, 1., 0.5, 0., 9, 5, 0);
  check_input(test_grid, TRAVELDIRECTION_EDGE_Y_PP, 1., 0.5, 1., 9, 5, 9);
  check_input(test_grid, TRAVELDIRECTION_EDGE_Z_NN, 0., 0., 0.5, 0, 0, 5);
  check_input(test_grid, TRAVELDIRECTION_EDGE_Z_NP, 0., 1., 0.5, 0, 9, 5);
  check_input(test_grid, TRAVELDIRECTION_EDGE_Z_PN, 1., 0., 0.5, 9, 0, 5);
  check_input(test_grid, TRAVELDIRECTION_EDGE_Z_PP, 1., 1., 0.5, 9, 9, 5);

  check_input(test_grid, TRAVELDIRECTION_CORNER_NNN, 0., 0., 0., 0, 0, 0);
  check_input(test_grid, TRAVELDIRECTION_CORNER_NNP, 0., 0., 1., 0, 0, 9);
  check_input(test_grid, TRAVELDIRECTION_CORNER_NPN, 0., 1., 0., 0, 9, 0);
  check_input(test_grid, TRAVELDIRECTION_CORNER_NPP, 0., 1., 1., 0, 9, 9);
  check_input(test_grid, TRAVELDIRECTION_CORNER_PNN, 1., 0., 0., 9, 0, 0);
  check_input(test_grid, TRAVELDIRECTION_CORNER_PNP, 1., 0., 1., 9, 0, 9);
  check_input(test_grid, TRAVELDIRECTION_CORNER_PPN, 1., 1., 0., 9, 9, 0);
  check_input(test_grid, TRAVELDIRECTION_CORNER_PPP, 1., 1., 1., 9, 9, 9);

  logmessage("get_start_index() test successful.", 0);

  logmessage("All tests successful.", 0);

  return 0;
}
