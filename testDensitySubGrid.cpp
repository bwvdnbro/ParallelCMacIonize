/*******************************************************************************
 * This file is part of CMacIonize
 * Copyright (C) 2017 Bert Vandenbroucke (bert.vandenbroucke@gmail.com)
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
 * @file testDensitySubGrid.cpp
 *
 * @brief Unit test for the DensitySubGrid class.
 *
 * @author Bert Vandenbroucke (bv7@st-andrews.ac.uk)
 */

// Project includes
#include "DensitySubGrid.hpp"
#include "PhotonBuffer.hpp"
#include "RandomGenerator.hpp"

// standard library includes
#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>

/*! @brief Output log level. The higher the value, the more stuff is printed to
 *  the stderr. Comment to disable logging altogether. */
#define LOG_OUTPUT 1

/*! @brief Activate this to unit test the directional algorithms. */
//#define TEST_DIRECTIONS

/**
 * @brief Write a message to the log with the given log level.
 *
 * @param message Message to write.
 * @param loglevel Log level. The message is only written if the LOG_OUTPUT
 * defined is higher than this value.
 */
#ifdef LOG_OUTPUT
#define logmessage(message, loglevel)                                          \
  if (loglevel < LOG_OUTPUT) {                                                 \
    std::cerr << message << std::endl;                                         \
  }
#else
#define logmessage(s, loglevel)
#endif

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
 * @param position Input position (in m).
 * @param input_direction Input TravelDirection.
 * @param rx Expected x index output.
 * @param ry Expected y index output.
 * @param rz Expected z index output.
 */
#define check_input(grid, position, input_direction, rx, ry, rz)               \
  {                                                                            \
    int three_index[3];                                                        \
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
 * @brief Draw a random direction.
 *
 * @param random_generator Random number generator to use.
 * @param direction Random direction (output).
 * @param inverse_direction Inverse random direction (output).
 */
inline static void get_random_direction(RandomGenerator &random_generator,
                                        double *direction,
                                        double *inverse_direction) {
  const double cost = 2. * random_generator.get_uniform_random_double() - 1.;
  const double sint = std::sqrt(std::max(1. - cost * cost, 0.));
  const double phi = 2. * M_PI * random_generator.get_uniform_random_double();
  const double cosp = std::cos(phi);
  const double sinp = std::sin(phi);
  direction[0] = sint * cosp;
  direction[1] = sint * sinp;
  direction[2] = cost;
  inverse_direction[0] = 1. / direction[0];
  inverse_direction[1] = 1. / direction[1];
  inverse_direction[2] = 1. / direction[2];
}

/**
 * @brief Unit test for the DensitySubGrid class.
 *
 * Runs a simple Stromgren sphere test with a homogeneous density field, a
 * single stellar source, and a hydrogen only gas with a constant
 * photoionization cross section and recombination rate.
 *
 * @param argc Number of command line arguments.
 * @param argv Command line arguments.
 * @return Exit code: 0 on success.
 */
int main(int argc, char **argv) {

#ifdef TEST_DIRECTIONS
  /// test directional routines
  {
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

    const double position[3] = {0.5, 0.5, 0.5};
    check_input(test_grid, position, TRAVELDIRECTION_INSIDE, 5, 5, 5);

    check_input(test_grid, position, TRAVELDIRECTION_FACE_X_N, 0, 5, 5);
    check_input(test_grid, position, TRAVELDIRECTION_FACE_X_P, 9, 5, 5);
    check_input(test_grid, position, TRAVELDIRECTION_FACE_Y_N, 5, 0, 5);
    check_input(test_grid, position, TRAVELDIRECTION_FACE_Y_P, 5, 9, 5);
    check_input(test_grid, position, TRAVELDIRECTION_FACE_Z_N, 5, 5, 0);
    check_input(test_grid, position, TRAVELDIRECTION_FACE_Z_P, 5, 5, 9);

    check_input(test_grid, position, TRAVELDIRECTION_EDGE_X_NN, 5, 0, 0);
    check_input(test_grid, position, TRAVELDIRECTION_EDGE_X_NP, 5, 0, 9);
    check_input(test_grid, position, TRAVELDIRECTION_EDGE_X_PN, 5, 9, 0);
    check_input(test_grid, position, TRAVELDIRECTION_EDGE_X_PP, 5, 9, 9);
    check_input(test_grid, position, TRAVELDIRECTION_EDGE_Y_NN, 0, 5, 0);
    check_input(test_grid, position, TRAVELDIRECTION_EDGE_Y_NP, 0, 5, 9);
    check_input(test_grid, position, TRAVELDIRECTION_EDGE_Y_PN, 9, 5, 0);
    check_input(test_grid, position, TRAVELDIRECTION_EDGE_Y_PP, 9, 5, 9);
    check_input(test_grid, position, TRAVELDIRECTION_EDGE_Z_NN, 0, 0, 5);
    check_input(test_grid, position, TRAVELDIRECTION_EDGE_Z_NP, 0, 9, 5);
    check_input(test_grid, position, TRAVELDIRECTION_EDGE_Z_PN, 9, 0, 5);
    check_input(test_grid, position, TRAVELDIRECTION_EDGE_Z_PP, 9, 9, 5);

    check_input(test_grid, position, TRAVELDIRECTION_CORNER_NNN, 0, 0, 0);
    check_input(test_grid, position, TRAVELDIRECTION_CORNER_NNP, 0, 0, 9);
    check_input(test_grid, position, TRAVELDIRECTION_CORNER_NPN, 0, 9, 0);
    check_input(test_grid, position, TRAVELDIRECTION_CORNER_NPP, 0, 9, 9);
    check_input(test_grid, position, TRAVELDIRECTION_CORNER_PNN, 9, 0, 0);
    check_input(test_grid, position, TRAVELDIRECTION_CORNER_PNP, 9, 0, 9);
    check_input(test_grid, position, TRAVELDIRECTION_CORNER_PPN, 9, 9, 0);
    check_input(test_grid, position, TRAVELDIRECTION_CORNER_PPP, 9, 9, 9);
  }
#endif

  /// Main simulation parameters
  // size of the box (corresponds to a -5 pc -> 5 pc cube)
  const double box[6] = {-1.543e17, -1.543e17, -1.543e17,
                         3.086e17,  3.086e17,  3.086e17};
  // number of cells: 60^3
  const int ncell[3] = {60, 60, 60};
  // number of photons to shoot per iteration: 10^6
  const unsigned int num_photon = 1000000;
  // number of iterations: 10
  const unsigned int number_of_iterations = 10;
  // number of subgrids: 3^3
  const int num_subgrid[3] = {3, 5, 4};
  // number of photon buffers to use
  // this number of buffers is pre-allocated and the photon traversal routine
  // arbitrarily uses them to store photons
  const unsigned int number_of_buffers = 2000;
  // reemission probability
  const double reemission_probability = 0.364;
  //  const double reemission_probability = 0.;

  // set up the grid of smaller grids used for the algorithm
  // each smaller grid stores a fraction of the total grid and has information
  // about the neighbouring subgrids
  std::vector< DensitySubGrid * > gridvec(num_subgrid[0] * num_subgrid[1] *
                                          num_subgrid[2]);
  const double subbox_side[3] = {box[3] / num_subgrid[0],
                                 box[4] / num_subgrid[1],
                                 box[5] / num_subgrid[2]};
  const int subbox_ncell[3] = {ncell[0] / num_subgrid[0],
                               ncell[1] / num_subgrid[1],
                               ncell[2] / num_subgrid[2]};
  // set up the subgrids
  for (int ix = 0; ix < num_subgrid[0]; ++ix) {
    for (int iy = 0; iy < num_subgrid[1]; ++iy) {
      for (int iz = 0; iz < num_subgrid[2]; ++iz) {
        const unsigned int index =
            ix * num_subgrid[1] * num_subgrid[2] + iy * num_subgrid[2] + iz;
        const double subbox[6] = {box[0] + ix * subbox_side[0],
                                  box[1] + iy * subbox_side[1],
                                  box[2] + iz * subbox_side[2],
                                  subbox_side[0],
                                  subbox_side[1],
                                  subbox_side[2]};
        gridvec[index] = new DensitySubGrid(subbox, subbox_ncell);
        DensitySubGrid &this_grid = *gridvec[index];
        // set up neighbouring information. We first make sure all neighbours
        // are initialized to NEIGHBOUR_OUTSIDE, indicating no neighbour
        for (int i = 0; i < 27; ++i) {
          this_grid.set_neighbour(i, NEIGHBOUR_OUTSIDE);
        }
        // now set up the correct neighbour relations for the neighbours that
        // exist
        for (int nix = -1; nix < 2; ++nix) {
          for (int niy = -1; niy < 2; ++niy) {
            for (int niz = -1; niz < 2; ++niz) {
              // get neighbour corrected indices
              const int cix = ix + nix;
              const int ciy = iy + niy;
              const int ciz = iz + niz;
              // if the indices above point to a real subgrid: set up the
              // neighbour relations
              if (cix >= 0 && cix < num_subgrid[0] && ciy >= 0 &&
                  ciy < num_subgrid[1] && ciz >= 0 && ciz < num_subgrid[2]) {
                // we use get_output_direction() to get the correct index for
                // the neighbour
                // the three_index components will either be
                //  - -ncell --> negative --> lower limit
                //  - 0 --> in range --> inside
                //  - ncell --> upper limit
                const int three_index[3] = {nix * subbox_ncell[0],
                                            niy * subbox_ncell[1],
                                            niz * subbox_ncell[2]};
                const int ngbi = this_grid.get_output_direction(three_index);
                // now get the actual ngb index
                const unsigned int ngb_index =
                    cix * num_subgrid[1] * num_subgrid[2] +
                    ciy * num_subgrid[2] + ciz;
                this_grid.set_neighbour(ngbi, ngb_index);
              }
            }
          }
        }
      }
    }
  }

  // set up the photon buffers
  std::vector< PhotonBuffer > photon_buffers(number_of_buffers);
  // next_free_buffer always stores the index of the next buffer that is still
  // available
  // free_photon_buffers stores, for every buffer, the next free index assuming
  // that buffer is also free. This way, the free buffers form a linked-list,
  // which enables obtaining and releasing free buffers:
  //  - to claim a free buffer, we just use next_free_buffer and set
  //    next_free_buffer to free_photon_buffers[next_free_buffer]
  //  - to release a buffer with index ifree, we set
  //    free_photon_buffers[ifree] = next_free_buffer, and set
  //    next_free_buffer to ifree
  // We also need to make sure the buffer._is_input flag is up to date:
  //  - 'true' means the buffer is claimed and should be processed
  //  - 'false' means the buffer is free
  unsigned int next_free_buffer = 0;
  std::vector< unsigned int > free_photon_buffers(number_of_buffers, 0);
  // initialize the free_photon_buffers list and mark all buffers as free
  for (unsigned int i = 0; i < number_of_buffers; ++i) {
    free_photon_buffers[i] = i + 1;
    // disable processing of buffers
    photon_buffers[i]._is_input = false;
  }
  unsigned int number_of_buffers_used = 0;

  // get a reference to the central buffer, as this is where our source is
  // located
  const unsigned int central_index =
      (num_subgrid[0] >> 1) * num_subgrid[1] * num_subgrid[2] +
      (num_subgrid[1] >> 1) * num_subgrid[2] + (num_subgrid[2] >> 1);

  // set up the random number generator
  RandomGenerator random_generator;

  // now for the main loop. This loop
  //  - shoots num_photon photons through the grid to get intensity estimates
  //  - computes the ionization equilibrium
  for (unsigned int iloop = 0; iloop < number_of_iterations; ++iloop) {
    // STEP 0: log output
    logmessage("Loop " << iloop + 1, 0);

    // STEP 1: photon shooting
    unsigned int num_photon_done = 0;
    unsigned int num_active_photons = 1;
    // this loop is repeated until all photons have been shot. It
    //  - creates one new buffer with source photons
    //  - shoots all photons in all buffers that are claimed and creates new
    //    buffers with output photons
    //  - updates the number of photons that are still active
    while (num_active_photons > 0) {
      // STEP 0: log output
      logmessage("Subloop (" << num_active_photons << ")", 1);

      // STEP 1: add new photons from the source
      num_active_photons = 0;
      // we only do this step as long as the source has photons...
      if (num_photon_done < num_photon) {
        // shoot PHOTONBUFFER_SIZE photons/all photons that are left (whatever
        // is smallest)
        const unsigned int num_photon_this_loop =
            std::min(PHOTONBUFFER_SIZE, num_photon - num_photon_done);
        num_photon_done += num_photon_this_loop;
        // make sure the main loop knows there are more photons coming
        num_active_photons += (num_photon - num_photon_done);
        // claim a new buffer and set its properties
        PhotonBuffer &input_buffer = photon_buffers[next_free_buffer];
        next_free_buffer = free_photon_buffers[next_free_buffer];
        number_of_buffers_used =
            std::max(number_of_buffers_used, next_free_buffer);
        myassert(next_free_buffer != number_of_buffers, "overflow!");
        input_buffer._actual_size = num_photon_this_loop;
        input_buffer._sub_grid_index = central_index;
        input_buffer._direction = TRAVELDIRECTION_INSIDE;
        input_buffer._is_input = true;
        // draw random photons and store them in the buffer
        for (unsigned int i = 0; i < num_photon_this_loop; ++i) {
          Photon &photon = input_buffer._photons[i];
          photon._position[0] = 0.;
          photon._position[1] = 0.;
          photon._position[2] = 0.;
          get_random_direction(random_generator, photon._direction,
                               photon._inverse_direction);
          // we currently assume equal weight for all photons
          photon._weight = 1.;
          photon._current_optical_depth = 0.;
          photon._target_optical_depth =
              -std::log(random_generator.get_uniform_random_double());
          // this is the fixed cross section we use for the moment
          photon._photoionization_cross_section = 6.3e-22;
        }
      }

      // STEP 2: shoot photons, by looping over the buffers and processing all
      //  active buffer. Each buffer creates new output buffers that might or
      //  might not be processed, depending on where they are in the list
      for (unsigned int ibuffer = 0; ibuffer < number_of_buffers; ++ibuffer) {
        PhotonBuffer &buffer = photon_buffers[ibuffer];
        // we only process active buffers
        if (buffer._is_input) {
          // get the subgrid connected to this buffer
          DensitySubGrid &this_grid = *gridvec[buffer._sub_grid_index];
          // try to set the lock for this region
          if (this_grid.lock()) {
            // lock succeeded: do photon traversal
            // create provisional output buffers
            unsigned int output_indices[27];
            for (int i = 0; i < 27; ++i) {
              const unsigned int ngb = this_grid.get_neighbour(i);
              // only create buffers that correspond to existing subgrids
              if (ngb != NEIGHBOUR_OUTSIDE) {
                output_indices[i] = next_free_buffer;
                photon_buffers[next_free_buffer]._is_input = true;
                photon_buffers[next_free_buffer]._sub_grid_index = ngb;
                photon_buffers[next_free_buffer]._direction =
                    output_to_input_direction(i);
                photon_buffers[next_free_buffer]._actual_size = 0;
                next_free_buffer = free_photon_buffers[next_free_buffer];
                number_of_buffers_used =
                    std::max(number_of_buffers_used, next_free_buffer);
                myassert(next_free_buffer != number_of_buffers, "overflow!");
              } else {
                output_indices[i] = NEIGHBOUR_OUTSIDE;
              }
            }
            // now do the actual photon traversal
            for (unsigned int i = 0; i < buffer._actual_size; ++i) {
              Photon &photon = buffer._photons[i];
              const int result = this_grid.interact(photon, buffer._direction);
              myassert(result >= 0 && result < 27, "fail");
              // check if an absorbed photon needs to be reemitted
              bool reemit = false;
              if (result == 0 &&
                  random_generator.get_uniform_random_double() <
                      reemission_probability) {
                reemit = true;
                get_random_direction(random_generator, photon._direction,
                                     photon._inverse_direction);
                photon._current_optical_depth = 0.;
                photon._target_optical_depth =
                    -std::log(random_generator.get_uniform_random_double());
              }
              // add the photon to an output buffer, if it still exists
              if ((result > 0 && output_indices[result] != NEIGHBOUR_OUTSIDE) ||
                  reemit) {
                PhotonBuffer &output_buffer =
                    photon_buffers[output_indices[result]];
                // add the photon to the correct output buffer
                const unsigned int index = output_buffer._actual_size;
                output_buffer._photons[index] = photon;
                myassert(output_buffer._photons[index]._position[0] ==
                                 photon._position[0] &&
                             output_buffer._photons[index]._position[1] ==
                                 photon._position[1] &&
                             output_buffer._photons[index]._position[2] ==
                                 photon._position[2],
                         "fail");

                ++output_buffer._actual_size;
                myassert(output_buffer._actual_size < PHOTONBUFFER_SIZE,
                         "output buffer size: " << output_buffer._actual_size);
              }
            }
            // remove the buffer
            buffer._is_input = false;
            free_photon_buffers[ibuffer] = next_free_buffer;
            next_free_buffer = ibuffer;
            // remove empty output buffers
            for (unsigned int i = 0; i < 27; ++i) {
              if (output_indices[i] != NEIGHBOUR_OUTSIDE &&
                  photon_buffers[output_indices[i]]._actual_size == 0) {
                photon_buffers[output_indices[i]]._is_input = false;
                free_photon_buffers[output_indices[i]] = next_free_buffer;
                next_free_buffer = output_indices[i];
              }
            }
            // we're done: unlock the subgrid
            this_grid.unlock();
          }
        } // if input buffer
      }   // for flexible_buffer elements

      // STEP 3: update the number of active photons counter
      for (unsigned int ibuffer = 0; ibuffer < photon_buffers.size();
           ++ibuffer) {
        num_active_photons += (photon_buffers[ibuffer]._is_input)
                                  ? photon_buffers[ibuffer]._actual_size
                                  : 0;
      }
    }

    // STEP 2: update the ionization structure for each subgrid
    for (unsigned int igrid = 0; igrid < gridvec.size(); ++igrid) {
      gridvec[igrid]->compute_neutral_fraction(num_photon);
    }
  }

  logmessage("Number of buffers used: " << number_of_buffers_used, 0);

  // OUTPUT:
  //  - ASCII output (for the VisIt plot script)
  std::ofstream ofile("intensities.txt");
  for (unsigned int igrid = 0; igrid < gridvec.size(); ++igrid) {
    gridvec[igrid]->print_intensities(ofile);
  }
  ofile.close();
  //  - binary output (for the Python plot script)
  std::ofstream bfile("intensities.dat");
  for (unsigned int igrid = 0; igrid < gridvec.size(); ++igrid) {
    gridvec[igrid]->output_intensities(bfile);
  }
  bfile.close();

  // garbage collection
  for (unsigned int igrid = 0; igrid < gridvec.size(); ++igrid) {
    delete gridvec[igrid];
  }

  return 0;
}
