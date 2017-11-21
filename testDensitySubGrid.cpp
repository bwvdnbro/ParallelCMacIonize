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
  const int num_subgrid[3] = {3, 3, 3};

  // set up the grid of smaller grids used for the algorithm
  // each smaller grid stores a fraction of the total grid and has its own
  // buffers to store photons
  std::vector< DensitySubGrid * > gridvec(num_subgrid[0] * num_subgrid[1] *
                                          num_subgrid[2]);
  const double subbox_side[3] = {box[3] / num_subgrid[0],
                                 box[4] / num_subgrid[1],
                                 box[5] / num_subgrid[2]};
  const int subbox_ncell[3] = {ncell[0] / num_subgrid[0],
                               ncell[1] / num_subgrid[1],
                               ncell[2] / num_subgrid[2]};
  for (unsigned int ix = 0; ix < num_subgrid[0]; ++ix) {
    for (unsigned int iy = 0; iy < num_subgrid[1]; ++iy) {
      for (unsigned int iz = 0; iz < num_subgrid[2]; ++iz) {
        const unsigned int index =
            ix * num_subgrid[1] * num_subgrid[2] + iy * num_subgrid[2] + iz;
        const double subbox[6] = {box[0] + ix * subbox_side[0],
                                  box[1] + iy * subbox_side[1],
                                  box[2] + iz * subbox_side[2],
                                  subbox_side[0],
                                  subbox_side[1],
                                  subbox_side[2]};
        gridvec[index] = new DensitySubGrid(subbox, subbox_ncell);
        DensitySubGrid &grid = *gridvec[index];
        // set up the buffers
        // make sure all buffers are empty (_actual_size == 0)
        // by default, all buffers are outside the box (we will set the buffers
        // that are not to the correct value below)
        for (int i = 0; i < 27; ++i) {
          PhotonBuffer &buffer = grid.get_input_buffer(i);
          buffer._direction = i;
          buffer._is_inside_box = false;
          buffer._actual_size = 0;
        }
        for (int i = 0; i < 27; ++i) {
          PhotonBuffer &buffer = grid.get_output_buffer(i);
          buffer._direction = i;
          buffer._is_inside_box = false;
          buffer._actual_size = 0;
        }
        // now set up the correct neighbour relations and flag the buffers that
        // are really inside the box
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
                const int ngbi = grid.get_output_direction(three_index);
                // now get the actual ngb index
                const unsigned int ngb_index =
                    cix * num_subgrid[1] * num_subgrid[2] +
                    ciy * num_subgrid[2] + ciz;
                grid.set_neighbour(ngbi, ngb_index);
                // set the corresponding buffers to inside
                PhotonBuffer &input_buffer = grid.get_input_buffer(ngbi);
                input_buffer._is_inside_box = true;
                PhotonBuffer &output_buffer = grid.get_output_buffer(ngbi);
                output_buffer._is_inside_box = true;
              }
            }
          }
        }
      }
    }
  }
  // get a reference to the central buffer, as this is where our source is
  // located
  DensitySubGrid &grid =
      *gridvec[(num_subgrid[0] >> 1) * num_subgrid[1] * num_subgrid[2] +
               (num_subgrid[1] >> 1) * num_subgrid[2] + (num_subgrid[2] >> 1)];
  PhotonBuffer &input_buffer = grid.get_input_buffer(TRAVELDIRECTION_INSIDE);

  // set up the random number generator
  RandomGenerator random_generator;

  // now for the main loop. This loop
  //  - shoots num_photon photons through the grid to get intensity estimates
  //  - computes the ionization equilibrium
  for (unsigned int iloop = 0; iloop < number_of_iterations; ++iloop) {
    // STEP 0: log output
    logmessage("Loop 1", 0);

    // STEP 1: photon shooting
    unsigned int num_photon_done = 0;
    unsigned int num_active_photons = 1;
    // this loop is repeated until all photons have been shot. It
    //  - adds PHOTONBUFFER_SIZE new source photons to the central buffer
    //  - shoots all photons in all buffers for all subgrids and effectively
    //    moves them from the input to the output buffers
    //  - moves photons from selected output buffers to input buffers of other
    //    subgrids
    while (num_active_photons > 0) {
      // STEP 0: log output
      logmessage("Subloop (" << num_active_photons << ")", 1);

      // STEP 1: add new photons to the source input buffer
      num_active_photons = 0;
      // we only do this step as long as the source has photons...
      if (num_photon_done < num_photon) {
        // shoot PHOTONBUFFER_SIZE photons/all photons that are left (whatever
        // is smallest)
        const unsigned int num_photon_this_loop =
            std::min(PHOTONBUFFER_SIZE, num_photon - num_photon_done);
        num_photon_done += num_photon_this_loop;
        num_active_photons += (num_photon - num_photon_done);
        input_buffer._actual_size = num_photon_this_loop;
        // draw random photons and store them in the buffer
        for (unsigned int i = 0; i < num_photon_this_loop; ++i) {
          Photon &photon = input_buffer._photons[i];
          photon._position[0] = 0.;
          photon._position[1] = 0.;
          photon._position[2] = 0.;
          const double cost =
              2. * random_generator.get_uniform_random_double() - 1.;
          const double sint = std::sqrt(std::max(1. - cost * cost, 0.));
          const double phi =
              2. * M_PI * random_generator.get_uniform_random_double();
          const double cosp = std::cos(phi);
          const double sinp = std::sin(phi);
          photon._direction[0] = sint * cosp;
          photon._direction[1] = sint * sinp;
          photon._direction[2] = cost;
          // we currently assume equal weight for all photons
          photon._weight = 1.;
          photon._current_optical_depth = 0.;
          photon._target_optical_depth =
              -std::log(random_generator.get_uniform_random_double());
          // this is the fixed cross section we use for the moment
          photon._photoionization_cross_section = 6.3e-22;
        }
      }

      // STEP 2: shoot photons, by looping over the subgrids and moving all
      //  photons in a grid's input buffer to one of its output buffers
      for (unsigned int igrid = 0; igrid < gridvec.size(); ++igrid) {
        DensitySubGrid &this_grid = *gridvec[igrid];
        // loop over all 27 input buffers
        for (int ibuffer = 0; ibuffer < 27; ++ibuffer) {
          PhotonBuffer &buffer = this_grid.get_input_buffer(ibuffer);
          // only process input buffers that are inside the box
          if (buffer._is_inside_box) {
            for (unsigned int i = 0; i < buffer._actual_size; ++i) {
              // shoot the photon through the subgrid
              Photon &photon = buffer._photons[i];
              const double inverse_direction[3] = {1. / photon._direction[0],
                                                   1. / photon._direction[1],
                                                   1. / photon._direction[2]};
              const int result = this_grid.interact(photon, inverse_direction,
                                                    buffer._direction);
              PhotonBuffer &output_buffer = grid.get_output_buffer(result);
              // add the photon to the correct output buffer
              const unsigned int index = output_buffer._actual_size;
              output_buffer._photons[index] = photon;
              ++output_buffer._actual_size;
            }
          }
        }
      }

      // STEP 3: move photons from selected output buffers to input buffers of
      //  other subgrids
      for (unsigned int igrid = 0; igrid < gridvec.size(); ++igrid) {
        DensitySubGrid &this_grid = *gridvec[igrid];
        // loop over all output buffers of the subgrid
        for (int ibuffer = 0; ibuffer < 27; ++ibuffer) {
          PhotonBuffer &output_buffer = this_grid.get_output_buffer(ibuffer);
          // if the output buffer is not the internal buffer of the subgrid, and
          // corresponds to an existing subgrid neighbour, then add all photons
          // to the matching input buffer of that neighbour
          if (ibuffer > 0 && output_buffer._is_inside_box) {
            // get the corresponding neighbour
            const unsigned int ngb = this_grid.get_neighbour(ibuffer);
            // get the input buffer of the neighbour that matches this output
            // buffer (what goes in through one corner, enters the neighbour
            // through the opposite corner)
            const int i_input = output_to_input_direction(ibuffer);
            PhotonBuffer &input_buffer =
                gridvec[ngb]->get_input_buffer(i_input);
            // copy the photons
            input_buffer._actual_size = output_buffer._actual_size;
            num_active_photons += output_buffer._actual_size;
            for (unsigned int i = 0; i < output_buffer._actual_size; ++i) {
              input_buffer._photons[i] = output_buffer._photons[i];
            }
          }
          output_buffer._actual_size = 0;
        }
      }
    }

    // STEP 2: update the ionization structure for each subgrid
    for (unsigned int igrid = 0; igrid < gridvec.size(); ++igrid) {
      gridvec[igrid]->compute_neutral_fraction(num_photon);
    }
  }

  // OUTPUT:
  //  - ASCII output (for the VisIt plot script)
  std::ofstream ofile("intensities.txt");
  grid.print_intensities(ofile);
  ofile.close();
  //  - binary output (for the Python plot script)
  std::ofstream bfile("intensities.dat");
  grid.output_intensities(bfile);
  bfile.close();

  // garbage collection
  for (unsigned int igrid = 0; igrid < gridvec.size(); ++igrid) {
    delete gridvec[igrid];
  }

  return 0;
}
