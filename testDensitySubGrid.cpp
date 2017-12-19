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

// Defines: we do these first, as some includes depend on them

/*! @brief Output log level. The higher the value, the more stuff is printed to
 *  the stderr. Comment to disable logging altogether. */
#define LOG_OUTPUT 1

/*! @brief Enable this to activate task output. */
#define TASK_OUTPUT

/*! @brief Enable this to activate cost output. */
#define COST_OUTPUT

/*! @brief Activate this to unit test the directional algorithms. */
//#define TEST_DIRECTIONS

/*! @brief Activate this to unit test the CostVector. */
//#define TEST_COSTVECTOR

#ifdef TASK_OUTPUT
// activate task output in Task.hpp
#define TASK_PLOT
#endif

// Project includes
#include "CostVector.hpp"
#include "DensitySubGrid.hpp"
#include "MemorySpace.hpp"
#include "NewQueue.hpp"
#include "PhotonBuffer.hpp"
#include "RandomGenerator.hpp"
#include "Task.hpp"
#include "Timer.hpp"

// standard library includes
#include <cmath>
#include <fstream>
#include <iostream>
#include <mpi.h>
#include <omp.h>
#include <sstream>
#include <sys/resource.h>
#include <vector>

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
    _Pragma("omp single") { std::cerr << message << std::endl; }               \
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
 * @brief Write a file with the start and end times of all tasks.
 *
 * @param iloop Iteration number (added to file name).
 * @param tasks Tasks to print.
 */
inline void output_tasks(const unsigned int iloop,
                         const ThreadSafeVector<Task> &tasks) {
#ifdef TASK_OUTPUT
  std::stringstream filename;
  filename << "tasks_";
  filename.fill('0');
  filename.width(2);
  filename << iloop;
  filename << ".txt";

  std::ofstream ofile(filename.str());
  ofile << "# thread\tstart\tstop\ttype\n";

  const size_t tsize = tasks.size();
  for (size_t i = 0; i < tsize; ++i) {
    const Task &task = tasks[i];
    ofile << task._thread_id << "\t" << task._start_time << "\t"
          << task._end_time << "\t" << task._type << "\n";
  }
#endif
}

/**
 * @brief Write files with cost information for an iteration.
 *
 * @param iloop Iteration number (added to file names).
 * @param ngrid Number of subgrids.
 * @param nthread Number of threads.
 * @param costs CostVector to print.
 */
inline void output_costs(const unsigned int iloop, const unsigned int ngrid,
                         const int nthread, const CostVector &costs) {
#ifdef COST_OUTPUT
  std::stringstream filename;
  filename << "costs_";
  filename.fill('0');
  filename.width(2);
  filename << iloop;
  filename << ".txt";

  std::ofstream ofile(filename.str());
  ofile << "# subgrid\tcost\trank\tthread\n";
  for (unsigned int i = 0; i < ngrid; ++i) {
    ofile << i << "\t" << costs.get_cost(i) << "\t"
          << "\t" << costs.get_process(i) << "\t" << costs.get_thread(i)
          << "\n";
  }
#endif
}

/**
 * @brief Get the unit of @f$2^{10e}@f$ bytes.
 *
 * @param exponent Exponent @f$e@f$.
 * @return Name for @f$2^{10e}@f$ bytes: (@f$2^{10}@f$ bytes = KB, ...).
 */
inline std::string byte_unit(uint_fast8_t exponent) {
  switch (exponent) {
  case 0:
    return "bytes";
  case 1:
    return "KB";
  case 2:
    return "MB";
  case 3:
    return "GB";
  case 4:
    return "TB";
  default:
    return "";
  }
}

/**
 * @brief Convert the given number of bytes to a human readable string.
 *
 * @param bytes Number of bytes.
 * @return std::string containing the given number of bytes in "bytes", "KB",
 * "MB", "GB"...
 */
inline std::string human_readable_bytes(size_t bytes) {
  uint_fast8_t sizecount = 0;
  double bytefloat = bytes;
  while ((bytes >> 10) > 0) {
    bytes >>= 10;
    ++sizecount;
    bytefloat /= 1024.;
  }
  std::stringstream bytestream;
  bytefloat = std::round(100. * bytefloat) * 0.01;
  bytestream << bytefloat << " " << byte_unit(sizecount);
  return bytestream.str();
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
 * @brief Fill the given PhotonBuffer with random photons.
 *
 * @param buffer PhotonBuffer to fill.
 * @param number_of_photons Number of photons to draw randomly.
 * @param random_generator RandomGenerator used to generate random numbers.
 * @param source_index Index of the subgrid that contains the source.
 */
inline static void fill_buffer(PhotonBuffer &buffer,
                               const unsigned int number_of_photons,
                               RandomGenerator &random_generator,
                               const unsigned int source_index) {
  buffer._actual_size = number_of_photons;
  buffer._sub_grid_index = source_index;
  buffer._direction = TRAVELDIRECTION_INSIDE;
  // draw random photons and store them in the buffer
  for (unsigned int i = 0; i < number_of_photons; ++i) {
    Photon &photon = buffer._photons[i];
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
    myassert(photon._direction[0] != 0. || photon._direction[1] != 0. ||
                 photon._direction[2] != 0.,
             "fail");
  }
}

/**
 * @brief Do the photon traversal for the given input buffer for the given
 * subgrid and store the result in the given output buffers.
 *
 * @param input_buffer Input PhotonBuffer.
 * @param subgrid DensitySubGrid to operate on.
 * @param output_buffers Output PhotonBuffers.
 */
inline static void do_photon_traversal(PhotonBuffer &input_buffer,
                                       DensitySubGrid &subgrid,
                                       PhotonBuffer *output_buffers,
                                       bool *output_buffer_flags) {
  for (unsigned int i = 0; i < input_buffer._actual_size; ++i) {
    Photon &photon = input_buffer._photons[i];
    myassert(photon._direction[0] != 0. || photon._direction[1] != 0. ||
                 photon._direction[2] != 0.,
             "size: " << input_buffer._actual_size);
    const int result = subgrid.interact(photon, input_buffer._direction);
    myassert(result >= 0 && result < 27, "fail");
    // add the photon to an output buffer, if it still exists
    if (output_buffer_flags[result]) {
      PhotonBuffer &output_buffer = output_buffers[result];
      // add the photon to the correct output buffer
      const unsigned int index = output_buffer._actual_size;
      output_buffer._photons[index] = photon;
      myassert(
          output_buffer._photons[index]._position[0] == photon._position[0] &&
              output_buffer._photons[index]._position[1] ==
                  photon._position[1] &&
              output_buffer._photons[index]._position[2] == photon._position[2],
          "fail");
      myassert(output_buffer._photons[index]._direction[0] != 0. ||
                   output_buffer._photons[index]._direction[1] != 0. ||
                   output_buffer._photons[index]._direction[2] != 0.,
               "size: " << output_buffer._actual_size);

      ++output_buffer._actual_size;
      myassert(output_buffer._actual_size < PHOTONBUFFER_SIZE,
               "output buffer size: " << output_buffer._actual_size);
    }
  }
}

/**
 * @brief Do reemission for the given PhotonBuffer.
 *
 * @param buffer PhotonBuffer to act on.
 * @param random_generator RandomGenerator to use to draw random numbers.
 * @param reemission_probability Reemission probability.
 */
inline static void do_reemission(PhotonBuffer &buffer,
                                 RandomGenerator &random_generator,
                                 const double reemission_probability) {
  unsigned int index = 0;
  for (unsigned int i = 0; i < buffer._actual_size; ++i) {
    if (random_generator.get_uniform_random_double() < reemission_probability) {
      Photon &photon = buffer._photons[i];
      get_random_direction(random_generator, photon._direction,
                           photon._inverse_direction);
      photon._current_optical_depth = 0.;
      photon._target_optical_depth =
          -std::log(random_generator.get_uniform_random_double());
      // we never overwrite a photon that should be preserved (we either
      // overwrite the photon itself, or a photon that is absorbed)
      buffer._photons[index] = photon;
      ++index;
    }
  }
  buffer._actual_size = index;
}

/**
 * @brief Make copies of the subgrids according to the given level matrix.
 *
 * The level matrix is per subgrid number that tells us how many copies we need
 * of that particular subgrid: for a subgrid at level \f$l\f$, \f$2^l\f$ copies
 * are made.
 *
 * The power of 2 hierarchy is necessary to get a consistent neighbour
 * structure, as we want neighbouring copies at the same level to be mutual
 * neighbours. For neighbours on different levels, the neighbour relations are
 * not mutual: if subgrid \f$A\f$ has level \f$l_l\f$ and its neighbour \f$B\f$
 * has level \f$l_h\f$ (\f$l_l < l_h\f$), then groups of \f$2^{l_h-l_l}\f$
 * copies of \f$B\f$ will share the same neighbour copy of \f$A\f$, while that
 * same neighbour copy of \f$A\f$ will only have one copy of \f$B\f$ out of that
 * group as neighbour.
 *
 * New copies are stored at the end of the subgrid list, and the index of the
 * original subgrid is retained in a separate list of originals.
 *
 * @param gridvec List of subgrids.
 * @param levels Desired copy level of each subgrid.
 * @param new_buffers Photon buffer space to add newly created copy buffers to.
 * @param originals List of originals for the newly created copies.
 */
inline void create_copies(std::vector<DensitySubGrid *> &gridvec,
                          std::vector<unsigned char> &levels,
                          MemorySpace &new_buffers,
                          std::vector<unsigned int> &originals) {

  // we need to do 2 loops:
  //  - one loop to create the copies and store the offset of the first copy
  //    for each subgrid
  //  - a second loop that sets the neighbours (and has access to all necessary
  //    copies to set inter-copy neighbour relations)

  // we need to store the original number of subgrids for reference
  const unsigned int number_of_unique_subgrids = gridvec.size();

  // array to store the offsets of new copies in
  std::vector<unsigned int> copy_offsets(gridvec.size(), 0);
  for (unsigned int i = 0; i < number_of_unique_subgrids; ++i) {
    const unsigned char level = levels[i];
    const unsigned int number_of_copies = 1 << level;
    // create the copies
    if (number_of_copies > 1) {
      copy_offsets.push_back(gridvec.size());
    }
    for (unsigned int j = 1; j < number_of_copies; ++j) {
      gridvec.push_back(new DensitySubGrid(*gridvec[i]));
    }
  }

  // neighbour setting
  for (unsigned int i = 0; i < number_of_unique_subgrids; ++i) {
    const unsigned char level = levels[i];
    const unsigned int number_of_copies = 1 << level;
    // first do the self-reference for each copy (if there are copies)
    for (unsigned int j = 1; j < number_of_copies; ++j) {
      const unsigned int copy = copy_offsets[i] + j - 1;
      gridvec[copy]->set_neighbour(i, copy);
      const unsigned int active_buffer = new_buffers.get_free_buffer();
      new_buffers[active_buffer]._sub_grid_index = copy;
      new_buffers[active_buffer]._direction = TRAVELDIRECTION_INSIDE;
      gridvec[copy]->set_active_buffer(i, active_buffer);
    }
    // now do the actual neighbours
    for (int j = 1; j < 27; ++j) {
      const unsigned int original_ngb = gridvec[i]->get_neighbour(j);
      const unsigned char ngb_level = levels[original_ngb];
      // check how the neighbour level compares to the subgrid level
      if (ngb_level == level) {
        // same, easy: just make copies mutual neighbours
        for (unsigned int k = 1; k < number_of_copies; ++k) {
          const unsigned int copy = copy_offsets[i] + k - 1;
          const unsigned int ngb_copy = copy_offsets[original_ngb] + k - 1;
          gridvec[copy]->set_neighbour(j, ngb_copy);
          const unsigned int active_buffer = new_buffers.get_free_buffer();
          new_buffers[active_buffer]._sub_grid_index = ngb_copy;
          new_buffers[active_buffer]._direction = output_to_input_direction(j);
          gridvec[copy]->set_active_buffer(j, active_buffer);
        }
      } else {
        // not the same: there are 2 options
        if (level > ngb_level) {
          // we have less neighbour copies, so some of our copies need to share
          // the same neighbour
          const unsigned int number_of_ngb_copies = 1 << (level - ngb_level);
          for (unsigned int k = 1; k < number_of_copies; ++k) {
            const unsigned int copy = copy_offsets[i] + k - 1;
            // the second term will always round down, which is what we want
            const unsigned int ngb_copy =
                copy_offsets[original_ngb] + (k - 1) / number_of_ngb_copies;
            gridvec[copy]->set_neighbour(j, ngb_copy);
            const unsigned int active_buffer = new_buffers.get_free_buffer();
            new_buffers[active_buffer]._sub_grid_index = ngb_copy;
            new_buffers[active_buffer]._direction =
                output_to_input_direction(j);
            gridvec[copy]->set_active_buffer(j, active_buffer);
          }
        } else {
          // we have more neighbour copies: pick a subset
          const unsigned int number_of_own_copies = 1 << (ngb_level - level);
          for (unsigned int k = 1; k < number_of_copies; ++k) {
            const unsigned int copy = copy_offsets[i] + k - 1;
            // the second term will skip some neighbour copies, which is what we
            // want
            const unsigned int ngb_copy =
                copy_offsets[original_ngb] + (k - 1) * number_of_own_copies;
            gridvec[copy]->set_neighbour(j, ngb_copy);
            const unsigned int active_buffer = new_buffers.get_free_buffer();
            new_buffers[active_buffer]._sub_grid_index = ngb_copy;
            new_buffers[active_buffer]._direction =
                output_to_input_direction(j);
            gridvec[copy]->set_active_buffer(j, active_buffer);
          }
        }
      }
    }
  }
}

/**
 * @brief Create a copy of the subgrid with the given index.
 *
 * @param original Original subgrid index.
 * @param gridvec Subgrids.
 * @param new_buffers Photon buffer space.
 * @param duplicates List of copies.
 */
inline void
make_copy(const unsigned int original, std::vector<DensitySubGrid *> &gridvec,
          MemorySpace &new_buffers,
          std::vector<std::pair<unsigned int, unsigned int>> &duplicates) {
  // copy subgrids 29 and 30, as we know they are very computationally expensive
  const unsigned int copy = gridvec.size();
  duplicates.push_back(std::make_pair(original, copy));

  gridvec.push_back(new DensitySubGrid(*gridvec[original]));
  for (int i = 0; i < 27; ++i) {
    if (i > 0) {
      const unsigned int original_ngb = gridvec[original]->get_neighbour(i);
      gridvec[copy]->set_neighbour(i, original_ngb);
      const unsigned int active_buffer = new_buffers.get_free_buffer();
      new_buffers[active_buffer]._sub_grid_index = original_ngb;
      new_buffers[active_buffer]._direction = output_to_input_direction(i);
      gridvec[copy]->set_active_buffer(i, active_buffer);
    } else {
      gridvec[copy]->set_neighbour(i, copy);
      const unsigned int active_buffer = new_buffers.get_free_buffer();
      new_buffers[active_buffer]._sub_grid_index = copy;
      new_buffers[active_buffer]._direction = TRAVELDIRECTION_INSIDE;
      gridvec[copy]->set_active_buffer(i, active_buffer);
    }
  }
}

/**
 * @brief Make sure the given copies of the given original subgrids are mutual
 * neighbours.
 *
 * @param original_A First original.
 * @param original_B Second original.
 * @param copy_A First copy.
 * @param copy_B Second copy.
 * @param gridvec Subgrids.
 * @param new_buffers Photon buffer space.
 */
inline void ensure_neighbours(const unsigned int original_A,
                              const unsigned int original_B,
                              const unsigned int copy_A,
                              const unsigned int copy_B,
                              std::vector<DensitySubGrid *> &gridvec,
                              MemorySpace &new_buffers) {
  for (int i = 1; i < 27; ++i) {
    if (gridvec[original_A]->get_neighbour(i) == original_B) {
      gridvec[copy_A]->set_neighbour(i, copy_B);
      unsigned int active_buffer = gridvec[copy_A]->get_active_buffer(i);
      new_buffers[active_buffer]._sub_grid_index = copy_B;
      gridvec[copy_B]->set_neighbour(output_to_input_direction(i), copy_A);
      active_buffer =
          gridvec[copy_B]->get_active_buffer(output_to_input_direction(i));
      new_buffers[active_buffer]._sub_grid_index = copy_A;
      break;
    }
  }
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

  // MPI initialisation
  MPI_Init(&argc, &argv);
  int rank_get, size_get;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank_get);
  MPI_Comm_size(MPI_COMM_WORLD, &size_get);
  const int MPI_rank = rank_get;
  const int MPI_size = size_get;

  if (MPI_rank == 0) {
    if (MPI_size > 1) {
      logmessage("Running on " << MPI_size << " processes.", 0);
    } else {
      logmessage("Running on a single process.", 0);
    }
  }

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
    return 0;
  }
#endif

#ifdef TEST_COSTVECTOR
  {
    RandomGenerator random_generator;
    CostVector costs(100, 16, 4);
    unsigned long cost_list[100];
    for (size_t i = 0; i < 100; ++i) {
      cost_list[i] = random_generator.get_uniform_random_double() * 0xffffffff;
      costs.add_cost(i, cost_list[i]);
    }
    // add an element with a ridiculously high cost to see how the algorithm
    // copes
    cost_list[42] = 0xfffffffff;
    costs.set_cost(42, cost_list[42]);
    costs.redistribute();
    std::ofstream ofile("cost_test.txt");
    ofile << "# element\tcost\trank\tthread\n";
    for (size_t i = 0; i < 100; ++i) {
      ofile << i << "\t" << cost_list[i] << "\t" << costs.get_process(i) << "\t"
            << costs.get_thread(i) << "\n";
    }
    return 0;
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
  // reemission probability
  //  const double reemission_probability = 0.364;
  const double reemission_probability = 0.;

  // set up the number of threads to use
  // we first determine the number of threads available (either by system
  // default, or because the user has set the OMP_NUM_THREADS environment
  // variable). We then check if a number of threads was specified on the
  // command line. We don't allow setting the number of threads to a value
  // larger than available, and use the available number as default if no value
  // was given on the command line. If the requested number of threads is larger
  // than what is available, we display a message.
  int num_threads_available;
#pragma omp parallel
  {
#pragma omp single
    num_threads_available = omp_get_num_threads();
  }

  int num_threads_request = num_threads_available;
  if (argc > 1) {
    num_threads_request = atoi(argv[1]);
  }

  if (num_threads_request > num_threads_available) {
    logmessage("More threads requested ("
                   << num_threads_request << ") than available ("
                   << num_threads_available
                   << "). Resetting to maximum available number of threads.",
               0);
    num_threads_request = num_threads_available;
  }

  omp_set_num_threads(num_threads_request);
  const int num_threads = num_threads_request;

  logmessage("Running with " << num_threads << " threads.", 0);

  // set up the queues
  std::vector<NewQueue *> new_queues(num_threads, nullptr);
  for (int i = 0; i < num_threads; ++i) {
    new_queues[i] = new NewQueue(2000);
  }

  // set up the memory space
  MemorySpace new_buffers(10000);
  // set up the task space
  ThreadSafeVector<Task> tasks(400000);

  Timer program_timer;
  program_timer.start();

  // set up the grid of smaller grids used for the algorithm
  // each smaller grid stores a fraction of the total grid and has information
  // about the neighbouring subgrids
  std::vector<DensitySubGrid *> gridvec(
      num_subgrid[0] * num_subgrid[1] * num_subgrid[2], nullptr);
  const double subbox_side[3] = {box[3] / num_subgrid[0],
                                 box[4] / num_subgrid[1],
                                 box[5] / num_subgrid[2]};
  const int subbox_ncell[3] = {ncell[0] / num_subgrid[0],
                               ncell[1] / num_subgrid[1],
                               ncell[2] / num_subgrid[2]};
  const unsigned int tot_num_subgrid =
      num_subgrid[0] * num_subgrid[1] * num_subgrid[2];

  // +6 for the 6 subgrid copies we make below
  CostVector costs(tot_num_subgrid + 6, num_threads, MPI_size);

// set up the subgrids (in parallel)
#pragma omp parallel default(shared)
  {
    // id of this specific thread
    const int thread_id = omp_get_thread_num();
    for (int ix = 0; ix < num_subgrid[0]; ++ix) {
      for (int iy = 0; iy < num_subgrid[1]; ++iy) {
        for (int iz = 0; iz < num_subgrid[2]; ++iz) {
          const unsigned int index =
              ix * num_subgrid[1] * num_subgrid[2] + iy * num_subgrid[2] + iz;
          if (costs.get_thread(index) == thread_id) {
            const double subbox[6] = {box[0] + ix * subbox_side[0],
                                      box[1] + iy * subbox_side[1],
                                      box[2] + iz * subbox_side[2],
                                      subbox_side[0],
                                      subbox_side[1],
                                      subbox_side[2]};
            gridvec[index] = new DensitySubGrid(subbox, subbox_ncell);
            DensitySubGrid &this_grid = *gridvec[index];
            // set up neighbouring information. We first make sure all
            // neighbours are initialized to NEIGHBOUR_OUTSIDE, indicating no
            // neighbour
            for (int i = 0; i < 27; ++i) {
              this_grid.set_neighbour(i, NEIGHBOUR_OUTSIDE);
              this_grid.set_active_buffer(i, NEIGHBOUR_OUTSIDE);
            }
            // now set up the correct neighbour relations for the neighbours
            // that exist
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
                      ciy < num_subgrid[1] && ciz >= 0 &&
                      ciz < num_subgrid[2]) {
                    // we use get_output_direction() to get the correct index
                    // for the neighbour
                    // the three_index components will either be
                    //  - -ncell --> negative --> lower limit
                    //  - 0 --> in range --> inside
                    //  - ncell --> upper limit
                    const int three_index[3] = {nix * subbox_ncell[0],
                                                niy * subbox_ncell[1],
                                                niz * subbox_ncell[2]};
                    const int ngbi =
                        this_grid.get_output_direction(three_index);
                    // now get the actual ngb index
                    const unsigned int ngb_index =
                        cix * num_subgrid[1] * num_subgrid[2] +
                        ciy * num_subgrid[2] + ciz;
                    this_grid.set_neighbour(ngbi, ngb_index);
                    const unsigned int active_buffer =
                        new_buffers.get_free_buffer();
                    PhotonBuffer &buffer = new_buffers[active_buffer];
                    buffer._sub_grid_index = ngb_index;
                    buffer._direction = output_to_input_direction(ngbi);
                    this_grid.set_active_buffer(ngbi, active_buffer);
                  } // if ci
                }   // for niz
              }     // for niy
            }       // for nix
          }         // if local index
        }           // for iz
      }             // for iy
    }               // for ix
  }                 // end parallel region

  // duplicated subgrids
  // the first index corresponds to the original, the second index to the copy
  // at the start of each iteration, the copy needs to be synced with the
  // original, while at the end of the iteration the contributions from the copy
  // need to be added to the original
  std::vector<std::pair<unsigned int, unsigned int>> duplicates;

  // make copies of the 2 central subgrids, so that multiple threads can work
  // on them simultaneously
  make_copy(29, gridvec, new_buffers, duplicates);
  make_copy(30, gridvec, new_buffers, duplicates);
  make_copy(29, gridvec, new_buffers, duplicates);
  make_copy(30, gridvec, new_buffers, duplicates);
  make_copy(29, gridvec, new_buffers, duplicates);
  make_copy(30, gridvec, new_buffers, duplicates);
  ensure_neighbours(29, 30, 60, 61, gridvec, new_buffers);
  ensure_neighbours(29, 30, 62, 63, gridvec, new_buffers);
  ensure_neighbours(29, 30, 64, 65, gridvec, new_buffers);

  std::ifstream initial_costs("costs_00.txt");
  if (initial_costs.good()) {
    // use cost information from a previous run as initial guess for the cost
    // skip the initial comment line
    std::string line;
    std::getline(initial_costs, line);
    unsigned int index;
    unsigned long cost;
    int rank, thread;
    for (unsigned int i = 0; i < gridvec.size(); ++i) {
      initial_costs >> index >> cost >> rank >> thread;
      myassert(index == i, "Wrong index!");
      costs.add_cost(index, cost);
    }
  } else {
    // no previous information available: use a very basic initial cost guess
    // (good enough to get the idle fraction below 50%)
    for (unsigned int i = 0; i < gridvec.size(); ++i) {
      costs.add_cost(i, 1);
    }
    costs.add_cost(29, 10000);
    costs.add_cost(60, 10000);
    costs.add_cost(62, 10000);
    costs.add_cost(64, 10000);
    costs.add_cost(30, 11000);
    costs.add_cost(61, 11000);
    costs.add_cost(63, 11000);
    costs.add_cost(65, 11000);
  }
  costs.redistribute();

  // get the central subgrid indices
  const unsigned int central_index[4] = {30, 61, 63, 65};

  // set up the random number generators
  std::vector<RandomGenerator> random_generator(num_threads);
  for (int i = 0; i < num_threads; ++i) {
    // make sure every thread on every process has a different seed
    random_generator[i].set_seed(42 + MPI_rank * num_threads + i);
  }

  // now for the main loop. This loop
  //  - shoots num_photon photons through the grid to get intensity estimates
  //  - computes the ionization equilibrium
  for (unsigned int iloop = 0; iloop < number_of_iterations; ++iloop) {
    const int central_queue[4] = {
        costs.get_thread(central_index[0]), costs.get_thread(central_index[1]),
        costs.get_thread(central_index[2]), costs.get_thread(central_index[3])};
    // STEP 0: log output
    logmessage("Loop " << iloop + 1, 0);

    for (unsigned int i = 0; i < duplicates.size(); ++i) {
      logmessage("Updating neutral fractions for " << duplicates[i].second
                                                   << ", copy of "
                                                   << duplicates[i].first,
                 0);
      gridvec[duplicates[i].second]->update_neutral_fractions(
          *gridvec[duplicates[i].first]);
    }

    // STEP 1: photon shooting
    // control variables (these are shared and updated atomically):
    //  - number of photon packets that has been created at the source
    unsigned int num_photon_done = 0;
    //  - number of active photon buffers across all threads
    unsigned int num_active_buffers = 0;
    //  - number of empty assigned buffers
    unsigned int num_empty = gridvec.size() * 27;
    const unsigned int num_empty_target = gridvec.size() * 27;
#pragma omp parallel default(shared)
    {
      // id of this specific thread
      const int thread_id = omp_get_thread_num();
      PhotonBuffer local_buffers[27];
      bool local_buffer_flags[27];
      for (int i = 0; i < 27; ++i) {
        local_buffers[i]._direction = output_to_input_direction(i);
        local_buffers[i]._actual_size = 0;
      }
      // set up the initial source photon task
      {
        const size_t task_index = tasks.get_free_element();
        tasks[task_index]._type = TASKTYPE_SOURCE_PHOTON;
        // buffer is ready to be processed: add to the queue
        new_queues[thread_id]->add_task(task_index);
      }
      // this loop is repeated until all photons have been shot
      while (num_photon_done < num_photon || num_active_buffers > 0 ||
             num_empty < num_empty_target) {

        logmessage("Subloop (num_active_buffers: "
                       << num_active_buffers
                       << ", num_photon_done: " << num_photon_done
                       << ", num_empty: " << num_empty << ")",
                   1);

        unsigned int current_index = new_queues[thread_id]->get_task();
        // task activation: if no task is found, try to launch a photon buffer
        // that is not yet full
        if (current_index == NO_TASK) {
          // try to activate a non-full buffer
          // note that we only try to access thread-local information, so as
          // long as we don't allow task stealing, this will be thread-safe
          unsigned int i = 0;
          while (i < gridvec.size() && current_index == NO_TASK) {
            if (costs.get_thread(i) == thread_id) {
              int j = 0;
              while (j < 27 && current_index == NO_TASK) {
                if (gridvec[i]->get_neighbour(j) != NEIGHBOUR_OUTSIDE &&
                    new_buffers[gridvec[i]->get_active_buffer(j)]._actual_size >
                        0) {
                  const unsigned int non_full_index =
                      gridvec[i]->get_active_buffer(j);
                  const unsigned int new_index = new_buffers.get_free_buffer();
                  new_buffers[new_index]._sub_grid_index =
                      new_buffers[non_full_index]._sub_grid_index;
                  new_buffers[new_index]._direction =
                      new_buffers[non_full_index]._direction;
                  gridvec[i]->set_active_buffer(j, new_index);
                  // add buffer to queue
                  atomic_pre_increment(num_active_buffers);
                  const size_t task_index = tasks.get_free_element();
                  if (j > 0) {
                    tasks[task_index]._type = TASKTYPE_PHOTON_TRAVERSAL;
                  } else {
                    tasks[task_index]._type = TASKTYPE_PHOTON_REEMIT;
                  }
                  tasks[task_index]._cell =
                      new_buffers[non_full_index]._sub_grid_index;
                  tasks[task_index]._buffer = non_full_index;
                  new_queues[costs.get_thread(
                                 new_buffers[non_full_index]._sub_grid_index)]
                      ->add_task(task_index);
                  // we have created a new empty buffer
                  atomic_pre_increment(num_empty);
                  // try again to get a task (probably the one we just created,
                  // unless another thread gave us some work in the meantime)
                  current_index = new_queues[thread_id]->get_task();
                }
                ++j;
              }
            }
            ++i;
          }
        }
        // keep processing buffers until the queue is empty
        while (current_index != NO_TASK) {
          Task &task = tasks[current_index];
          if (task._type == TASKTYPE_SOURCE_PHOTON) {
            task.start(thread_id);
            if (num_photon_done < num_photon) {
              const unsigned int num_photon_done_now =
                  atomic_post_add(num_photon_done, PHOTONBUFFER_SIZE);
              if (num_photon_done_now < num_photon) {
                // spawn a new source photon task
                {
                  const size_t task_index = tasks.get_free_element();
                  tasks[task_index]._type = TASKTYPE_SOURCE_PHOTON;
                  // buffer is ready to be processed: add to the queue
                  new_queues[thread_id]->add_task(task_index);
                }

                // we will create a new buffer
                atomic_pre_increment(num_active_buffers);

                // if this is the last buffer: cap the total number of photons
                // to the requested value
                // (note that num_photon_done_now is the number of photons that
                //  was generated BEFORE this task, after this task,
                //  num_photon_done_now + PHOTONBUFFER_SIZE photons will have
                //  been generated, unless this number is capped)
                unsigned int num_photon_this_loop = PHOTONBUFFER_SIZE;
                if (num_photon_done_now + PHOTONBUFFER_SIZE > num_photon) {
                  num_photon_this_loop += (num_photon - num_photon_done_now);
                }

                // get a free photon buffer in the central queue
                unsigned int buffer_index = new_buffers.get_free_buffer();
                // no need to lock this buffer
                PhotonBuffer &input_buffer = new_buffers[buffer_index];
                int which_central_index =
                    random_generator[thread_id].get_uniform_random_double() *
                    4.;
                myassert(which_central_index >= 0 && which_central_index < 4,
                         "Oopsie!");
                unsigned int this_central_index =
                    central_index[which_central_index];
                fill_buffer(input_buffer, num_photon_this_loop,
                            random_generator[thread_id], this_central_index);

                const size_t task_index = tasks.get_free_element();
                tasks[task_index]._type = TASKTYPE_PHOTON_TRAVERSAL;
                tasks[task_index]._cell = this_central_index;
                tasks[task_index]._buffer = buffer_index;
                // buffer is ready to be processed: add to the queue
                new_queues[central_queue[which_central_index]]->add_task(
                    task_index);
              }
            }
            task.stop();
          } else if (task._type == TASKTYPE_PHOTON_TRAVERSAL) {
            unsigned long task_start, task_end;
            task_tick(task_start);
            task.start(thread_id);
            const unsigned int current_buffer_index = task._buffer;
            // we don't allow task-stealing, so no need to lock anything just
            // now
            PhotonBuffer &buffer = new_buffers[current_buffer_index];
            const unsigned int igrid = buffer._sub_grid_index;
            DensitySubGrid &this_grid = *gridvec[buffer._sub_grid_index];
            // prepare output buffers
            for (int i = 0; i < 27; ++i) {
              const unsigned int ngb = this_grid.get_neighbour(i);
              if (ngb != NEIGHBOUR_OUTSIDE) {
                local_buffer_flags[i] = true;
                local_buffers[i]._actual_size = 0;
              } else {
                local_buffer_flags[i] = false;
              }
            }
            if (reemission_probability == 0.) {
              local_buffer_flags[TRAVELDIRECTION_INSIDE] = false;
            }
            do_photon_traversal(buffer, this_grid, local_buffers,
                                local_buffer_flags);
            // add none empty buffers to the appropriate queues
            // we go backwards, so that the local queue is added to the task
            // list last
            for (int i = 27; i >= 0; --i) {
              if (local_buffer_flags[i]) {
                if (local_buffers[i]._actual_size > 0) {
                  const unsigned int ngb = this_grid.get_neighbour(i);
                  unsigned int new_index = this_grid.get_active_buffer(i);
                  if (new_buffers[new_index]._actual_size == 0) {
                    // we are adding photons to an empty buffer
                    atomic_pre_decrement(num_empty);
                  }
                  unsigned int add_index =
                      new_buffers.add_photons(new_index, local_buffers[i]);
                  if (add_index != new_index) {
                    // add buffer to queue
                    atomic_pre_increment(num_active_buffers);
                    const size_t task_index = tasks.get_free_element();
                    if (i > 0) {
                      tasks[task_index]._type = TASKTYPE_PHOTON_TRAVERSAL;
                    } else {
                      tasks[task_index]._type = TASKTYPE_PHOTON_REEMIT;
                    }
                    tasks[task_index]._cell =
                        new_buffers[new_index]._sub_grid_index;
                    tasks[task_index]._buffer = new_index;
                    new_queues[costs.get_thread(ngb)]->add_task(task_index);
                    myassert(new_buffers[add_index]._sub_grid_index == ngb,
                             "Wrong subgrid");
                    myassert(new_buffers[add_index]._direction ==
                                 output_to_input_direction(i),
                             "Wrong direction");
                    this_grid.set_active_buffer(i, add_index);
                    if (new_buffers[add_index]._actual_size == 0) {
                      // we have created a new empty buffer
                      atomic_pre_increment(num_empty);
                    }
                  }
                }
              }
            }
            // delete the original buffer
            atomic_pre_subtract(num_active_buffers);
            new_buffers.free_buffer(current_buffer_index);
            task.stop();
            task_tick(task_end);
            costs.add_cost(igrid, task_end - task_start);
          } else if (task._type == TASKTYPE_PHOTON_REEMIT) {
            unsigned long task_start, task_end;
            task_tick(task_start);
            task.start(thread_id);
            const unsigned int current_buffer_index = task._buffer;
            PhotonBuffer &buffer = new_buffers[current_buffer_index];
            do_reemission(buffer, random_generator[thread_id],
                          reemission_probability);
            const size_t task_index = tasks.get_free_element();
            tasks[task_index]._type = TASKTYPE_PHOTON_TRAVERSAL;
            tasks[task_index]._cell = task._cell;
            tasks[task_index]._buffer = current_buffer_index;
            new_queues[thread_id]->add_task(task_index);
            task.stop();
            task_tick(task_end);
            costs.add_cost(task._cell, task_end - task_start);
          } else {
            logmessage("Unknown task!", 0);
          }
          // try to get a new buffer from the local queue
          current_index = new_queues[thread_id]->get_task();
        }
      }
    } // parallel region
    logmessage("Total number of tasks: " << tasks.size(), 0);

    for (unsigned int i = 0; i < duplicates.size(); ++i) {
      logmessage("Updating ionization integrals for " << duplicates[i].first
                                                      << " using copy "
                                                      << duplicates[i].second,
                 0);
      gridvec[duplicates[i].first]->update_intensities(
          *gridvec[duplicates[i].second]);
    }

    // STEP 2: update the ionization structure for each subgrid
    unsigned int igrid = 0;
#pragma omp parallel default(shared)
    {
      while (igrid < tot_num_subgrid) {
        const unsigned int current_igrid = atomic_post_increment(igrid);
        if (current_igrid < tot_num_subgrid) {
          gridvec[current_igrid]->compute_neutral_fraction(num_photon);
        }
      }
    }

    // output useful information about this iteration
    output_tasks(iloop, tasks);
    output_costs(iloop, gridvec.size(), num_threads, costs);

    // clear task buffer
    tasks.clear();
    // redistribute the subgrids among the threads to balance the computational
    // costs (based on this iteration)
    costs.redistribute();
  } // main loop

  program_timer.stop();
  logmessage("Total program time: " << program_timer.value() << " s.", 0);

  struct rusage resource_usage;
  getrusage(RUSAGE_SELF, &resource_usage);
  size_t max_memory =
      static_cast<size_t>(resource_usage.ru_maxrss) * static_cast<size_t>(1024);
  logmessage("Maximum memory usage: " << human_readable_bytes(max_memory), 0);

  // OUTPUT:
  //  - ASCII output (for the VisIt plot script)
  std::ofstream ofile("intensities.txt");
  for (unsigned int igrid = 0; igrid < tot_num_subgrid; ++igrid) {
    gridvec[igrid]->print_intensities(ofile);
  }
  ofile.close();
  //  - binary output (for the Python plot script)
  std::ofstream bfile("intensities.dat");
  for (unsigned int igrid = 0; igrid < tot_num_subgrid; ++igrid) {
    gridvec[igrid]->output_intensities(bfile);
  }
  bfile.close();

  // garbage collection
  for (unsigned int igrid = 0; igrid < gridvec.size(); ++igrid) {
    delete gridvec[igrid];
  }

  return 0;
}
