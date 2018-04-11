/*******************************************************************************
 * This file is part of CMacIonize
 * Copyright (C) 2017, 2018 Bert Vandenbroucke (bert.vandenbroucke@gmail.com)
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

// Space for ideas, TODOs...
//
// - check if MPI_Testany is actually faster than an own implementation, and if
//   it favours specific entries in the list (since we always only remove 1
//   element and if this is always the first, this might have negative impact
//   on scaling)
//
// Important knowledge: if you compile with debug symbols, the following
// command can link addresses in error output to file positions:
//  addr2line 0x40fafc -e testDensitySubGrid
//
// Don't use logmessage inside locked regions!! It contains an omp single
// statement that automatically locks, and the locks interfere with region
// locks.
// Also: avoid using logmessage anywhere near the end of a parallel block, as
// it will deadlock if one of the threads already left the parallel region (or
// cause extremely weird behaviour)
//
// GCC thread sanitizer does not seem to work: it gives false positives on
// OpenMP functions...
//
// Send tasks do not have to be executed by the thread that spawns them and
// could act as filler tasks for idle threads. However, to enable this, we would
// need a general queue (which could also contain the photon source tasks)...
//
// Receive tasks cannot be put in a queue (at least not as a general "check for
// incoming communications" task). We could let the check for incoming
// communications spawn receive tasks that are put in the general queue though.

// Defines: we do these first, as some includes depend on them

/*! @brief Output log level. The higher the value, the more stuff is printed to
 *  the stderr. Comment to disable logging altogether. */
#define LOG_OUTPUT 1

/*! @brief Uncomment this to enable run time assertions. */
#define DO_ASSERTS

/*! @brief Enable this to activate task output. */
#define TASK_OUTPUT

/*! @brief Enable this to activate cost output. */
#define COST_OUTPUT

/*! @brief Enable this to activate message output. */
#define MESSAGE_OUTPUT

/*! @brief Enable this to only execute a single iteration (until cost and task
 *  output is written). */
//#define SINGLE_ITERATION

/*! @brief Enable this to rebalance subgrids across threads and processes in
 *  between iterations. */
//#define DO_REBALANCING

#ifdef TASK_OUTPUT
// activate task output in Task.hpp
#define TASK_PLOT
#endif

// global variables, as we need them in the log macro
int MPI_rank, MPI_size;

// Project includes
#include "CommandLineParser.hpp"
#include "CostVector.hpp"
#include "DensitySubGrid.hpp"
#include "Log.hpp"
#include "MPIMessage.hpp"
#include "MemorySpace.hpp"
#include "NewQueue.hpp"
#include "PhotonBuffer.hpp"
#include "RandomGenerator.hpp"
#include "Task.hpp"
#include "TaskSpace.hpp"
#include "Timer.hpp"
#include "Utilities.hpp"
#include "YAMLDictionary.hpp"

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
 * @brief Write a file with the start and end times of all tasks.
 *
 * @param iloop Iteration number (added to file name).
 * @param tasks Tasks to print.
 */
inline void output_tasks(const unsigned int iloop, const TaskSpace &tasks) {
#ifdef TASK_OUTPUT
  // compose the file name
  std::stringstream filename;
  filename << "tasks_";
  filename.fill('0');
  filename.width(2);
  filename << iloop;
  filename << ".txt";

  // now output
  // each process outputs its own tasks in turn, process 0 is responsible for
  // creating the file and the other processes append to it
  for (int irank = 0; irank < MPI_size; ++irank) {
    // only the active process writes
    if (irank == MPI_rank) {
      // the file mode depends on the rank
      std::ios_base::openmode mode;
      if (irank == 0) {
        // rank 0 creates (or overwrites) the file
        mode = std::ofstream::trunc;
      } else {
        // all other ranks append to it
        mode = std::ofstream::app;
      }
      // now open the file
      std::ofstream ofile(filename.str(), mode);

      // rank 0 writes the header
      if (irank == 0) {
        ofile << "# rank\tthread\tstart\tstop\ttype\n";
      }

      // write the task info
      const size_t tsize = tasks.size();
      for (size_t i = 0; i < tsize; ++i) {
        const Task &task = tasks[i];
        ofile << MPI_rank << "\t" << task._thread_id << "\t" << task._start_time
              << "\t" << task._end_time << "\t" << task._type << "\n";
      }
    }
    // only one process at a time is allowed to write
    MPI_Barrier(MPI_COMM_WORLD);
  }
#endif
}

/**
 * @brief Write files with cost information for an iteration.
 *
 * @param iloop Iteration number (added to file names).
 * @param ngrid Number of subgrids.
 * @param costs CostVector to print.
 * @param copies List that links subgrids to copies.
 * @param original List that links subgrid copies to originals.
 */
inline void output_costs(const unsigned int iloop, const unsigned int ngrid,
                         const CostVector &costs,
                         const std::vector< unsigned int > &copies,
                         const std::vector< unsigned int > &originals) {
#ifdef COST_OUTPUT
  // first compose the file name
  std::stringstream filename;
  filename << "costs_";
  filename.fill('0');
  filename.width(2);
  filename << iloop;
  filename << ".txt";

  // now output
  // each process outputs its own costs in turn, process 0 is responsible for
  // creating the file and the other processes append to it
  // note that in principle each process holds all cost information. However,
  // the actual costs will only be up to date on the local process that holds
  // the subgrid.
  for (int irank = 0; irank < MPI_size; ++irank) {
    if (irank == MPI_rank) {
      // the file mode depends on the rank
      std::ios_base::openmode mode;
      if (irank == 0) {
        // rank 0 creates (or overwrites) the file
        mode = std::ofstream::trunc;
      } else {
        // all other ranks append to it
        mode = std::ofstream::app;
      }
      // now open the file
      std::ofstream ofile(filename.str(), mode);

      // rank 0 writes the file header
      if (irank == 0) {
        ofile << "# subgrid\tcomputational cost\tphoton cost\tsource "
                 "cost\trank\tthread\n";
      }

      // output the cost information
      for (unsigned int i = 0; i < ngrid; ++i) {
        // only output local information
        if (costs.get_process(i) == MPI_rank) {
          ofile << i << "\t" << costs.get_computational_cost(i) << "\t"
                << costs.get_photon_cost(i) << "\t" << costs.get_source_cost(i)
                << "\t" << costs.get_process(i) << "\t" << costs.get_thread(i)
                << "\n";
        }
        if (copies[i] < 0xffffffff) {
          unsigned int copy = copies[i];
          while (copy - ngrid < originals.size() &&
                 originals[copy - ngrid] == i) {
            // only output local information
            if (costs.get_process(copy) == MPI_rank) {
              ofile << i << "\t" << costs.get_computational_cost(copy) << "\t"
                    << costs.get_photon_cost(copy) << "\t"
                    << costs.get_source_cost(copy) << "\t"
                    << costs.get_process(copy) << "\t" << costs.get_thread(copy)
                    << "\n";
            }
            ++copy;
          }
        }
      }
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }
#endif
}

/**
 * @brief Write file with communication information for an iteration.
 *
 * @param iloop Iteration number (added to file names).
 * @param ngrid Number of subgrids.
 * @param nthread Number of threads.
 * @param costs CostVector to print.
 * @param copies List that links subgrids to copies.
 * @param original List that links subgrid copies to originals.
 */
inline void output_messages(const unsigned int iloop,
                            const std::vector< MPIMessage > &message_log,
                            const size_t message_log_size) {
#ifdef MESSAGE_OUTPUT
  // first compose the file name
  std::stringstream filename;
  filename << "messages_";
  filename.fill('0');
  filename.width(2);
  filename << iloop;
  filename << ".txt";

  // now output
  // each process outputs its own costs in turn, process 0 is responsible for
  // creating the file and the other processes append to it
  // note that in principle each process holds all cost information. However,
  // the actual costs will only be up to date on the local process that holds
  // the subgrid.
  for (int irank = 0; irank < MPI_size; ++irank) {
    if (irank == MPI_rank) {
      // the file mode depends on the rank
      std::ios_base::openmode mode;
      if (irank == 0) {
        // rank 0 creates (or overwrites) the file
        mode = std::ofstream::trunc;
      } else {
        // all other ranks append to it
        mode = std::ofstream::app;
      }
      // now open the file
      std::ofstream ofile(filename.str(), mode);

      // rank 0 writes the file header
      if (irank == 0) {
        ofile << "# rank\tthread\ttype\tother_rank\ttag\ttimestamp\n";
      }

      // output the cost information
      for (size_t i = 0; i < message_log_size; ++i) {
        const MPIMessage &message = message_log[i];
        ofile << MPI_rank << "\t" << message._thread << "\t" << message._type
              << "\t" << message._rank << "\t" << message._tag << "\t"
              << message._timestamp << "\n";
      }
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }
#endif
}

/**
 * @brief Output the neutral fractions for inspection of the physical result.
 *
 * @oaran costs CostVector.
 * @param gridvec Subgrids.
 * @param tot_num_subgrid Total number of original subgrids.
 */
inline void
output_neutral_fractions(const CostVector &costs,
                         const std::vector< DensitySubGrid * > &gridvec,
                         const unsigned int tot_num_subgrid) {

  //  - ASCII output (for the VisIt plot script)
  for (int irank = 0; irank < MPI_size; ++irank) {
    // only the active process writes
    if (irank == MPI_rank) {
      // the file mode depends on the rank
      std::ios_base::openmode mode;
      if (irank == 0) {
        // rank 0 creates (or overwrites) the file
        mode = std::ofstream::trunc;
      } else {
        // all other ranks append to it
        mode = std::ofstream::app;
      }
      // now open the file
      std::ofstream ofile("intensities.txt", mode);

      // write the neutral fractions
      for (unsigned int igrid = 0; igrid < tot_num_subgrid; ++igrid) {
        // only the owning process writes
        if (costs.get_process(igrid) == MPI_rank) {
          gridvec[igrid]->print_intensities(ofile);
        }
      }
    }
    // only one process at a time is allowed to write
    MPI_Barrier(MPI_COMM_WORLD);
  }

  //  - binary output (for the Python plot script)
  for (int irank = 0; irank < MPI_size; ++irank) {
    // only the active process writes
    if (irank == MPI_rank) {
      // the file mode depends on the rank
      std::ios_base::openmode mode;
      if (irank == 0) {
        // rank 0 creates (or overwrites) the file
        mode = std::ofstream::trunc;
      } else {
        // all other ranks append to it
        mode = std::ofstream::app;
      }
      // now open the file
      std::ofstream ofile("intensities.dat", mode);

      // write the neutral fractions
      for (unsigned int igrid = 0; igrid < tot_num_subgrid; ++igrid) {
        // only the owning process writes
        if (costs.get_process(igrid) == MPI_rank) {
          gridvec[igrid]->output_intensities(ofile);
        }
      }
    }
    // only one process at a time is allowed to write
    MPI_Barrier(MPI_COMM_WORLD);
  }
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

  // draw two pseudo random numbers
  const double cost = 2. * random_generator.get_uniform_random_double() - 1.;
  const double phi = 2. * M_PI * random_generator.get_uniform_random_double();

  // now use them to get all directional angles
  const double sint = std::sqrt(std::max(1. - cost * cost, 0.));
  const double cosp = std::cos(phi);
  const double sinp = std::sin(phi);

  // set the direction...
  direction[0] = sint * cosp;
  direction[1] = sint * sinp;
  direction[2] = cost;

  // ...and its inverse
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

  // set general buffer information
  buffer._actual_size = number_of_photons;
  buffer._sub_grid_index = source_index;
  buffer._direction = TRAVELDIRECTION_INSIDE;

  // draw random photons and store them in the buffer
  for (unsigned int i = 0; i < number_of_photons; ++i) {

    Photon &photon = buffer._photons[i];

    // initial position: we currently assume a single source at the origin
    photon._position[0] = 0.;
    photon._position[1] = 0.;
    photon._position[2] = 0.;

    // initial direction: isotropic distribution
    get_random_direction(random_generator, photon._direction,
                         photon._inverse_direction);

    // we currently assume equal weight for all photons
    photon._weight = 1.;

    // current optical depth (always zero) and target (exponential distribution)
    photon._current_optical_depth = 0.;
    photon._target_optical_depth =
        -std::log(random_generator.get_uniform_random_double());

    // this is the fixed cross section we use for the moment
    photon._photoionization_cross_section = 6.3e-22;

    // make sure the photon is moving in *a* direction
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

  // make sure all output buffers are empty initially
  for (int i = 0; i < 27; ++i) {
    myassert(!output_buffer_flags[i] || output_buffers[i]._actual_size == 0,
             "Non-empty starting output buffer!");
  }

  // now loop over the input buffer photons and traverse them one by one
  for (unsigned int i = 0; i < input_buffer._actual_size; ++i) {

    // active photon
    Photon &photon = input_buffer._photons[i];

    // make sure the photon is moving in *a* direction
    myassert(photon._direction[0] != 0. || photon._direction[1] != 0. ||
                 photon._direction[2] != 0.,
             "size: " << input_buffer._actual_size);

    // traverse the photon through the active subgrid
    const int result = subgrid.interact(photon, input_buffer._direction);

    // check that the photon ended up in a valid output buffer
    myassert(result >= 0 && result < 27, "fail");

    // add the photon to an output buffer, if it still exists (if the
    // corresponding output buffer does not exist, this means the photon left
    // the simulation box)
    if (output_buffer_flags[result]) {

      // get the correct output buffer
      PhotonBuffer &output_buffer = output_buffers[result];

      // add the photon
      const unsigned int index = output_buffer._actual_size;
      output_buffer._photons[index] = photon;

      // make sure we actually added this photon
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

      // increase the active size of the output buffer by 1 (we added a photon)
      ++output_buffer._actual_size;

      // check that the output buffer did not overflow
      myassert(output_buffer._actual_size <= PHOTONBUFFER_SIZE,
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

  // loop over the active buffer and decide which photons to reemit
  // we will be overwriting non-reemitted photons, that's why we need two index
  // variables
  unsigned int index = 0;
  for (unsigned int i = 0; i < buffer._actual_size; ++i) {
    // only a fraction (= 'reemission_probability') of the photons is actually
    // reemitted
    if (random_generator.get_uniform_random_double() < reemission_probability) {
      // give the photon a new random isotropic direction
      Photon &photon = buffer._photons[i];
      get_random_direction(random_generator, photon._direction,
                           photon._inverse_direction);
      // reset the current optical depth (always zero) and target
      photon._current_optical_depth = 0.;
      photon._target_optical_depth =
          -std::log(random_generator.get_uniform_random_double());
      // NOTE: we can never overwrite a photon that should be preserved (we
      // either overwrite the photon itself, or a photon that was not reemitted)
      buffer._photons[index] = photon;
      ++index;
    }
  }
  // update the active size of the buffer: some photons were not reemitted, so
  // the active size will shrink
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
 * @param copies Index of the first copy of each subgrid.
 */
inline void create_copies(std::vector< DensitySubGrid * > &gridvec,
                          std::vector< unsigned char > &levels,
                          MemorySpace &new_buffers,
                          std::vector< unsigned int > &originals,
                          std::vector< unsigned int > &copies) {

  // we need to do 2 loops:
  //  - one loop to create the copies and store the offset of the first copy
  //    for each subgrid
  //  - a second loop that sets the neighbours (and has access to all necessary
  //    copies to set inter-copy neighbour relations)

  // we need to store the original number of subgrids for reference
  const unsigned int number_of_unique_subgrids = gridvec.size();

  // array to store the offsets of new copies in
  copies.resize(gridvec.size(), 0xffffffff);
  for (unsigned int i = 0; i < number_of_unique_subgrids; ++i) {
    const unsigned char level = levels[i];
    const unsigned int number_of_copies = 1 << level;
    // create the copies
    if (number_of_copies > 1) {
      copies[i] = gridvec.size();
    }
    for (unsigned int j = 1; j < number_of_copies; ++j) {
      gridvec.push_back(new DensitySubGrid(*gridvec[i]));
      originals.push_back(i);
    }
  }

  // neighbour setting
  for (unsigned int i = 0; i < number_of_unique_subgrids; ++i) {
    const unsigned char level = levels[i];
    const unsigned int number_of_copies = 1 << level;
    // first do the self-reference for each copy (if there are copies)
    for (unsigned int j = 1; j < number_of_copies; ++j) {
      const unsigned int copy = copies[i] + j - 1;
      gridvec[copy]->set_neighbour(0, copy);
      const unsigned int active_buffer = new_buffers.get_free_buffer();
      new_buffers[active_buffer]._sub_grid_index = copy;
      new_buffers[active_buffer]._direction = TRAVELDIRECTION_INSIDE;
      gridvec[copy]->set_active_buffer(0, active_buffer);
    }
    // now do the actual neighbours
    for (int j = 1; j < 27; ++j) {
      const unsigned int original_ngb = gridvec[i]->get_neighbour(j);
      if (original_ngb != NEIGHBOUR_OUTSIDE) {
        const unsigned char ngb_level = levels[original_ngb];
        // check how the neighbour level compares to the subgrid level
        if (ngb_level == level) {
          // same, easy: just make copies mutual neighbours
          // and leave the original grid as is
          for (unsigned int k = 1; k < number_of_copies; ++k) {
            const unsigned int copy = copies[i] + k - 1;
            const unsigned int ngb_copy = copies[original_ngb] + k - 1;
            gridvec[copy]->set_neighbour(j, ngb_copy);
            const unsigned int active_buffer = new_buffers.get_free_buffer();
            new_buffers[active_buffer]._sub_grid_index = ngb_copy;
            new_buffers[active_buffer]._direction =
                output_to_input_direction(j);
            gridvec[copy]->set_active_buffer(j, active_buffer);
          }
        } else {
          // not the same: there are 2 options
          if (level > ngb_level) {
            // we have less neighbour copies, so some of our copies need to
            // share the same neighbour
            // some of our copies might also need to share the original
            // neighbour
            const unsigned int number_of_ngb_copies = 1 << (level - ngb_level);
            for (unsigned int k = 1; k < number_of_copies; ++k) {
              const unsigned int copy = copies[i] + k - 1;
              // this term will round down, which is what we want
              const unsigned int ngb_index = k / number_of_ngb_copies;
              const unsigned int ngb_copy =
                  (ngb_index > 0) ? copies[original_ngb] + ngb_index - 1
                                  : original_ngb;
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
              const unsigned int copy = copies[i] + k - 1;
              // the second term will skip some neighbour copies, which is what
              // we want
              const unsigned int ngb_copy =
                  copies[original_ngb] + (k - 1) * number_of_own_copies;
              gridvec[copy]->set_neighbour(j, ngb_copy);
              const unsigned int active_buffer = new_buffers.get_free_buffer();
              new_buffers[active_buffer]._sub_grid_index = ngb_copy;
              new_buffers[active_buffer]._direction =
                  output_to_input_direction(j);
              gridvec[copy]->set_active_buffer(j, active_buffer);
            }
          }
        }
      } else {
        // flag this neighbour as NEIGHBOUR_OUTSIDE for all copies
        for (unsigned int k = 1; k < number_of_copies; ++k) {
          const unsigned int copy = copies[i] + k - 1;
          gridvec[copy]->set_neighbour(j, NEIGHBOUR_OUTSIDE);
        }
      }
    }
  }
}

/**
 * @brief Initialize MPI.
 *
 * @param argc Number of command line arguments.
 * @param argv Command line arguments.
 * @param MPI_rank Variable to store the active MPI rank in.
 * @param MPI_size Variable to store the total MPI size in.
 */
inline void initialize_MPI(int &argc, char **argv, int &MPI_rank,
                           int &MPI_size) {

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &MPI_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &MPI_size);

  if (MPI_rank == 0) {
    if (MPI_size > 1) {
      logmessage("Running on " << MPI_size << " processes.", 0);
    } else {
      logmessage("Running on a single process.", 0);
    }
  }
}

/**
 * @brief Parse the command line options.
 *
 * @param argc Number of command line options.
 * @param argv Command line options.
 * @param num_threads_request Variable to store the requested number of threads
 * in.
 * @param paramfile_name Variable to store the parameter file name in.
 */
inline void parse_command_line(int argc, char **argv, int &num_threads_request,
                               std::string &paramfile_name) {

  CommandLineParser commandlineparser("testDensitySubGrid");
  commandlineparser.add_required_option< int_fast32_t >(
      "threads", 't', "Number of shared memory threads to use.");
  commandlineparser.add_required_option< std::string >(
      "params", 'p', "Name of the parameter file.");
  commandlineparser.parse_arguments(argc, argv);

  num_threads_request = commandlineparser.get_value< int_fast32_t >("threads");
  paramfile_name = commandlineparser.get_value< std::string >("params");
}

/**
 * @brief Read the parameter file.
 *
 * @param paramfile_name Name of the parameter file.
 * @param box Variable to store the box anchor and size in.
 * @param reemission_probability Variable to store the reemission probability
 * in.
 * @param ncell Variable to store the total number of cells in.
 * @param num_subgrid Variable to store the total number of subgrids in.
 * @param num_photon Variable to store the number of photon packets in.
 * @param number_of_iterations Variable to store the number of iterations in.
 * @param queue_size_per_thread Variable to store the size of the queue for
 * each thread in.
 * @param memoryspace_size Variable to store the size of the memory space in.
 * @param number_of_tasks Variable to store the size of the task space in.
 * @param MPI_buffer_size Variable to store the size of the MPI buffer in.
 * @param copy_factor Variable to store the copy factor in.
 * @param general_queue_size General queue size.
 */
inline void read_parameters(std::string paramfile_name, double box[6],
                            double &reemission_probability, int ncell[3],
                            int num_subgrid[3], unsigned int &num_photon,
                            unsigned int &number_of_iterations,
                            unsigned int &queue_size_per_thread,
                            unsigned int &memoryspace_size,
                            unsigned int &number_of_tasks,
                            unsigned int &MPI_buffer_size, double &copy_factor,
                            unsigned int &general_queue_size) {

  std::ifstream paramfile(paramfile_name);
  if (!paramfile) {
    cmac_error("Unable to open parameter file \"%s\"!", paramfile_name.c_str());
  }

  YAMLDictionary parameters(paramfile);

  const CoordinateVector<> param_box_anchor =
      parameters.get_physical_vector< QUANTITY_LENGTH >("box:anchor");
  const CoordinateVector<> param_box_sides =
      parameters.get_physical_vector< QUANTITY_LENGTH >("box:sides");

  box[0] = param_box_anchor.x();
  box[1] = param_box_anchor.y();
  box[2] = param_box_anchor.z();
  box[3] = param_box_sides.x();
  box[4] = param_box_sides.y();
  box[5] = param_box_sides.z();

  reemission_probability =
      parameters.get_value< double >("reemission_probability");

  const CoordinateVector< int > param_ncell =
      parameters.get_value< CoordinateVector< int > >("ncell");

  ncell[0] = param_ncell.x();
  ncell[1] = param_ncell.y();
  ncell[2] = param_ncell.z();

  const CoordinateVector< int > param_num_subgrid =
      parameters.get_value< CoordinateVector< int > >("num_subgrid");

  num_subgrid[0] = param_num_subgrid.x();
  num_subgrid[1] = param_num_subgrid.y();
  num_subgrid[2] = param_num_subgrid.z();

  num_photon = parameters.get_value< unsigned int >("num_photon");
  number_of_iterations =
      parameters.get_value< unsigned int >("number_of_iterations");

  queue_size_per_thread =
      parameters.get_value< unsigned int >("queue_size_per_thread");
  memoryspace_size = parameters.get_value< unsigned int >("memoryspace_size");
  number_of_tasks = parameters.get_value< unsigned int >("number_of_tasks");
  MPI_buffer_size = parameters.get_value< unsigned int >("MPI_buffer_size");
  copy_factor = parameters.get_value< double >("copy_factor");

  general_queue_size =
      parameters.get_value< unsigned int >("general_queue_size");

  logmessage("\n##\n# Parameters:\n##", 0);
  if (MPI_rank == 0) {
    parameters.print_contents(std::cerr, true);
  }
  logmessage("##\n", 0);
}

/**
 * @brief Set the number of threads to use during the simulation.
 *
 * We first determine the number of threads available (either by system default,
 * or because the user has set the OMP_NUM_THREADS environment variable). We
 * then check if a number of threads was specified on the command line. We don't
 * allow setting the number of threads to a value larger than available, and use
 * the available number as default if no value was given on the command line. If
 * the requested number of threads is larger than what is available, we display
 * a message.
 *
 * @param num_threads_request Requested number of threads.
 * @param num_threads Variable to store the actual number of threads that will
 * be used in.
 */
inline void set_number_of_threads(int num_threads_request, int &num_threads) {

  // check how many threads are available
  int num_threads_available;
#pragma omp parallel
  {
#pragma omp single
    num_threads_available = omp_get_num_threads();
  }

  // now check if this is compatible with what was requested
  if (num_threads_request > num_threads_available) {
    // NO: warn the user
    logmessage("More threads requested ("
                   << num_threads_request << ") than available ("
                   << num_threads_available
                   << "). Resetting to maximum available number of threads.",
               0);
    num_threads_request = num_threads_available;
  }

  // set the number of threads to the requested/maximal allowed value
  omp_set_num_threads(num_threads_request);
  num_threads = num_threads_request;

  logmessage("Running with " << num_threads << " threads.", 0);
}

/**
 * @brief Execute a source photon task.
 *
 * @param task Task to execute.
 * @param thread_id Thread that will execute the task.
 * @param num_photon_sourced Number of photon packets that has already been
 * generated at the source.
 * @param num_photon_local Total number of photon packets that needs to be
 * generated at the source on this process.
 * @param tasks Task space to create new tasks in.
 * @param new_queues Task queues for individual threads.
 * @param new_buffers PhotonBuffer memory space.
 * @param random_generator Random_generator for the active thread.
 * @param central_index List of subgrids that contain the source position.
 * @param gridvec List of all subgrids.
 * @param central_queue List with the corresponding thread number for each entry
 * in central_index.
 * @param num_active_buffers Number of active photon buffers on this process.
 */
inline void execute_source_photon_task(
    Task &task, const int thread_id, unsigned int &num_photon_sourced,
    const unsigned int num_photon_local, TaskSpace &tasks,
    std::vector< NewQueue * > &new_queues, MemorySpace &new_buffers,
    RandomGenerator &random_generator,
    const std::vector< unsigned int > &central_index,
    std::vector< DensitySubGrid * > &gridvec,
    const std::vector< int > &central_queue, unsigned int &num_active_buffers) {

  // log the start time of the task (if task output is enabled)
  task.start(thread_id);

  // we will create a new buffer
  atomic_pre_increment(num_active_buffers);

  unsigned int num_photon_this_loop = task._buffer;

  // get a free photon buffer in the central queue
  unsigned int buffer_index = new_buffers.get_free_buffer();
  PhotonBuffer &input_buffer = new_buffers[buffer_index];
  // assign the buffer to a random thread that has a copy of the
  // subgrid that contains the source position. This should ensure
  // a balanced load for these threads.
  unsigned int which_central_index =
      random_generator.get_uniform_random_double() * central_index.size();
  myassert(which_central_index >= 0 &&
               which_central_index < central_index.size(),
           "Invalid source subgrid thread index!");
  unsigned int this_central_index = central_index[which_central_index];

  // now actually fill the buffer with random photon packets
  fill_buffer(input_buffer, num_photon_this_loop, random_generator,
              this_central_index);

  // add to the queue of the corresponding thread
  const size_t task_index = tasks.get_free_task(28);
  tasks[task_index]._type = TASKTYPE_PHOTON_TRAVERSAL;
  tasks[task_index]._cell = this_central_index;
  tasks[task_index]._buffer = buffer_index;

  // note that this statement should be last, as the buffer might
  // be processed as soon as this statement is executed
  new_queues[central_queue[which_central_index]]->add_task(task_index);

  // log the end time of the task
  task.stop();
}

/**
 * @brief Execute a photon packet traversal task.
 *
 * @param task Task to execute.
 * @param thread_id Thread that will execute the task.
 * @param tasks Task space to create new tasks in.
 * @param new_queues Task queues for individual threads.
 * @param general_queue General queue shared by all threads.
 * @param new_buffers PhotonBuffer memory space.
 * @param gridvec List of all subgrids.
 * @param local_buffers Temporary buffers used in photon packet propagation
 * tasks on this thread.
 * @param local_buffer_flags Temporary flags used in photon packet propagation
 * tasks on this thread.
 * @param reemission_probability Reemission probability.
 * @param costs CostVector that contains information about the load balancing
 * and domain decomposition.
 * @param num_photon_done Number of photon packets that has been completely
 * finished.
 * @param num_empty Number of empty buffers on this process.
 * @param num_active_buffers Number of active photon buffers on this process.
 */
inline void execute_photon_traversal_task(
    Task &task, const int thread_id, TaskSpace &tasks,
    std::vector< NewQueue * > &new_queues, NewQueue &general_queue,
    MemorySpace &new_buffers, std::vector< DensitySubGrid * > &gridvec,
    PhotonBuffer *local_buffers, bool *local_buffer_flags,
    const double reemission_probability, CostVector &costs,
    unsigned int &num_photon_done, unsigned int &num_empty,
    unsigned int &num_active_buffers) {

  // variables used to determine the cost of photon traversal tasks
  unsigned long task_start, task_end;
  task_tick(task_start);

  // log the start of the task
  task.start(thread_id);

  const unsigned int current_buffer_index = task._buffer;
  PhotonBuffer &buffer = new_buffers[current_buffer_index];
  const unsigned int igrid = buffer._sub_grid_index;
  DensitySubGrid &this_grid = *gridvec[buffer._sub_grid_index];

  myassert(costs.get_process(igrid) == MPI_rank,
           "This process should not be working on this subgrid!");

  // prepare output buffers: make sure they are empty and that buffers
  // corresponding to directions outside the simulation box are
  // disabled
  for (int i = 0; i < 27; ++i) {
    const unsigned int ngb = this_grid.get_neighbour(i);
    if (ngb != NEIGHBOUR_OUTSIDE) {
      local_buffer_flags[i] = true;
      local_buffers[i]._actual_size = 0;
    } else {
      local_buffer_flags[i] = false;
    }
  }

  // if reemission is disabled, disable output to the internal buffer
  if (reemission_probability == 0.) {
    local_buffer_flags[TRAVELDIRECTION_INSIDE] = false;
  }

  // keep track of the original number of photons
  unsigned int num_photon_done_now = buffer._actual_size;

  // add to the photon cost of this subgrid (we need to do this now, as we will
  // be subtracting non-finished packets from num_photon_done_now below)
  costs.add_photon_cost(igrid, num_photon_done_now);

  // now do the actual photon traversal
  do_photon_traversal(buffer, this_grid, local_buffers, local_buffer_flags);

  // add none empty buffers to the appropriate queues
  // we go backwards, so that the local queue is added to the task
  // list last (we want to potentially feed hungry threads before we
  // feed ourselves)
  for (int i = 26; i >= 0; --i) {

    // only process enabled, non-empty output buffers
    if (local_buffer_flags[i] && local_buffers[i]._actual_size > 0) {

      // photon packets that are still present in an output buffer
      // are not done yet
      num_photon_done_now -= local_buffers[i]._actual_size;

      // move photon packets from the local temporary buffer (that is
      // guaranteed to be large enough) to the actual output buffer
      // for that direction (which might cause on overflow)
      const unsigned int ngb = this_grid.get_neighbour(i);
      unsigned int new_index = this_grid.get_active_buffer(i);
      if (new_buffers[new_index]._actual_size == 0) {
        // we are adding photons to an empty buffer
        atomic_pre_decrement(num_empty);
      }
      unsigned int add_index =
          new_buffers.add_photons(new_index, local_buffers[i]);

      // check if the original buffer is full
      if (add_index != new_index) {

        // a new active buffer was created
        atomic_pre_increment(num_active_buffers);

        // YES: create a task for the buffer and add it to the queue
        const size_t task_index = tasks.get_free_task(28);
        tasks[task_index]._cell = new_buffers[new_index]._sub_grid_index;
        tasks[task_index]._buffer = new_index;
        // the task type depends on the buffer: photon packets in the
        // internal buffer were absorbed and could be reemitted,
        // photon packets in the other buffers left the subgrid and
        // need to be traversed in the neighbouring subgrid
        if (i > 0) {
          if (costs.get_process(ngb) != MPI_rank) {
            tasks[task_index]._type = TASKTYPE_SEND;
            // add the task to the general queue
            general_queue.add_task(task_index);
          } else {
            tasks[task_index]._type = TASKTYPE_PHOTON_TRAVERSAL;
            // add the task to the queue of the corresponding thread
            const unsigned int queue_index = costs.get_thread(ngb);
            new_queues[queue_index]->add_task(task_index);
          }
        } else {
          tasks[task_index]._type = TASKTYPE_PHOTON_REEMIT;
          // add the task to the general queue
          general_queue.add_task(task_index);
        }

        myassert(new_buffers[add_index]._sub_grid_index == ngb,
                 "Wrong subgrid");
        myassert(new_buffers[add_index]._direction ==
                     output_to_input_direction(i),
                 "Wrong direction");
        // new_buffers.add_photons already created a new empty
        // buffer, set it as the active buffer for this output
        // direction
        this_grid.set_active_buffer(i, add_index);
        if (new_buffers[add_index]._actual_size == 0) {
          // we have created a new empty buffer
          atomic_pre_increment(num_empty);
        }

      } // if (add_index != new_index)

    } // if (local_buffer_flags[i] &&
      //     local_buffers[i]._actual_size > 0)

  } // for (int i = 26; i >= 0; --i)

  // add photons that were absorbed (if reemission was disabled) or
  // that left the system to the global count
  atomic_pre_add(num_photon_done, num_photon_done_now);

  // delete the original buffer, as we are done with it
  new_buffers.free_buffer(current_buffer_index);

  myassert(num_active_buffers > 0, "Number of active buffers < 0!");
  atomic_pre_decrement(num_active_buffers);

  // log the end time of the task
  task.stop();

  // update the cost computation for this subgrid
  task_tick(task_end);
  costs.add_computational_cost(igrid, task_end - task_start);
}

/**
 * @brief Execute a photon packet reemission task.
 *
 * @param task Task to execute.
 * @param thread_id Thread that will execute the task.
 * @param tasks Task space to create new tasks in.
 * @param new_queues Task queues for individual threads.
 * @param new_buffers PhotonBuffer memory space.
 * @param random_generator Random_generator for the active thread.
 * @param reemission_probability Reemission probability.
 * @param costs CostVector that contains information about the load balancing
 * and domain decomposition.
 * @param num_photon_done Number of photon packets that has been completely
 * finished.
 */
inline void execute_photon_reemit_task(
    Task &task, const int thread_id, TaskSpace &tasks,
    std::vector< NewQueue * > &new_queues, MemorySpace &new_buffers,
    RandomGenerator &random_generator, const double reemission_probability,
    CostVector &costs, unsigned int &num_photon_done) {

  // log the start of the task
  task.start(thread_id);

  // get the buffer
  const unsigned int current_buffer_index = task._buffer;
  PhotonBuffer &buffer = new_buffers[current_buffer_index];

  // keep track of the original number of photons in the buffer
  unsigned int num_photon_done_now = buffer._actual_size;

  // reemit photon packets
  do_reemission(buffer, random_generator, reemission_probability);

  // find the number of photon packets that was absorbed and not
  // reemitted...
  num_photon_done_now -= buffer._actual_size;
  // ...and add it to the global count
  atomic_pre_add(num_photon_done, num_photon_done_now);

  // the reemitted photon packets are ready to be propagated: create
  // a new propagation task
  const size_t task_index = tasks.get_free_task(28);
  tasks[task_index]._type = TASKTYPE_PHOTON_TRAVERSAL;
  tasks[task_index]._cell = task._cell;
  tasks[task_index]._buffer = current_buffer_index;
  // add it to the queue of the corresponding thread
  new_queues[costs.get_thread(buffer._sub_grid_index)]->add_task(task_index);

  // log the end time of the task
  task.stop();
}

/**
 * @brief Execute a send task.
 *
 * @param task Task to execute.
 * @param thread_id Thread that will execute the task.
 * @param new_buffers PhotonBuffer memory space.
 * @param costs CostVector that contains information about the load balancing
 * and domain decomposition.
 * @param MPI_lock Lock to guarantee thread-safe access to the MPI library.
 * @param MPI_last_request Index of the last MPI request that was used.
 * @param MPI_buffer_requests List of MPI requests that can be used.
 * @param MPI_buffer MPI communication buffer.
 * @param message_log MPI message log.
 * @param message_log_size Size of the MPI message log.
 * @param num_active_buffers Number of active photon buffers on this process.
 */
inline void execute_send_task(Task &task, const int thread_id,
                              MemorySpace &new_buffers, CostVector &costs,
                              Lock &MPI_lock, unsigned int &MPI_last_request,
                              std::vector< MPI_Request > &MPI_buffer_requests,
                              char *MPI_buffer,
                              std::vector< MPIMessage > &message_log,
                              size_t &message_log_size,
                              unsigned int &num_active_buffers) {

  // log the start of the task
  task.start(thread_id);

  // get the buffer
  const unsigned int current_buffer_index = task._buffer;
  PhotonBuffer &buffer = new_buffers[current_buffer_index];

  // pack it
  // first: get a free MPI_Request
  MPI_lock.lock();
  unsigned int request_index = MPI_last_request;
  while (MPI_buffer_requests[request_index] != MPI_REQUEST_NULL) {
    request_index = (request_index + 1) % MPI_buffer_requests.size();
    myassert(request_index != MPI_last_request,
             "Unable to obtain a free MPI request!");
  }
  MPI_last_request = request_index;
  MPI_Request &request = MPI_buffer_requests[request_index];
  // now use the request index to find the right spot in the buffer
  buffer.pack(&MPI_buffer[request_index * PHOTONBUFFER_MPI_SIZE]);

  // send the message (non-blocking)
  const int sendto = costs.get_process(buffer._sub_grid_index);
  MPI_Isend(&MPI_buffer[request_index * PHOTONBUFFER_MPI_SIZE],
            PHOTONBUFFER_MPI_SIZE, MPI_PACKED, sendto,
            MPIMESSAGETAG_PHOTONBUFFER, MPI_COMM_WORLD, &request);

  // log the send event
  log_send(message_log[message_log_size], sendto, thread_id,
           MPIMESSAGETAG_PHOTONBUFFER);
  ++message_log_size;
  myassert(message_log_size < message_log.size(),
           "Too many messages for message log!");

  MPI_lock.unlock();

  // remove the buffer from this process (the data are stored in the MPI buffer)
  new_buffers.free_buffer(current_buffer_index);

  myassert(num_active_buffers > 0, "Number of active buffers < 0!");
  atomic_pre_decrement(num_active_buffers);

  // log the end time of the task
  task.stop();
}

/**
 * @brief Execute a single task.
 *
 * @param task Task to execute.
 * @param thread_id Thread that will execute the task.
 * @param num_photon_sourced Number of photon packets that has already been
 * generated at the source.
 * @param num_photon_local Total number of photon packets that needs to be
 * generated at the source on this process.
 * @param tasks Task space to create new tasks in.
 * @param new_queues Task queues for individual threads.
 * @param general_queue General queue shared by all threads.
 * @param new_buffers PhotonBuffer memory space.
 * @param random_generator Random_generator for the active thread.
 * @param central_index List of subgrids that contain the source position.
 * @param gridvec List of all subgrids.
 * @param central_queue List with the corresponding thread number for each entry
 * in central_index.
 * @param local_buffers Temporary buffers used in photon packet propagation
 * tasks on this thread.
 * @param local_buffer_flags Temporary flags used in photon packet propagation
 * tasks on this thread.
 * @param reemission_probability Reemission probability.
 * @param costs CostVector that contains information about the load balancing
 * and domain decomposition.
 * @param num_photon_done Number of photon packets that has been completely
 * finished.
 * @param MPI_lock Lock to guarantee thread-safe access to the MPI library.
 * @param MPI_last_request Index of the last MPI request that was used.
 * @param MPI_buffer_requests List of MPI requests that can be used.
 * @param MPI_buffer MPI communication buffer.
 * @param message_log MPI message log.
 * @param message_log_size Size of the MPI message log.
 * @param num_empty Number of empty buffers on this process.
 * @param num_active_buffers Number of active photon buffers on this process.
 */
inline void
execute_task(Task &task, const int thread_id, unsigned int &num_photon_sourced,
             const unsigned int num_photon_local, TaskSpace &tasks,
             std::vector< NewQueue * > &new_queues, NewQueue &general_queue,
             MemorySpace &new_buffers, RandomGenerator &random_generator,
             const std::vector< unsigned int > &central_index,
             std::vector< DensitySubGrid * > &gridvec,
             const std::vector< int > &central_queue,
             PhotonBuffer *local_buffers, bool *local_buffer_flags,
             const double reemission_probability, CostVector &costs,
             unsigned int &num_photon_done, Lock &MPI_lock,
             unsigned int &MPI_last_request,
             std::vector< MPI_Request > &MPI_buffer_requests, char *MPI_buffer,
             std::vector< MPIMessage > &message_log, size_t &message_log_size,
             unsigned int &num_empty, unsigned int &num_active_buffers) {

  myassert(task._end_time == 0, "Task already executed!");

  // Different tasks are processed in different ways.
  switch (task._type) {
  case TASKTYPE_SOURCE_PHOTON:

    /// generate random photon packets from the source

    execute_source_photon_task(task, thread_id, num_photon_sourced,
                               num_photon_local, tasks, new_queues, new_buffers,
                               random_generator, central_index, gridvec,
                               central_queue, num_active_buffers);
    break;

  case TASKTYPE_PHOTON_TRAVERSAL:

    /// propagate photon packets from a buffer through a subgrid

    execute_photon_traversal_task(
        task, thread_id, tasks, new_queues, general_queue, new_buffers, gridvec,
        local_buffers, local_buffer_flags, reemission_probability, costs,
        num_photon_done, num_empty, num_active_buffers);
    break;

  case TASKTYPE_PHOTON_REEMIT:

    /// reemit absorbed photon packets

    execute_photon_reemit_task(task, thread_id, tasks, new_queues, new_buffers,
                               random_generator, reemission_probability, costs,
                               num_photon_done);
    break;

  case TASKTYPE_SEND:

    /// send a buffer to another process

    execute_send_task(task, thread_id, new_buffers, costs, MPI_lock,
                      MPI_last_request, MPI_buffer_requests, MPI_buffer,
                      message_log, message_log_size, num_active_buffers);
    break;

  default:

    // should never happen
    cmac_error("Unknown task: %i!", task._type);
  }
}

/**
 * @brief Prematurely activate a buffer to feed hungry threads.
 *
 * @param current_index Index of a task. Should be NO_TASK upon entry. Upon
 * exit, this variable will contain the index of a task that can be executed by
 * the current thread.
 * @param thread_id Thread that executes this code.
 * @param tasks Task space.
 * @param new_queues Per thread task queues.
 * @param general_queue General queue that is shared by all threads.
 * @param new_buffers PhotonBuffer memory space.
 * @param gridvec List of subgrids.
 * @param costs CostVector used for load balancing and domain decomposition.
 * @param num_empty Number of empty buffers on this process.
 * @param num_active_buffers Number of active photon buffers on this process.
 */
inline void activate_buffer(unsigned int &current_index, const int thread_id,
                            TaskSpace &tasks,
                            std::vector< NewQueue * > &new_queues,
                            NewQueue &general_queue, MemorySpace &new_buffers,
                            std::vector< DensitySubGrid * > &gridvec,
                            CostVector &costs, unsigned int &num_empty,
                            unsigned int &num_active_buffers) {
  // try to activate a non-full buffer
  // note that we only try to access thread-local information, so as
  // long as we don't allow task stealing, this will be thread-safe
  unsigned int i = 0;
  // loop over all subgrids
  while (i < gridvec.size() && current_index == NO_TASK) {
    // we only activate subgrids that belong to this thread to make
    // sure we don't create conflicts
    // Note that this could mean we prematurely activate tasks for
    // another thread (and even another process).
    if (costs.get_process(i) == MPI_rank && costs.get_thread(i) == thread_id) {
      int j = 0;
      // loop over all buffers of this subgrid
      while (j < 27 && current_index == NO_TASK) {
        // only process existing buffers that are non empty
        if (gridvec[i]->get_neighbour(j) != NEIGHBOUR_OUTSIDE &&
            new_buffers[gridvec[i]->get_active_buffer(j)]._actual_size > 0) {
          // found one! Prematurely activate this subgrid.
          const unsigned int non_full_index = gridvec[i]->get_active_buffer(j);
          // Create a new empty buffer and set it as active buffer for
          // this subgrid.
          // Note that this is thread safe, as the only thread that can
          // the old buffer at this moment in time is the same
          // thread that replaces the buffer here.
          // So there is no risk of accidentally replacing the buffer
          // while another thread is doing the same thing.
          const unsigned int new_index = new_buffers.get_free_buffer();
          new_buffers[new_index]._sub_grid_index =
              new_buffers[non_full_index]._sub_grid_index;
          new_buffers[new_index]._direction =
              new_buffers[non_full_index]._direction;
          gridvec[i]->set_active_buffer(j, new_index);
          // we are creating a new active photon buffer
          atomic_pre_increment(num_active_buffers);
          // Add the buffer to the queue of the corresponding thread.
          // Note that this could be another thread than this thread, in
          // which case this thread will still be hungry (and we might
          // be over-feeding the other thread).
          // However, this is the only mechanism through which we can
          // feed hungry threads with food from this thread, as we do
          // not allow threads to feed themselves with food that does
          // not belong to them.
          const size_t task_index = tasks.get_free_task(28);
          tasks[task_index]._cell = new_buffers[non_full_index]._sub_grid_index;
          tasks[task_index]._buffer = non_full_index;
          if (j > 0) {
            if (costs.get_process(
                    new_buffers[non_full_index]._sub_grid_index) != MPI_rank) {
              tasks[task_index]._type = TASKTYPE_SEND;
              general_queue.add_task(task_index);
            } else {
              tasks[task_index]._type = TASKTYPE_PHOTON_TRAVERSAL;
              const unsigned int queue_index =
                  costs.get_thread(new_buffers[non_full_index]._sub_grid_index);
              new_queues[queue_index]->add_task(task_index);
            }
          } else {
            tasks[task_index]._type = TASKTYPE_PHOTON_REEMIT;
            general_queue.add_task(task_index);
          }
          // we created a new empty buffer
          atomic_pre_increment(num_empty);

          // Try again to get a task. This could be the task we just
          // created, a task that was added by another thread while we
          // were doing the task activation, or still NO_TASK, in which
          // case we continue waking up buffers.
          current_index = new_queues[thread_id]->get_task();
          if (current_index == NO_TASK) {
            current_index = general_queue.get_task();
          }
        }
        ++j;
      }
    }
    ++i;
  }
}

/**
 * @brief Check if a non-blocking send finished and release the associated
 * memory if this happened.
 *
 * @param MPI_buffer_requests MPI requests for open non-blocking sends.
 */
inline void
check_for_finished_sends(std::vector< MPI_Request > &MPI_buffer_requests) {
  // check if any of the non-blocking sends finished
  int index, flag;
  MPI_Testany(MPI_buffer_requests.size(), &MPI_buffer_requests[0], &index,
              &flag, MPI_STATUS_IGNORE);
  // we cannot test flag, since flag will also be true if the array only
  // contains MPI_REQUEST_NULL values
  if (index != MPI_UNDEFINED) {
    // release the request, this will automatically release the
    // corresponding space in the buffer
    MPI_buffer_requests[index] = MPI_REQUEST_NULL;
  }
}

/**
 * @brief Check for incoming MPI communications and process them.
 *
 * @param message_log MPI message log.
 * @param message_log_size Size of the MPI message log.
 * @param new_buffers PhotonBuffer memory space.
 * @param costs CostVector responsible for load-balancing and domain
 * decomposition.
 * @param tasks Task space.
 * @param new_queues Per thread queues.
 * @param num_photon_done_since_last Number of photon packets that finished
 * completely since the last time the process thought it was done.
 * @param global_run_flag Global flag that controls when we should finish.
 * @param thread_id Active thread.
 * @param num_active_buffers Number of active photon buffers on this process.
 */
inline void check_for_incoming_communications(
    std::vector< MPIMessage > &message_log, size_t &message_log_size,
    MemorySpace &new_buffers, CostVector &costs, TaskSpace &tasks,
    std::vector< NewQueue * > &new_queues,
    unsigned int &num_photon_done_since_last, bool &global_run_flag,
    const int thread_id, unsigned int &num_active_buffers) {

  // check for incoming communications
  // this is a non-blocking event: we just check if a message is ready to
  // receive
  MPI_Status status;
  int flag;
  MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &flag, &status);
  // flag will be 1 if there is a message ready to receive; status stores the
  // source rank and tag of the message
  if (flag) {
    const int source = status.MPI_SOURCE;
    const int tag = status.MPI_TAG;

    // We can receive a message! How we receive it depends on the tag.
    if (tag == MPIMESSAGETAG_PHOTONBUFFER) {

      // set up a dummy task to show receives in task plots
      size_t task_index = tasks.get_free_task(0);
      Task &receive_task = tasks[task_index];
      receive_task._type = TASKTYPE_RECV;

      receive_task.start(thread_id);

      // incoming photon buffer
      // we need to receive it and schedule a new propagation task

      // receive the message
      char buffer[PHOTONBUFFER_MPI_SIZE];
      MPI_Recv(buffer, PHOTONBUFFER_MPI_SIZE, MPI_PACKED, source, tag,
               MPI_COMM_WORLD, MPI_STATUS_IGNORE);

      // log the receive event
      log_recv(message_log[message_log_size], source, thread_id, tag);
      ++message_log_size;
      myassert(message_log_size < message_log.size(),
               "Too many messages for message log!");

      // get a new free buffer
      unsigned int buffer_index = new_buffers.get_free_buffer();
      PhotonBuffer &input_buffer = new_buffers[buffer_index];

      // we created a new buffer
      atomic_pre_increment(num_active_buffers);

      // fill the buffer
      input_buffer.unpack(buffer);

      unsigned int subgrid_index = input_buffer._sub_grid_index;
      myassert(costs.get_process(subgrid_index) == MPI_rank,
               "Message arrived on wrong process!");
      unsigned int thread_index = costs.get_thread(subgrid_index);

      // add to the queue of the corresponding thread
      task_index = tasks.get_free_task(28);
      tasks[task_index]._type = TASKTYPE_PHOTON_TRAVERSAL;
      tasks[task_index]._cell = subgrid_index;
      tasks[task_index]._buffer = buffer_index;
      // note that this statement should be last, as the buffer might
      // be processed as soon as this statement is executed
      new_queues[thread_index]->add_task(task_index);

      receive_task.stop();

    } else if (tag == MPIMESSAGETAG_LOCAL_PROCESS_FINISHED) {

      // a process finished its local work and sends its tallies to the master
      // process to check if the traversal step finished
      myassert(MPI_rank == 0,
               "Only the master rank should receive this type of message!");

      // receive the tally
      unsigned int tally;
      MPI_Recv(&tally, 1, MPI_UNSIGNED, source, tag, MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);

      // log the receive event
      log_recv(message_log[message_log_size], source, thread_id, tag);
      ++message_log_size;
      myassert(message_log_size < message_log.size(),
               "Too many messages for message log!");

      // add the tally to the current total
      atomic_pre_add(num_photon_done_since_last, tally);

    } else if (tag == MPIMESSAGETAG_STOP) {

      // the master rank detected a real finish: finish the photon traversal
      // step
      myassert(MPI_rank != 0,
               "The master rank should not receive this type of message!");

      // we do not really need to receive this message, as the tag contains all
      // the information we need
      MPI_Recv(nullptr, 0, MPI_INT, source, tag, MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);

      // log the receive event
      log_recv(message_log[message_log_size], source, thread_id, tag);
      ++message_log_size;
      myassert(message_log_size < message_log.size(),
               "Too many messages for message log!");

      // set the stop condition, all threads will stop the propagation step
      global_run_flag = false;

    } else {
      cmac_error("Unknown tag: %i!", tag);
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

  // first: start timing
  Timer program_timer;
  program_timer.start();

  //////////////////////////////////////////////////////////////////////////////
  /// Initialization
  //////////////////////////////////////////////////////////////////////////////

  //////////////////////
  // MPI initialization
  /////////////////////

  initialize_MPI(argc, argv, MPI_rank, MPI_size);

  // disable std::cout output for all processes other than rank 0 (log messages
  // by default are only written by rank 0)
  if (MPI_rank > 0) {
    std::cout.rdbuf(nullptr);
  }

  /////////////////////

  //////////////////////////////////
  // Parse the command line options
  /////////////////////////////////

  int num_threads_request;
  std::string paramfile_name;
  parse_command_line(argc, argv, num_threads_request, paramfile_name);

  /////////////////////////////////

  ///////////////////////////////////////
  // Set up the number of threads to use
  //////////////////////////////////////

  int num_threads;
  set_number_of_threads(num_threads_request, num_threads);

  //////////////////////////////////////

  ///////////////////////////
  // Read the parameter file
  //////////////////////////

  double box[6], reemission_probability;
  int ncell[3], num_subgrid[3];
  unsigned int num_photon, number_of_iterations;

  unsigned int queue_size_per_thread, memoryspace_size, number_of_tasks,
      MPI_buffer_size, general_queue_size;
  double copy_factor;

  read_parameters(paramfile_name, box, reemission_probability, ncell,
                  num_subgrid, num_photon, number_of_iterations,
                  queue_size_per_thread, memoryspace_size, number_of_tasks,
                  MPI_buffer_size, copy_factor, general_queue_size);

  //////////////////////////

  ////////////////////////////////
  // Set up task based structures
  ///////////////////////////////

  // set up the queues used to queue tasks
  std::vector< NewQueue * > new_queues(num_threads, nullptr);
  for (int i = 0; i < num_threads; ++i) {
    new_queues[i] = new NewQueue(queue_size_per_thread);
  }

  NewQueue general_queue(general_queue_size);

  // set up the task space used to store tasks
  TaskSpace tasks(number_of_tasks, 28 * number_of_tasks);

  // set up the memory space used to store photon packet buffers
  MemorySpace new_buffers(memoryspace_size);

  // set up the cost vector used to load balance
  const unsigned int tot_num_subgrid =
      num_subgrid[0] * num_subgrid[1] * num_subgrid[2];
  CostVector costs(tot_num_subgrid, num_threads, MPI_size);

  ///////////////////////////////

  ///////////////////////////////////
  // Set up MPI communication buffer
  //////////////////////////////////

  char *MPI_buffer = nullptr;
  // we use a single, locked request buffer to handle memory management for
  // non-blocking communication.
  // We do this because any communication requires locking anyway, and because
  // it gives us a more elegant way to regularly check for finished sends.
  std::vector< MPI_Request > MPI_buffer_requests;
  Lock MPI_lock;
  unsigned int MPI_last_request = 0;
  if (MPI_size > 1) {
    MPI_buffer = new char[MPI_buffer_size];
    const unsigned int num_photonbuffers =
        MPI_buffer_size / PHOTONBUFFER_MPI_SIZE;
    MPI_buffer_requests.resize(num_photonbuffers, MPI_REQUEST_NULL);

    logmessage(
        "MPI_buffer_size: " << Utilities::human_readable_bytes(MPI_buffer_size),
        0);
  }

  // set up the message log buffer
  // we don't need to use a thread safe vector, as only one thread is allowed
  // access to MPI at the same time
  std::vector< MPIMessage > message_log;
  if (MPI_size > 1) {
    message_log.resize(number_of_tasks);
  }
  size_t message_log_size = 0;

  //////////////////////////////////

  ///////////////////////////////////////
  // Set up the random number generators
  //////////////////////////////////////

  std::vector< RandomGenerator > random_generator(num_threads);
  for (int i = 0; i < num_threads; ++i) {
    // make sure every thread on every process has a different seed
    random_generator[i].set_seed(42 + MPI_rank * num_threads + i);
  }

  //////////////////////////////////////

  ///////////////////
  // Set up the grid
  //////////////////

  // set up the grid of smaller grids used for the algorithm
  // each smaller grid stores a fraction of the total grid and has information
  // about the neighbouring subgrids
  std::vector< DensitySubGrid * > gridvec(tot_num_subgrid, nullptr);

  // the actual grid is only constructed on rank 0
  if (MPI_rank == 0) {

    const double subbox_side[3] = {box[3] / num_subgrid[0],
                                   box[4] / num_subgrid[1],
                                   box[5] / num_subgrid[2]};
    const int subbox_ncell[3] = {ncell[0] / num_subgrid[0],
                                 ncell[1] / num_subgrid[1],
                                 ncell[2] / num_subgrid[2]};

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

  } // if(MPI_rank == 0)

  //////////////////

  ////////////////////////////////////////////
  // Initialize the photon source information
  ///////////////////////////////////////////

  // Get the index of the (one) subgrid that contains the source position
  const unsigned int source_indices[3] = {
      (unsigned int)((-box[0] / box[3]) * num_subgrid[0]),
      (unsigned int)((-box[1] / box[4]) * num_subgrid[1]),
      (unsigned int)((-box[2] / box[5]) * num_subgrid[2])};

  const unsigned int central_index =
      source_indices[0] * num_subgrid[1] * num_subgrid[2] +
      source_indices[1] * num_subgrid[2] + source_indices[2];

  ///////////////////////////////////////////

  //////////////////////////////
  // Initialize the cost vector
  /////////////////////////////

  // we need 2 cost vectors:
  //  - an actual computational cost vector used for shared memory balancing
  std::vector< unsigned long > initial_cost_vector(tot_num_subgrid, 0);
  //  - a photon cost vector that is a more stable reference for the distributed
  //    memory decomposition
  std::vector< unsigned int > initial_photon_cost(tot_num_subgrid, 0);
  std::ifstream initial_costs("costs_00.txt");
  if (initial_costs.good()) {
    // use cost information from a previous run as initial guess for the cost
    // skip the initial comment line
    std::string line;
    std::getline(initial_costs, line);
    unsigned int index;
    unsigned long computational_cost;
    unsigned int photon_cost, source_cost;
    int rank, thread;
    while (std::getline(initial_costs, line)) {
      std::istringstream lstream(line);
      lstream >> index >> computational_cost >> photon_cost >> source_cost >>
          rank >> thread;
      if (index < tot_num_subgrid) {
        initial_cost_vector[index] += computational_cost;
        initial_photon_cost[index] += photon_cost;
      }
    }
  } else {
    // no initial cost information: assume a uniform cost
    for (unsigned int i = 0; i < tot_num_subgrid; ++i) {
      initial_cost_vector[i] = 1;
      initial_photon_cost[i] = 1;
    }
  }

  /////////////////////////////

  //////////////////////////////////////////////////////////////////////////////

  //////////////////////////////////////////////////////////////////////////////
  /// Main loop
  //////////////////////////////////////////////////////////////////////////////

  /////////////////////////////////////////////////
  // Make copies and do the initial load balancing
  ////////////////////////////////////////////////

  // make copies of the most expensive subgrids, so that multiple threads can
  // work on them simultaneously
  // this is only done by rank 0
  std::vector< unsigned int > originals;
  std::vector< unsigned int > copies(tot_num_subgrid, 0xffffffff);
  std::vector< unsigned char > levels(tot_num_subgrid, 0);
  if (MPI_rank == 0) {

    // get the average cost per thread
    unsigned long avg_cost_per_thread = 0;
    for (unsigned int i = 0; i < tot_num_subgrid; ++i) {
      avg_cost_per_thread += initial_cost_vector[i];
    }
    avg_cost_per_thread /= (num_threads * MPI_size);
    // now set the levels accordingly
    for (unsigned int i = 0; i < tot_num_subgrid; ++i) {
      if (copy_factor * initial_cost_vector[i] > avg_cost_per_thread) {
        // note that this in principle should be 1 higher. However, we do not
        // count the original.
        unsigned int number_of_copies =
            copy_factor * initial_cost_vector[i] / avg_cost_per_thread;
        // make sure the number of copies of the source subgrid is at least
        // equal to the number of processes
        if (i == central_index &&
            number_of_copies < static_cast< unsigned int >(MPI_size)) {
          number_of_copies = MPI_size;
        }
        // get the highest bit
        unsigned int level = 0;
        while (number_of_copies > 0) {
          number_of_copies >>= 1;
          ++level;
        }
        levels[i] = level;
      }
    }

    // now create the copies
    create_copies(gridvec, levels, new_buffers, originals, copies);
  }

  // communicate the new size of the grid to all processes and make sure the
  // local gridvec is up to date (all processes other than rank 0 still have a
  // completely empty gridvec)
  unsigned int new_size = gridvec.size();
  MPI_Bcast(&new_size, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
  gridvec.resize(new_size, nullptr);

  // communicate the originals and copies to all processes
  unsigned int originals_size = originals.size();
  MPI_Bcast(&originals_size, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
  originals.resize(originals_size, 0);
  MPI_Bcast(&originals[0], originals_size, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
  MPI_Bcast(&copies[0], tot_num_subgrid, MPI_UNSIGNED, 0, MPI_COMM_WORLD);

  // NOTE that the code below should have the same inputs (and hence outputs) on
  // all processes

  // initialize the actual cost vector
  costs.reset(gridvec.size());

  // set the initial cost data (that were loaded before)
  for (unsigned int i = 0; i < tot_num_subgrid; ++i) {
    costs.add_computational_cost(i, initial_cost_vector[i]);
    costs.add_photon_cost(i, initial_photon_cost[i]);
  }

  // add copy cost data: divide original cost by number of copies
  for (unsigned int igrid = 0; igrid < tot_num_subgrid; ++igrid) {
    std::vector< unsigned int > this_copies;
    unsigned int copy = copies[igrid];
    if (copy < 0xffffffff) {
      while (copy < gridvec.size() &&
             originals[copy - tot_num_subgrid] == igrid) {
        this_copies.push_back(copy);
        ++copy;
      }
    }
    if (this_copies.size() > 0) {
      const unsigned long cost =
          costs.get_computational_cost(igrid) / (this_copies.size() + 1);
      const unsigned int photon_cost =
          costs.get_photon_cost(igrid) / (this_copies.size() + 1);
      costs.set_computational_cost(igrid, cost);
      costs.set_photon_cost(igrid, photon_cost);
      for (unsigned int i = 0; i < this_copies.size(); ++i) {
        costs.set_computational_cost(this_copies[i], cost);
        costs.set_photon_cost(this_copies[i], photon_cost);
      }
    }
  }

  // set up the graph for the distributed memory domain decomposition
  // figure out the source costs
  std::vector< std::vector< unsigned int > > ngbs(gridvec.size());
  std::vector< unsigned int > source_cost(gridvec.size(), 0);
  // only rank 0 does this for the moment
  if (MPI_rank == 0) {

    // find the graph of the subgrids (with copies) and set the source costs

    for (size_t igrid = 0; igrid < gridvec.size(); ++igrid) {
      source_cost[igrid] =
          (igrid == central_index) ||
          (igrid >= tot_num_subgrid &&
           originals[igrid - tot_num_subgrid] == central_index);

      DensitySubGrid &subgrid = *gridvec[igrid];
      // only include the face neighbours, as they are the only ones with a
      // significant communication load
      for (int ingb = 21; ingb < 27; ++ingb) {
        unsigned int ngb = subgrid.get_neighbour(ingb);
        if (ngb != NEIGHBOUR_OUTSIDE) {
          // try to find the neighbour in the neighbour list for igrid
          unsigned int index = 0;
          while (index < ngbs[igrid].size() && ngbs[igrid][index] != ngb) {
            ++index;
          }
          if (index == ngbs[igrid].size()) {
            // not found: add it
            ngbs[igrid].push_back(ngb);
          }
          // now do the same for ngb and igrid (to cover the case where they are
          // not mutual neighbours because of copies)
          index = 0;
          while (index < ngbs[ngb].size() && ngbs[ngb][index] != igrid) {
            ++index;
          }
          if (index == ngbs[ngb].size()) {
            // not found: add it
            ngbs[ngb].push_back(igrid);
          }
        }
      }
    }
  }

  // broadcast ngbs and source costs to all processes
  unsigned int ngbsize = ngbs.size();
  std::vector< unsigned int > ngbsizes(ngbsize, 0);
  for (unsigned int i = 0; i < ngbsize; ++i) {
    ngbsizes[i] = ngbs[i].size();
  }
  MPI_Bcast(&ngbsize, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
  ngbsizes.resize(ngbsize, 0);
  MPI_Bcast(&ngbsizes[0], ngbsize, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
  ngbs.resize(ngbsize);
  for (unsigned int i = 0; i < ngbsize; ++i) {
    ngbs[i].resize(ngbsizes[i], 0);
    MPI_Bcast(&ngbs[i][0], ngbsizes[i], MPI_UNSIGNED, 0, MPI_COMM_WORLD);
  }
  unsigned int source_cost_size = source_cost.size();
  MPI_Bcast(&source_cost_size, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
  source_cost.resize(source_cost_size, 0);
  MPI_Bcast(&source_cost[0], source_cost_size, MPI_UNSIGNED, 0, MPI_COMM_WORLD);

  // set source costs
  for (unsigned int igrid = 0; igrid < gridvec.size(); ++igrid) {
    costs.set_source_cost(igrid, source_cost[igrid]);
  }

  // now do the actual domain decomposition
  costs.redistribute(ngbs);

  // now it is time to move the subgrids to the process where they belong
  // rank 0 sends, all other ranks receive
  if (MPI_rank == 0) {
    unsigned int buffer_position = 0;
    for (int irank = 1; irank < MPI_size; ++irank) {
      unsigned int rank_size = 0;
      for (unsigned int igrid = 0; igrid < gridvec.size(); ++igrid) {
        if (costs.get_process(igrid) == irank) {
          int position = 0;
          myassert(buffer_position + rank_size <= MPI_buffer_size,
                   "MPI buffer overflow!");
          // length 4 since buffer contains 8-bit chars and we are adding a
          // 32-bit integer
          MPI_Pack(&igrid, 1, MPI_UNSIGNED,
                   &MPI_buffer[buffer_position + rank_size], 4, &position,
                   MPI_COMM_WORLD);
          rank_size += 4;
          myassert(buffer_position + rank_size <= MPI_buffer_size,
                   "MPI buffer overflow!");
          gridvec[igrid]->pack(&MPI_buffer[buffer_position + rank_size],
                               MPI_buffer_size - buffer_position - rank_size);
          rank_size += gridvec[igrid]->get_MPI_size();

          // delete the original subgrid
          delete gridvec[igrid];
          gridvec[igrid] = nullptr;
        }
      }
      // we could also just use MPI_Probe to figure out the size of the incoming
      // message
      MPI_Send(&rank_size, 1, MPI_UNSIGNED, irank, 0, MPI_COMM_WORLD);
      MPI_Send(&MPI_buffer[buffer_position], rank_size, MPI_PACKED, irank, 1,
               MPI_COMM_WORLD);
      buffer_position += rank_size;
    }
  } else {
    const int ncell_dummy[3] = {1, 1, 1};
    unsigned int rank_size;
    MPI_Recv(&rank_size, 1, MPI_UNSIGNED, 0, 0, MPI_COMM_WORLD,
             MPI_STATUS_IGNORE);
    MPI_Recv(MPI_buffer, rank_size, MPI_PACKED, 0, 1, MPI_COMM_WORLD,
             MPI_STATUS_IGNORE);
    unsigned int buffer_position = 0;
    while (rank_size > 0) {
      unsigned int igrid;
      int position = 0;
      myassert(buffer_position <= MPI_buffer_size, "MPI buffer overflow!");
      MPI_Unpack(&MPI_buffer[buffer_position], 4, &position, &igrid, 1,
                 MPI_UNSIGNED, MPI_COMM_WORLD);
      buffer_position += 4;
      rank_size -= 4;
      gridvec[igrid] = new DensitySubGrid(box, ncell_dummy);
      myassert(buffer_position <= MPI_buffer_size, "MPI buffer overflow!");
      gridvec[igrid]->unpack(&MPI_buffer[buffer_position],
                             MPI_buffer_size - buffer_position);
      buffer_position += gridvec[igrid]->get_MPI_size();
      rank_size -= gridvec[igrid]->get_MPI_size();
    }
  }

  // Just for now: wait until the communication is finished before proceeding.
  MPI_Barrier(MPI_COMM_WORLD);

  // set up the initial photon buffers for each subgrid
  for (unsigned int igrid = 0; igrid < gridvec.size(); ++igrid) {
    if (costs.get_process(igrid) == MPI_rank) {
      DensitySubGrid &this_grid = *gridvec[igrid];
      for (int i = 0; i < 27; ++i) {
        const unsigned int ngb_index = this_grid.get_neighbour(i);
        if (ngb_index != NEIGHBOUR_OUTSIDE) {
          const unsigned int active_buffer = new_buffers.get_free_buffer();
          PhotonBuffer &buffer = new_buffers[active_buffer];
          buffer._sub_grid_index = ngb_index;
          buffer._direction = output_to_input_direction(i);
          this_grid.set_active_buffer(i, active_buffer);
        }
      }
    }
  }

  ////////////////////////////////////////////////

  ////////////////////
  // Actual main loop
  ///////////////////

  // now for the main loop. This loop
  //  - shoots num_photon photons through the grid to get intensity estimates
  //  - computes the ionization equilibrium
  for (unsigned int iloop = 0; iloop < number_of_iterations; ++iloop) {

    // make a global list of all subgrids (on this process) that contain the
    // (single) photon source position, we will distribute the initial
    // propagation tasks evenly across these subgrids
    std::vector< unsigned int > central_index;
    central_index.push_back(
        source_indices[0] * num_subgrid[1] * num_subgrid[2] +
        source_indices[1] * num_subgrid[2] + source_indices[2]);
    unsigned int copy = copies[central_index[0]];
    if (copy < 0xffffffff) {
      while (copy < gridvec.size() &&
             originals[copy - tot_num_subgrid] == central_index[0]) {
        central_index.push_back(copy);
        ++copy;
      }
    }
    logmessage("Number of central subgrid copies: " << central_index.size(), 0);
    const unsigned int tot_num_copies = central_index.size();

    // remove all central indices that are not on this process
    central_index.erase(
        std::remove_if(central_index.begin(), central_index.end(),
                       [&costs](unsigned int index) {
                         return (costs.get_process(index) != MPI_rank);
                       }),
        central_index.end());

    for (unsigned int i = 0; i < central_index.size(); ++i) {
      myassert(costs.get_process(central_index[i]) == MPI_rank,
               "Wrong subgrid index in list!");
    }

    // make sure all processes have source photons (we do not enforce this
    // anywhere, so this assertion might very well fail)
    myassert(central_index.size() > 0, "Rank without source photons!");

    // divide the total number of photons over the processes, weighted with
    // the number of copies each process holds
    const unsigned int num_photon_per_copy = num_photon / tot_num_copies;
    // for now just assume the total number of copies is a divisor of the total
    // number of photons, and crash if it isn't
    myassert(num_photon_per_copy * tot_num_copies == num_photon,
             "Photon number not conserved!");
    const unsigned int num_photon_local =
        num_photon_per_copy * central_index.size();

    // get the corresponding thread ranks
    std::vector< int > central_queue(central_index.size());
    for (unsigned int i = 0; i < central_index.size(); ++i) {
      central_queue[i] = costs.get_thread(central_index[i]);
      // set the source cost
      costs.set_source_cost(central_index[i], 1);
    }

    // STEP 0: log output
    logmessage("Loop " << iloop + 1, 0);

    // STEP 1: photon shooting
    logmessage("Starting photon shoot loop", 0);
    // GLOBAL control variables (these are shared and updated atomically):
    //  - number of photon packets that has been created at the source
    unsigned int num_photon_sourced = 0;
    //  - number of photon packets that has left the system, either through
    //    absorption or by crossing a simulation box wall
    unsigned int num_photon_done = 0;
    bool global_run_flag = true;
    // local control variables
    const unsigned int num_empty_target = 27 * gridvec.size();
    unsigned int num_empty = 27 * gridvec.size();
    unsigned int num_active_buffers = 0;
    // global control variable
    unsigned int num_photon_done_since_last = 0;
#pragma omp parallel default(shared)
    {
      // id of this specific thread
      const int thread_id = omp_get_thread_num();
      PhotonBuffer local_buffers[27];
      bool local_buffer_flags[27];
      for (int i = 0; i < 27; ++i) {
        local_buffers[i]._direction = output_to_input_direction(i);
        local_buffers[i]._actual_size = 0;
        local_buffer_flags[i] = true;
      }

      // add source photon tasks to the general queue
      while (num_photon_sourced < num_photon_local) {
        const unsigned int num_photon_sourced_now =
            atomic_post_add(num_photon_sourced, PHOTONBUFFER_SIZE);
        if (num_photon_sourced_now < num_photon_local) {
          unsigned int num_photon_this_loop = PHOTONBUFFER_SIZE;
          if (num_photon_sourced_now + PHOTONBUFFER_SIZE > num_photon_local) {
            num_photon_this_loop += (num_photon_local - num_photon_sourced_now);
          }

          // create task
          const size_t task_index = tasks.get_free_task(0);
          tasks[task_index]._type = TASKTYPE_SOURCE_PHOTON;
          // store the number of photons in the _buffer field, which is not used
          // at the moment
          tasks[task_index]._buffer = num_photon_this_loop;
          general_queue.add_task(task_index);
        }
      }

      // this loop is repeated until all local photons have been propagated
      // note that this condition automatically covers the condition
      //  num_photon_sourced < num_photon_local
      // as unsourced photons cannot contribute to num_photon_done
      // this condition definitely needs to change for MPI...
      while (global_run_flag) {

        // first do MPI related stuff
        // we only allow one thread at a time to use the MPI library
        if (MPI_size > 1 && MPI_lock.try_lock()) {

          check_for_finished_sends(MPI_buffer_requests);

          check_for_incoming_communications(
              message_log, message_log_size, new_buffers, costs, tasks,
              new_queues, num_photon_done_since_last, global_run_flag,
              thread_id, num_active_buffers);

          MPI_lock.unlock();
        }

        // get a first task
        // upon first entry of the while loop, this will be one of the photon
        // source tasks we just created
        unsigned int current_index = new_queues[thread_id]->get_task();
        if (current_index == NO_TASK) {
          current_index = general_queue.get_task();
        }

        // task activation: if no task is found, try to launch a photon buffer
        // that is not yet full and prematurely schedule it
        if (current_index == NO_TASK) {
          activate_buffer(current_index, thread_id, tasks, new_queues,
                          general_queue, new_buffers, gridvec, costs, num_empty,
                          num_active_buffers);
        }

        // Keep processing tasks until the queue is empty.
        while (current_index != NO_TASK) {

          Task &task = tasks[current_index];
          execute_task(task, thread_id, num_photon_sourced, num_photon_local,
                       tasks, new_queues, general_queue, new_buffers,
                       random_generator[thread_id], central_index, gridvec,
                       central_queue, local_buffers, local_buffer_flags,
                       reemission_probability, costs,
                       num_photon_done_since_last, MPI_lock, MPI_last_request,
                       MPI_buffer_requests, MPI_buffer, message_log,
                       message_log_size, num_empty, num_active_buffers);

          // this would be the right place to delete the task (if we don't want
          // to output it)

          // now do MPI related stuff
          // we only allow one thread at a time to use the MPI library
          if (MPI_size > 1 && MPI_lock.try_lock()) {

            check_for_finished_sends(MPI_buffer_requests);

            check_for_incoming_communications(
                message_log, message_log_size, new_buffers, costs, tasks,
                new_queues, num_photon_done_since_last, global_run_flag,
                thread_id, num_active_buffers);

            MPI_lock.unlock();
          }

          // We finished a task: try to get a new task from the local queue
          current_index = new_queues[thread_id]->get_task();
          if (current_index == NO_TASK) {
            current_index = general_queue.get_task();
          }

        } // while (current_index != NO_TASK)

        // check if the local process finished
        if (num_empty == num_empty_target && num_active_buffers == 0) {

          logmessage_lockfree("thread " << MPI_rank << "." << thread_id
                                        << " thinks we're ready!",
                              1);
          // we use the MPI lock so that we know for sure we are the only
          // thread that has access to num_photon_done_since_last:
          // other threads cannot do anything as long as there is no incoming
          // communication, and an incoming communication can only be received
          // by a thread that holds the MPI lock
          if (MPI_lock.try_lock()) {
            if (num_empty == num_empty_target && num_active_buffers == 0 &&
                num_photon_done_since_last > 0) {
              if (MPI_rank == 0) {
                num_photon_done += num_photon_done_since_last;
                num_photon_done_since_last = 0;
                logmessage_lockfree("num_photon_done = " << num_photon_done
                                                         << " (" << num_photon
                                                         << ")",
                                    1);
                if (num_photon_done == num_photon) {
                  // send stop signal to all other processes
                  for (int irank = 1; irank < MPI_size; ++irank) {
                    MPI_Request request;
                    // we don't need to actually send anything, just sending
                    // the tag is enough
                    MPI_Isend(nullptr, 0, MPI_INT, irank, MPIMESSAGETAG_STOP,
                              MPI_COMM_WORLD, &request);

                    log_send(message_log[message_log_size], irank, thread_id,
                             MPIMESSAGETAG_STOP);
                    ++message_log_size;
                    myassert(message_log_size < message_log.size(),
                             "Too many messages for message log!");

                    MPI_Request_free(&request);
                  }
                  // make sure the master process also stops
                  global_run_flag = false;
                }
              } else {
                // send tally to master rank
                MPI_Request request;
                MPI_Isend(&num_photon_done_since_last, 1, MPI_UNSIGNED, 0,
                          MPIMESSAGETAG_LOCAL_PROCESS_FINISHED, MPI_COMM_WORLD,
                          &request);

                log_send(message_log[message_log_size], 0, thread_id,
                         MPIMESSAGETAG_LOCAL_PROCESS_FINISHED);
                ++message_log_size;
                myassert(message_log_size < message_log.size(),
                         "Too many messages for message log!");

                // https://www.open-mpi.org/doc/v2.0/man3/MPI_Request_free.3.php
                //  MPI_Request_free marks the request object for deallocation
                //  and sets request to MPI_REQUEST_NULL. Any ongoing
                //  communication that is associated with the request will be
                //  allowed to complete. The request will be deallocated only
                //  after its completion.
                // in other words: we can safely throw away the request
                MPI_Request_free(&request);

                num_photon_done_since_last = 0;
              }
            }
            MPI_lock.unlock();
          }
        }

      } // while (global_run_flag)

      logmessage("Thread " << thread_id << " exited loop!", 0);

// make sure all threads finished before continuing
#pragma omp barrier

    } // parallel region

    // make sure all requests are freed
    MPI_Waitall(MPI_buffer_requests.size(), &MPI_buffer_requests[0],
                MPI_STATUSES_IGNORE);

    // wait for all processes to exit the main loop
    MPI_Barrier(MPI_COMM_WORLD);

    // check that all requests finished (should be guaranteed by the MPI_Waitall
    // call above)
    for (unsigned int i = 0; i < MPI_buffer_requests.size(); ++i) {
      myassert(MPI_buffer_requests[i] == MPI_REQUEST_NULL,
               "Not all communications were finished, but the main loop "
               "finished nonetheless!");
    }

    // some useful log output to help us determine a good value for the queue
    // and task space sizes
    logmessage("Total number of tasks: " << tasks.size(), 0);

    logmessage("Updating copies...", 0);

    // combine the counter values for subgrids with copies
    // if the original is on the local process, this is easy. If it is not, we
    // need to communicate.
    std::vector< MPI_Request > requests(originals.size(), MPI_REQUEST_NULL);
    unsigned int num_to_receive = 0;
    // we just assume this is big enough (for now)
    const size_t buffer_part_size = MPI_buffer_size / (originals.size() + 1);
    for (unsigned int i = 0; i < originals.size(); ++i) {
      const unsigned int original = originals[i];
      const unsigned int copy = tot_num_subgrid + i;
      if (costs.get_process(original) == MPI_rank) {
        if (costs.get_process(copy) == MPI_rank) {
          // no communication required
          gridvec[original]->update_intensities(*gridvec[copy]);
        } else {
          // we need to set up a recv
          ++num_to_receive;
        }
      } else {
        if (costs.get_process(copy) == MPI_rank) {
          // we need to send
          gridvec[copy]->pack(&MPI_buffer[i * buffer_part_size],
                              buffer_part_size);
          unsigned int sendsize = gridvec[copy]->get_MPI_size();
          MPI_Isend(&MPI_buffer[i * buffer_part_size], sendsize, MPI_PACKED,
                    costs.get_process(original), original, MPI_COMM_WORLD,
                    &requests[i]);
        } // else: no action required
      }
    }

    // this time we do use MPI_Probe to figure out the size of the incoming
    // message
    unsigned int num_received = 0;
    while (num_received < num_to_receive) {
      MPI_Status status;
      MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
      const int source = status.MPI_SOURCE;
      const int tag = status.MPI_TAG;
      int size;
      MPI_Get_count(&status, MPI_PACKED, &size);
      MPI_Recv(&MPI_buffer[originals.size() * buffer_part_size], size,
               MPI_PACKED, source, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

      const int ncell_dummy[3] = {1, 1, 1};
      DensitySubGrid dummy(box, ncell_dummy);
      dummy.unpack(&MPI_buffer[originals.size() * buffer_part_size], size);

      // the tag tells us which original to update
      gridvec[tag]->update_intensities(dummy);

      ++num_received;
    }

    // make sure all requests are freed
    MPI_Waitall(requests.size(), &requests[0], MPI_STATUSES_IGNORE);

    // wait for all processes to get here (not necessary, but useful)
    MPI_Barrier(MPI_COMM_WORLD);

    logmessage("Done.", 0);

    // STEP 2: update the ionization structure for each subgrid
    logmessage("Updating ionisation structure", 0);
    unsigned int igrid = 0;
#pragma omp parallel default(shared)
    {
      while (igrid < tot_num_subgrid) {
        const unsigned int current_igrid = atomic_post_increment(igrid);
        if (current_igrid < tot_num_subgrid) {
          // only subgrids on this process are done
          if (costs.get_process(current_igrid) == MPI_rank) {
            gridvec[current_igrid]->compute_neutral_fraction(num_photon);
          }
        }
      }
    }

    // update the neutral fractions for copies
    // if the original is on the local process, this is easy. If it is not, we
    // need to communicate.
    num_to_receive = 0;
    for (unsigned int i = 0; i < originals.size(); ++i) {
      const unsigned int original = originals[i];
      const unsigned int copy = tot_num_subgrid + i;
      if (costs.get_process(copy) == MPI_rank) {
        if (costs.get_process(original) == MPI_rank) {
          // no communication required
          gridvec[copy]->update_neutral_fractions(*gridvec[original]);
        } else {
          // we need to set up a recv
          ++num_to_receive;
        }
      } else {
        if (costs.get_process(original) == MPI_rank) {
          // we need to send
          gridvec[original]->pack(&MPI_buffer[i * buffer_part_size],
                                  buffer_part_size);
          unsigned int sendsize = gridvec[original]->get_MPI_size();
          MPI_Isend(&MPI_buffer[i * buffer_part_size], sendsize, MPI_PACKED,
                    costs.get_process(copy), copy, MPI_COMM_WORLD,
                    &requests[i]);
        } // else: no action required
      }
    }

    // very similar to the send code above
    num_received = 0;
    while (num_received < num_to_receive) {
      MPI_Status status;
      MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
      const int source = status.MPI_SOURCE;
      const int tag = status.MPI_TAG;
      int size;
      MPI_Get_count(&status, MPI_PACKED, &size);
      MPI_Recv(&MPI_buffer[originals.size() * buffer_part_size], size,
               MPI_PACKED, source, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

      const int ncell_dummy[3] = {1, 1, 1};
      DensitySubGrid dummy(box, ncell_dummy);
      dummy.unpack(&MPI_buffer[originals.size() * buffer_part_size], size);

      // the tag tells us which original to update
      gridvec[tag]->update_neutral_fractions(dummy);

      ++num_received;
    }

    MPI_Waitall(requests.size(), &requests[0], MPI_STATUSES_IGNORE);

    MPI_Barrier(MPI_COMM_WORLD);

    // output useful information about this iteration (if enabled)
    logmessage("Writing task and cost information", 0);
    output_tasks(iloop, tasks);
    output_messages(iloop, message_log, message_log_size);
    output_costs(iloop, tot_num_subgrid, costs, copies, originals);

#ifdef SINGLE_ITERATION
    // stop here to see how we did for 1 iteration
    MPI_Barrier(MPI_COMM_WORLD);

    output_neutral_fractions(costs, gridvec, tot_num_subgrid);

    return MPI_Finalize();
#endif

    // reset the MPI request counter
    MPI_last_request = 0;

    // clear message log
    message_log_size = 0;

    // clear task buffer
    tasks.clear();

#ifdef DO_REBALANCING
    // rebalance the work based on the cost information during the last
    // iteration
    // we first need to communicate cost information
    logmessage("Rebalancing", 0);
    // fill initial_cost_vector with costs
    unsigned long avg_cost_per_thread = 0;
    for (unsigned int i = 0; i < tot_num_subgrid; ++i) {
      initial_cost_vector[i] = costs.get_computational_cost(i);
      avg_cost_per_thread += initial_cost_vector[i];
    }
    for (unsigned int i = 0; i < originals.size(); ++i) {
      initial_cost_vector[originals[i]] +=
          costs.get_computational_cost(tot_num_subgrid + i);
      avg_cost_per_thread += initial_cost_vector[originals[i]];
    }
    avg_cost_per_thread /= num_threads;

    // update the copy levels
    for (unsigned int i = 0; i < tot_num_subgrid; ++i) {
      if (copy_factor * initial_cost_vector[i] > avg_cost_per_thread) {
        // note that this in principle should be 1 higher. However, we do not
        // count the original.
        unsigned int number_of_copies =
            copy_factor * initial_cost_vector[i] / avg_cost_per_thread;
        // get the highest bit
        unsigned int level = 0;
        while (number_of_copies > 0) {
          number_of_copies >>= 1;
          ++level;
        }
        levels[i] = level;
      }
    }

    // reset neighbours to old values before we make new copies
    for (unsigned int i = 0; i < tot_num_subgrid; ++i) {
      for (int j = 1; j < 27; ++j) {
        if (gridvec[i]->get_neighbour(j) != NEIGHBOUR_OUTSIDE) {
          if (gridvec[i]->get_neighbour(j) >= tot_num_subgrid) {
            gridvec[i]->set_neighbour(j, originals[i - tot_num_subgrid]);
          }
        }
      }
    }

    // delete old copies
    for (unsigned int i = 0; i < originals.size(); ++i) {
      // free all associated buffers
      for (int j = 0; j < 27; ++j) {
        if (gridvec[tot_num_subgrid + i]->get_neighbour(j) !=
            NEIGHBOUR_OUTSIDE) {
          new_buffers.free_buffer(
              gridvec[tot_num_subgrid + i]->get_active_buffer(j));
        }
      }
      delete gridvec[tot_num_subgrid + i];
    }
    gridvec.resize(tot_num_subgrid);

    // make new copies
    originals.clear();
    copies.clear();
    copies.resize(tot_num_subgrid, 0xffffffff);
    create_copies(gridvec, levels, new_buffers, originals, copies);

    costs.reset(gridvec.size());
    // make sure costs are up to date
    for (unsigned int igrid = 0; igrid < tot_num_subgrid; ++igrid) {
      std::vector< unsigned int > this_copies;
      unsigned int copy = copies[igrid];
      if (copy < 0xffffffff) {
        while (copy < gridvec.size() &&
               originals[copy - tot_num_subgrid] == igrid) {
          this_copies.push_back(copy);
          ++copy;
        }
      }
      if (this_copies.size() > 0) {
        const unsigned long cost =
            initial_cost_vector[igrid] / (this_copies.size() + 1);
        costs.set_computational_cost(igrid, cost);
        for (unsigned int i = 0; i < this_copies.size(); ++i) {
          costs.set_computational_cost(this_copies[i], cost);
        }
      } else {
        costs.set_computational_cost(igrid, initial_cost_vector[igrid]);
      }
    }

    // redistribute the subgrids among the threads to balance the computational
    // costs (based on this iteration)
    costs.redistribute(ngbs);

// now do the communication: some subgrids might move between processes
// ...
#else // DO_REBALANCING
    costs.clear_costs();
#endif

  } // main loop

  ///////////////////

  //////////////////////////////////////////////////////////////////////////////

  //////////////////////////////////////////////////////////////////////////////
  /// Clean up
  //////////////////////////////////////////////////////////////////////////////

  ///////////////////////
  // Output final result
  //////////////////////

  output_neutral_fractions(costs, gridvec, tot_num_subgrid);

  //////////////////////

  ///////////////////////////////////
  // Output memory usage information
  //////////////////////////////////

  struct rusage resource_usage;
  getrusage(RUSAGE_SELF, &resource_usage);
  const size_t max_memory = static_cast< size_t >(resource_usage.ru_maxrss) *
                            static_cast< size_t >(1024);
  logmessage(
      "Maximum memory usage: " << Utilities::human_readable_bytes(max_memory),
      0);

  //////////////////////////////////

  //////////////////////
  // Garbage collection
  /////////////////////

  // grid
  for (unsigned int igrid = 0; igrid < gridvec.size(); ++igrid) {
    delete gridvec[igrid];
  }

  // queues
  for (int i = 0; i < num_threads; ++i) {
    delete new_queues[i];
  }

  // MPI buffer
  delete[] MPI_buffer;

  /////////////////////

  ////////////////
  // Clean up MPI
  ///////////////

  const int MPI_exit_code = MPI_Finalize();

  ///////////////

  //////////////////////////////////////////////////////////////////////////////

  // finally: stop timing and output the result
  program_timer.stop();
  logmessage("Total program time: " << program_timer.value() << " s.", 0);

  return MPI_exit_code;
}
