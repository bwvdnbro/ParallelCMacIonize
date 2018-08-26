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

// Defines: we do these first, as some includes depend on them

/*! @brief Output log level. The higher the value, the more stuff is printed to
 *  the stderr. Comment to disable logging altogether. */
#define LOG_OUTPUT 1

/*! @brief Enable this to disable all run time assertions and output that could
 *  slow down the algorithm. */
#define MEANING_OF_HASTE

/*! @brief Output. Select at least one of the below. */
//#define MM_FILE
//#define BIN_FILE
#define HDF5_FILE

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

/*! @brief Enable this to output an ASCII file with the result. */
//#define OUTPUT_ASCII_RESULT

/*! @brief Enable this to output task statistics. */
#define TASK_STATS

/*! @brief Enable this to output queue statistics. */
#define QUEUE_STATS

/*! @brief Enable this to output memory space statistics. */
#define MEMORYSPACE_STATS

/*! @brief Enable this to track memory usage manually. */
#define MEMORY_TRACKING

/*! @brief Disable hydro variables to decrease memory footprint of grid. */
#define NO_HYDRO

////////////////////////////////////////////////////////////////////////////////

#ifdef MEANING_OF_HASTE
#undef DO_ASSERTS
#undef TASK_OUTPUT
#undef COST_OUTPUT
#undef MESSAGE_OUTPUT
#undef OUTPUT_ASCII_RESULT
#undef TASK_STATS
#undef QUEUE_STATS
#undef MEMORYSPACE_STATS
#undef EDGECOST_STATS
#undef MEMORY_TRACKING
#endif

////////////////////////////////////////////////////////////////////////////////

// defines required by included files

#ifdef TASK_OUTPUT
// activate task output in Task.hpp
#define TASK_PLOT
#endif

#ifdef QUEUE_STATS
// activate queue statistics in Queue.hpp
#define QUEUE_STATS
#endif

#if defined(MEMORYSPACE_STATS) || defined(TASK_STATS)
#define THREADSAFEVECTOR_STATS
#endif

////////////////////////////////////////////////////////////////////////////////

#ifdef MEMORY_TRACKING
// global memory tracking variables
unsigned long current_memory_size, max_memory_size;

/**
 * @brief Update the maximum memory load variable.
 */
#define memory_tracking_update()                                               \
  max_memory_size = std::max(max_memory_size, current_memory_size);
#endif

/**
 * @brief Initialize the memory tracking variables.
 */
#ifdef MEMORY_TRACKING
#define memory_tracking_init()                                                 \
  current_memory_size = 0;                                                     \
  max_memory_size = 0;
#else
#define memory_tracking_init()
#endif

/**
 * @brief Log the use of the given variable in memory.
 *
 * @param variable Variable name or type to log.
 */
#ifdef MEMORY_TRACKING
#define memory_tracking_log_variable(variable)                                 \
  current_memory_size += sizeof(variable);                                     \
  memory_tracking_update();
#else
#define memory_tracking_log_variable(variable)
#endif

/**
 * @brief Unlog the use of the given variable in memory.
 *
 * @param variable Variable name or type to unlog.
 */
#ifdef MEMORY_TRACKING
#define memory_tracking_unlog_variable(variable)                               \
  current_memory_size -= sizeof(variable);
#else
#define memory_tracking_unlog_variable(variable)
#endif

/**
 * @brief Log the use of the given memory size in memory.
 *
 * @param size Size in memory to log (in bytes).
 */
#ifdef MEMORY_TRACKING
#define memory_tracking_log_size(size)                                         \
  current_memory_size += size;                                                 \
  memory_tracking_update();
#else
#define memory_tracking_log_size(size)
#endif

/**
 * @brief Unlog the use of the given memory size in memory.
 *
 * @param size Size in memory to unlog (in bytes).
 */
#ifdef MEMORY_TRACKING
#define memory_tracking_unlog_size(size) current_memory_size -= size;
#else
#define memory_tracking_unlog_size(size)
#endif

/**
 * @brief Print information about the memory usage.
 */
#ifdef MEMORY_TRACKING
#define memory_tracking_report()                                               \
  output_memory_size("Maximum memory usage", max_memory_size);
#else
#define memory_tracking_report()
#endif

/**
 * @brief Output the total memory size accross all processes (assuming the size
 * for each process is stored in the given variable).
 *
 * @param label Label to prepend to the output message.
 * @param variable Size in memory for a single process.
 */
#define output_memory_size(label, variable)                                    \
  { logmessage(label << ": " << Utilities::human_readable_bytes(variable), 0); }

////////////////////////////////////////////////////////////////////////////////

// Project includes
#include "CoarseDensityGrid.hpp"
#include "CommandLineParser.hpp"
#include "DensitySubGrid.hpp"
#include "DistributedDensityGrid.hpp"
#include "Log.hpp"
#include "MemorySpace.hpp"
#include "PhotonBuffer.hpp"
#include "Queue.hpp"
#include "RandomGenerator.hpp"
#include "Source.hpp"
#include "Task.hpp"
#include "Timer.hpp"
#include "Utilities.hpp"
#include "YAMLDictionary.hpp"

// standard library includes
#include <cmath>
#include <fstream>
#include <hdf5.h>
#include <iostream>
#include <omp.h>
#include <sstream>
#include <sys/resource.h>
#include <vector>

/**
 * @brief Write a file with the start and end times of all tasks.
 *
 * @param iloop Iteration number (added to file name).
 * @param tasks Tasks to print.
 * @param iteration_start Start CPU cycle count of the iteration on this
 * process.
 * @param iteration_end End CPU cycle count of the iteration on this process.
 */
inline void output_tasks(const unsigned int iloop,
                         ThreadSafeVector< Task > &tasks,
                         const unsigned long iteration_start,
                         const unsigned long iteration_end) {
#ifdef TASK_STATS
  {
    // first compose the file name
    std::stringstream filename;
    filename << "task_stats_";
    filename.fill('0');
    filename.width(2);
    filename << iloop;
    filename << ".txt";

    // now output
    // open the file
    std::ofstream ofile(filename.str(), std::ofstream::trunc);

    ofile << "# rank\tsize\n";

    ofile << "0\t" << tasks.get_max_number_taken() << "\n";
    tasks.reset_max_number_taken();
  }
#endif

#ifdef TASK_OUTPUT
  {
    // compose the file name
    std::stringstream filename;
    filename << "tasks_";
    filename.fill('0');
    filename.width(2);
    filename << iloop;
    filename << ".txt";

    // now output
    // open the file
    std::ofstream ofile(filename.str(), std::ofstream::trunc);

    ofile << "# rank\tthread\tstart\tstop\ttype\n";

    // write the start and end CPU cycle count
    // this is a dummy task executed by thread 0 (so that the min or max
    // thread count is not affected), but with non-existing type -1
    ofile << "0\t0\t" << iteration_start << "\t" << iteration_end << "\t-1\n";

    // write the task info
    const size_t tsize = tasks.size();
    for (size_t i = 0; i < tsize; ++i) {
      const Task &task = tasks[i];
      myassert(task.done(), "Task was never executed!");
      int type, thread_id;
      unsigned long start, end;
      task.get_timing_information(type, thread_id, start, end);
      ofile << "0\t" << thread_id << "\t" << start << "\t" << end << "\t"
            << type << "\n";
    }
  }
#endif
}

/**
 * @brief Write file with queue size information for an iteration.
 *
 * @param iloop Iteration number (added to file names).
 * @param queues Per thread queues.
 * @param general_queue General queue.
 */
inline void output_queues(const unsigned int iloop,
                          std::vector< Queue * > &queues,
                          Queue &general_queue) {
#ifdef QUEUE_STATS
  // first compose the file name
  std::stringstream filename;
  filename << "queues_";
  filename.fill('0');
  filename.width(2);
  filename << iloop;
  filename << ".txt";

  // now output
  // open the file
  std::ofstream ofile(filename.str(), std::ofstream::trunc);

  ofile << "# rank\tqueue\tsize\n";

  // start with the general queue (-1)
  ofile << "0\t-1\t" << general_queue.get_max_queue_size() << "\n";
  general_queue.reset_max_queue_size();

  // now do the other queues
  for (size_t i = 0; i < queues.size(); ++i) {
    Queue &queue = *queues[i];
    ofile << "0\t" << i << "\t" << queue.get_max_queue_size() << "\n";
    queue.reset_max_queue_size();
  }
#endif
}

/**
 * @brief Write file with memory space size information for an iteration.
 *
 * @param iloop Iteration number (added to file names).
 * @param memory_space Memory space.
 */
inline void output_memoryspace(const unsigned int iloop,
                               MemorySpace &memory_space) {
#ifdef MEMORYSPACE_STATS
  // first compose the file name
  std::stringstream filename;
  filename << "memoryspace_";
  filename.fill('0');
  filename.width(2);
  filename << iloop;
  filename << ".txt";

  // now output
  // open the file
  std::ofstream ofile(filename.str(), std::ofstream::trunc);

  ofile << "# rank\tsize\n";

  ofile << "0\t" << memory_space.get_max_number_elements() << "\n";
  memory_space.reset_max_number_elements();
#endif
}

/**
 * @brief Output the neutral fractions for inspection of the physical result.
 *
 * @param gridvec Subgrids.
 * @param tot_num_subgrid Total number of original subgrids.
 */
inline size_t
output_neutral_fractions(const std::vector< DensitySubGrid * > &gridvec,
                         const unsigned int tot_num_subgrid) {

#ifdef OUTPUT_ASCII_RESULT
  // ASCII output (for the VisIt plot script)
  // open the file
  std::ofstream ofile("intensities.txt", std::ofstream::trunc);

  // write the neutral fractions
  for (unsigned int igrid = 0; igrid < tot_num_subgrid; ++igrid) {
    gridvec[igrid]->print_intensities(ofile);
  }
#endif

  // binary output
  // we use a memory-mapped file, which allows us
  //  - to write simultaneously using multiple threads
  //  - to write simultaneously using multiple processes
  // we can do this because we know the size of the file beforehand, and we can
  // assign a unique location in the file to each subgrid

  // find the size (in bytes) of a single subgrid
  const size_t blocksize = gridvec[0]->get_output_size();

  // now memory-map the file
  // the file is shared by all processes
  MemoryMap file("neutral_fractions.dat", tot_num_subgrid * blocksize);

  memory_tracking_log_size(file.get_memory_size());

  // write the subgrids using multiple threads
  Atomic< unsigned int > igrid(0);
#pragma omp parallel default(shared)
  {
    while (igrid.value() < tot_num_subgrid) {
      const unsigned int this_igrid = igrid.post_increment();
      // only write local subgrids
      if (this_igrid < tot_num_subgrid) {
        const size_t offset = this_igrid * blocksize;
        gridvec[this_igrid]->output_intensities(offset, file);
      }
    }
  }

  memory_tracking_unlog_size(file.get_memory_size());

  return tot_num_subgrid * blocksize;
}

/**
 * @brief Output the neutral fractions for inspection of the physical result.
 *
 * This version uses HDF5.
 *
 * @param gridvec Subgrids.
 * @param tot_num_subgrid Total number of original subgrids.
 * @return Total size that was written (in bytes).
 */
inline size_t
output_neutral_fractions_hdf5(const std::vector< DensitySubGrid * > &gridvec,
                              const unsigned int tot_num_subgrid) {

  const hid_t file = H5Fcreate("neutral_fractions.hdf5", H5F_ACC_TRUNC,
                               H5P_DEFAULT, H5P_DEFAULT);

  const hid_t group =
      H5Gcreate(file, "/PartType0", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

  const hid_t space = H5Screate(H5S_SIMPLE);

  const size_t blocksize = gridvec[0]->get_number_of_cells();

  const int rank = 1;
  const hsize_t shape[1] = {tot_num_subgrid * blocksize};

  H5Sset_extent_simple(space, rank, shape, shape);

  const hid_t data = H5Dcreate(group, "NeutralFractionH", H5T_NATIVE_DOUBLE,
                               space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

  const hsize_t slab_shape[1] = {blocksize};
  const hid_t memspace = H5Screate(H5S_SIMPLE);
  H5Sset_extent_simple(memspace, rank, slab_shape, slab_shape);

  size_t output_size = 0;
  for (unsigned int igrid = 0; igrid < tot_num_subgrid; ++igrid) {
    const hsize_t offset[1] = {igrid * blocksize};
    H5Sselect_hyperslab(space, H5S_SELECT_SET, offset, nullptr, slab_shape,
                        nullptr);
    H5Dwrite(data, H5T_NATIVE_DOUBLE, memspace, space, H5P_DEFAULT,
             gridvec[igrid]->get_neutral_fraction());
    output_size += blocksize * sizeof(double);
  }

  H5Sclose(memspace);
  H5Dclose(data);
  H5Sclose(space);
  H5Gclose(group);
  H5Fclose(file);

  return output_size;
}

/**
 * @brief Output the neutral fractions for inspection of the physical result.
 *
 * This version uses a simple binary file.
 *
 * @param gridvec Subgrids.
 * @param tot_num_subgrid Total number of original subgrids.
 * @return Total size that was written (in bytes).
 */
inline size_t
output_neutral_fractions_binary(const std::vector< DensitySubGrid * > &gridvec,
                                const unsigned int tot_num_subgrid) {

  std::ofstream bfile("neutral_fractions.bin", std::ios::binary);

  const size_t blocksize = gridvec[0]->get_number_of_cells();

  for (unsigned int igrid = 0; igrid < tot_num_subgrid; ++igrid) {
    bfile.write(reinterpret_cast< const char * >(
                    gridvec[igrid]->get_neutral_fraction()),
                blocksize * sizeof(double));
  }

  return blocksize * tot_num_subgrid * sizeof(double);
}

/**
 * @brief Draw a random direction.
 *
 * @param random_generator Random number generator to use.
 * @param direction Random direction (output).
 */
inline static void get_random_direction(RandomGenerator &random_generator,
                                        double *direction) {

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
}

/**
 * @brief Fill the given PhotonBuffer with random photons.
 *
 * @param buffer PhotonBuffer to fill.
 * @param number_of_photons Number of photons to draw randomly.
 * @param random_generator RandomGenerator used to generate random numbers.
 * @param source_index Index of the subgrid that contains the source.
 * @param source Actual source.
 */
inline static void fill_buffer(PhotonBuffer &buffer,
                               const unsigned int number_of_photons,
                               RandomGenerator &random_generator,
                               const unsigned int source_index,
                               const Source &source) {

  // set general buffer information
  buffer.grow(number_of_photons);
  buffer.set_subgrid_index(source_index);
  buffer.set_direction(TRAVELDIRECTION_INSIDE);

  double position[3];
  source.get_position(position);

  // draw random photons and store them in the buffer
  for (unsigned int i = 0; i < number_of_photons; ++i) {

    Photon &photon = buffer[i];

    // initial position: we currently assume a single source at the origin
    photon.set_position(position[0], position[1], position[2]);

    // initial direction: isotropic distribution
    get_random_direction(random_generator, photon.get_direction());

    // we currently assume equal weight for all photons
    photon.set_weight(1.);

    // target optical depth (exponential distribution)
    photon.set_target_optical_depth(
        -std::log(random_generator.get_uniform_random_double()));

    // this is the fixed cross section we use for the moment
    photon.set_photoionization_cross_section(6.3e-22);

    // make sure the photon is moving in *a* direction
    myassert(photon.get_direction()[0] != 0. ||
                 photon.get_direction()[1] != 0. ||
                 photon.get_direction()[2] != 0.,
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
  for (int i = 0; i < TRAVELDIRECTION_NUMBER; ++i) {
    myassert(!output_buffer_flags[i] || output_buffers[i].size() == 0,
             "Non-empty starting output buffer!");
  }

  // now loop over the input buffer photons and traverse them one by one
  for (unsigned int i = 0; i < input_buffer.size(); ++i) {

    // active photon
    Photon &photon = input_buffer[i];

    // make sure the photon is moving in *a* direction
    myassert(photon.get_direction()[0] != 0. ||
                 photon.get_direction()[1] != 0. ||
                 photon.get_direction()[2] != 0.,
             "size: " << input_buffer.size());

    // traverse the photon through the active subgrid
    const int result = subgrid.interact(photon, input_buffer.get_direction());

    // check that the photon ended up in a valid output buffer
    myassert(result >= 0 && result < TRAVELDIRECTION_NUMBER, "fail");

    // add the photon to an output buffer, if it still exists (if the
    // corresponding output buffer does not exist, this means the photon left
    // the simulation box)
    if (output_buffer_flags[result]) {

      // get the correct output buffer
      PhotonBuffer &output_buffer = output_buffers[result];

      // add the photon
      const unsigned int index = output_buffer.get_next_free_photon();
      output_buffer[index] = photon;

      // make sure we actually added this photon
      myassert(output_buffer[index].get_position()[0] ==
                       photon.get_position()[0] &&
                   output_buffer[index].get_position()[1] ==
                       photon.get_position()[1] &&
                   output_buffer[index].get_position()[2] ==
                       photon.get_position()[2],
               "fail");
      myassert(output_buffer[index].get_direction()[0] != 0. ||
                   output_buffer[index].get_direction()[1] != 0. ||
                   output_buffer[index].get_direction()[2] != 0.,
               "size: " << output_buffer.size());

      // check that the output buffer did not overflow
      myassert(output_buffer.size() <= PHOTONBUFFER_SIZE,
               "output buffer size: " << output_buffer.size());
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
  for (unsigned int i = 0; i < buffer.size(); ++i) {
    // only a fraction (= 'reemission_probability') of the photons is actually
    // reemitted
    if (random_generator.get_uniform_random_double() < reemission_probability) {
      // give the photon a new random isotropic direction
      Photon &photon = buffer[i];
      get_random_direction(random_generator, photon.get_direction());
      // reset the target optical depth
      photon.set_target_optical_depth(
          -std::log(random_generator.get_uniform_random_double()));
      // NOTE: we can never overwrite a photon that should be preserved (we
      // either overwrite the photon itself, or a photon that was not reemitted)
      buffer[index] = photon;
      ++index;
    }
  }
  // update the active size of the buffer: some photons were not reemitted, so
  // the active size will shrink
  buffer.grow(index);
}

/**
 * @brief Impose restrictions on the given copy levels.
 *
 * We want to make sure the level differences between neighbouring subgrids are
 * small enough. We do this in multiple stages:
 *  - we first loop over all levels and find the maximum level
 *  - we then loop over all subgrids. For subgrids that have their level equal
 *    to the maximum, we check the level difference between that subgrid and its
 *    direct neighbours. If the level difference is larger than 1, we increase
 *    the level for that neighbour.
 *  - we then lower the maximum level and repeat the procedure until the maximum
 *    level is 1
 *
 * @param levels Cost-based copy level of each subgrid.
 * @param num_subgrid Number of subgrids in each dimension.
 */
inline void copy_restrictions(std::vector< unsigned char > &levels,
                              int num_subgrid[3]) {
  unsigned char max_level = 0;
  const size_t levelsize = levels.size();
  for (size_t i = 0; i < levelsize; ++i) {
    max_level = std::max(max_level, levels[i]);
  }

  while (max_level > 0) {
    for (size_t i = 0; i < levelsize; ++i) {
      if (levels[i] == max_level) {
        const int ix = i / (num_subgrid[1] * num_subgrid[2]);
        const int iy =
            (i - ix * num_subgrid[1] * num_subgrid[2]) / num_subgrid[2];
        const int iz =
            i - ix * num_subgrid[1] * num_subgrid[2] - iy * num_subgrid[2];
        if (ix > 0) {
          const size_t ngbi = (ix - 1) * num_subgrid[1] * num_subgrid[2] +
                              iy * num_subgrid[2] + iz;
          if (levels[ngbi] < levels[i] - 1) {
            levels[ngbi] = levels[i] - 1;
          }
        }
        if (ix < num_subgrid[0] - 1) {
          const size_t ngbi = (ix + 1) * num_subgrid[1] * num_subgrid[2] +
                              iy * num_subgrid[2] + iz;
          if (levels[ngbi] < levels[i] - 1) {
            levels[ngbi] = levels[i] - 1;
          }
        }
        if (iy > 0) {
          const size_t ngbi = ix * num_subgrid[1] * num_subgrid[2] +
                              (iy - 1) * num_subgrid[2] + iz;
          if (levels[ngbi] < levels[i] - 1) {
            levels[ngbi] = levels[i] - 1;
          }
        }
        if (iy < num_subgrid[1] - 1) {
          const size_t ngbi = ix * num_subgrid[1] * num_subgrid[2] +
                              (iy + 1) * num_subgrid[2] + iz;
          if (levels[ngbi] < levels[i] - 1) {
            levels[ngbi] = levels[i] - 1;
          }
        }
        if (iz > 0) {
          const size_t ngbi = ix * num_subgrid[1] * num_subgrid[2] +
                              iy * num_subgrid[2] + iz - 1;
          if (levels[ngbi] < levels[i] - 1) {
            levels[ngbi] = levels[i] - 1;
          }
        }
        if (iz < num_subgrid[2] - 1) {
          const size_t ngbi = ix * num_subgrid[1] * num_subgrid[2] +
                              iy * num_subgrid[2] + iz + 1;
          if (levels[ngbi] < levels[i] - 1) {
            levels[ngbi] = levels[i] - 1;
          }
        }
      }
    }
    --max_level;
  }
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
    }
    // now do the actual neighbours
    for (int j = 1; j < TRAVELDIRECTION_NUMBER; ++j) {
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
 * @param number_of_sources Number of photon sources.
 * @param random_density Generate a random density field?
 * @param queue_size_per_thread Variable to store the size of the queue for
 * each thread in.
 * @param memoryspace_size Variable to store the size of the memory space in.
 * @param number_of_tasks Variable to store the size of the task space in.
 * @param MPI_buffer_size Variable to store the size of the MPI buffer in.
 * @param cost_copy_factor Variable to store the cost copy factor in.
 * @param edge_copy_factor Variable to store the edge copy factor in.
 * @param general_queue_size General queue size.
 */
inline void read_parameters(
    std::string paramfile_name, double box[6], double &reemission_probability,
    int ncell[3], int num_subgrid[3], unsigned int &num_photon,
    unsigned int &number_of_iterations, unsigned int &number_of_sources,
    bool &random_density, unsigned int &queue_size_per_thread,
    unsigned int &memoryspace_size, unsigned int &number_of_tasks,
    unsigned int &MPI_buffer_size, double &cost_copy_factor,
    double &edge_copy_factor, unsigned int &general_queue_size) {

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
  number_of_sources = parameters.get_value< unsigned int >("number_of_sources");
  random_density = parameters.get_value< bool >("random_density");

  queue_size_per_thread =
      parameters.get_value< unsigned int >("queue_size_per_thread");
  memoryspace_size = parameters.get_value< unsigned int >("memoryspace_size");
  number_of_tasks = parameters.get_value< unsigned int >("number_of_tasks");
  MPI_buffer_size = parameters.get_value< unsigned int >("MPI_buffer_size");
  cost_copy_factor = parameters.get_value< double >("cost_copy_factor");
  edge_copy_factor = parameters.get_value< double >("edge_copy_factor");

  general_queue_size =
      parameters.get_value< unsigned int >("general_queue_size");

  logmessage("\n##\n# Parameters:\n##", 0);
  parameters.print_contents(std::cerr, true);
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
 * @param num_photon_local Total number of photon packets that needs to be
 * generated at the source on this process.
 * @param tasks Task space to create new tasks in.
 * @param new_queues Task queues for individual threads.
 * @param new_buffers PhotonBuffer memory space.
 * @param random_generator Random_generator for the active thread.
 * @param gridvec List of all subgrids.
 * @param num_active_buffers Number of active photon buffers on this process.
 * @param num_tasks_to_add Counter for the number of newly created tasks.
 * @param tasks_to_add List of tasks to add after this tasks finishes.
 * @param queues_to_add Queues to add new tasks to.
 * @param sources Photon sources.
 */
inline void execute_source_photon_task(
    Task &task, const int thread_id, const unsigned int num_photon_local,
    ThreadSafeVector< Task > &tasks, std::vector< Queue * > &new_queues,
    MemorySpace &new_buffers, RandomGenerator &random_generator,
    std::vector< DensitySubGrid * > &gridvec,
    Atomic< unsigned int > &num_active_buffers, unsigned int &num_tasks_to_add,
    unsigned int *tasks_to_add, int *queues_to_add,
    const std::vector< Source * > &sources) {

  // log the start time of the task (if task output is enabled)
  task.start(thread_id);

  // we will create a new buffer
  num_active_buffers.pre_increment();

  unsigned int num_photon_this_loop = task.get_buffer();
  unsigned int source_index = task.get_subgrid();
  const Source &source = *sources[source_index];

  // get a free photon buffer in the central queue
  unsigned int buffer_index = new_buffers.get_free_buffer();
  PhotonBuffer &input_buffer = new_buffers[buffer_index];
  const unsigned int this_central_index =
      source.get_subgrid_index(random_generator);

  // now actually fill the buffer with random photon packets
  fill_buffer(input_buffer, num_photon_this_loop, random_generator,
              this_central_index, source);

  // add to the queue of the corresponding thread
  DensitySubGrid &subgrid = *gridvec[this_central_index];
  const size_t task_index = tasks.get_free_element();
  Task &new_task = tasks[task_index];
  new_task.set_type(TASKTYPE_PHOTON_TRAVERSAL);
  new_task.set_subgrid(this_central_index);
  new_task.set_buffer(buffer_index);

  // add dependency for task:
  //  - subgrid
  // (the output buffers belong to the subgrid and do not count as a dependency)
  new_task.set_dependency(subgrid.get_dependency());

  queues_to_add[num_tasks_to_add] = subgrid.get_owning_thread();
  tasks_to_add[num_tasks_to_add] = task_index;
  ++num_tasks_to_add;

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
 * @param num_photon_done Number of photon packets that has been completely
 * finished.
 * @param num_empty Number of empty buffers on this process.
 * @param num_active_buffers Number of active photon buffers on this process.
 * @param num_tasks_to_add Counter for the number of newly created tasks.
 * @param tasks_to_add List of tasks to add after this tasks finishes.
 * @param queues_to_add Queues to add new tasks to.
 */
inline void execute_photon_traversal_task(
    Task &task, const int thread_id, ThreadSafeVector< Task > &tasks,
    std::vector< Queue * > &new_queues, Queue &general_queue,
    MemorySpace &new_buffers, std::vector< DensitySubGrid * > &gridvec,
    PhotonBuffer *local_buffers, bool *local_buffer_flags,
    const double reemission_probability,
    Atomic< unsigned int > &num_photon_done, Atomic< unsigned int > &num_empty,
    Atomic< unsigned int > &num_active_buffers, unsigned int &num_tasks_to_add,
    unsigned int *tasks_to_add, int *queues_to_add) {

  // log the start of the task
  task.start(thread_id);

  const unsigned int current_buffer_index = task.get_buffer();
  PhotonBuffer &buffer = new_buffers[current_buffer_index];
  const unsigned int igrid = buffer.get_subgrid_index();
  DensitySubGrid &this_grid = *gridvec[igrid];

#ifdef EDGECOST_STATS
  const int input_direction = buffer.get_direction();
  if (input_direction > 0) {
    // photons enter the subgrid through an edge: log the communication cost
    this_grid.add_communication_cost(input_direction, buffer.size());
  }
#endif

  // prepare output buffers: make sure they are empty and that buffers
  // corresponding to directions outside the simulation box are
  // disabled
  for (int i = 0; i < TRAVELDIRECTION_NUMBER; ++i) {
    const unsigned int ngb = this_grid.get_neighbour(i);
    if (ngb != NEIGHBOUR_OUTSIDE) {
      local_buffer_flags[i] = true;
      local_buffers[i].reset();
    } else {
      local_buffer_flags[i] = false;
    }
  }

  // if reemission is disabled, disable output to the internal buffer
  if (reemission_probability == 0.) {
    local_buffer_flags[TRAVELDIRECTION_INSIDE] = false;
  }

  // keep track of the original number of photons
  unsigned int num_photon_done_now = buffer.size();

  // now do the actual photon traversal
  do_photon_traversal(buffer, this_grid, local_buffers, local_buffer_flags);

  // add none empty buffers to the appropriate queues
  unsigned char largest_index = TRAVELDIRECTION_NUMBER;
  unsigned int largest_size = 0;
  for (int i = 0; i < TRAVELDIRECTION_NUMBER; ++i) {

    // only process enabled, non-empty output buffers
    if (local_buffer_flags[i] && local_buffers[i].size() > 0) {

#ifdef EDGECOST_STATS
      // photons leave the subgrid through an edge: log the communication cost
      if (i > 0) {
        this_grid.add_communication_cost(i, local_buffers[i].size());
      }
#endif

      // photon packets that are still present in an output buffer
      // are not done yet
      num_photon_done_now -= local_buffers[i].size();

      // move photon packets from the local temporary buffer (that is
      // guaranteed to be large enough) to the actual output buffer
      // for that direction (which might cause on overflow)
      const unsigned int ngb = this_grid.get_neighbour(i);
      unsigned int new_index = this_grid.get_active_buffer(i);

      if (new_index == NEIGHBOUR_OUTSIDE) {
        // buffer was not created yet: create it now
        new_index = new_buffers.get_free_buffer();
        PhotonBuffer &buffer = new_buffers[new_index];
        buffer.set_subgrid_index(ngb);
        buffer.set_direction(TravelDirections::output_to_input_direction(i));
        this_grid.set_active_buffer(i, new_index);
      }

      if (new_buffers[new_index].size() == 0) {
        // we are adding photons to an empty buffer
        num_empty.pre_decrement();
      }
      unsigned int add_index =
          new_buffers.add_photons(new_index, local_buffers[i]);

      // check if the original buffer is full
      if (add_index != new_index) {

        // a new active buffer was created
        num_active_buffers.pre_increment();

        // new_buffers.add_photons already created a new empty
        // buffer, set it as the active buffer for this output
        // direction
        this_grid.set_active_buffer(i, add_index);
        if (new_buffers[add_index].size() == 0) {
          // we have created a new empty buffer
          num_empty.pre_increment();
        }

        // YES: create a task for the buffer and add it to the queue
        // the task type depends on the buffer: photon packets in the
        // internal buffer were absorbed and could be reemitted,
        // photon packets in the other buffers left the subgrid and
        // need to be traversed in the neighbouring subgrid
        if (i > 0) {
          DensitySubGrid &subgrid =
              *gridvec[new_buffers[new_index].get_subgrid_index()];
          const size_t task_index = tasks.get_free_element();
          Task &new_task = tasks[task_index];
          new_task.set_subgrid(new_buffers[new_index].get_subgrid_index());
          new_task.set_buffer(new_index);
          new_task.set_type(TASKTYPE_PHOTON_TRAVERSAL);

          // add dependencies for task:
          //  - subgrid
          new_task.set_dependency(subgrid.get_dependency());

          // add the task to the queue of the corresponding thread
          const unsigned int queue_index = gridvec[ngb]->get_owning_thread();
          queues_to_add[num_tasks_to_add] = queue_index;
          tasks_to_add[num_tasks_to_add] = task_index;
          ++num_tasks_to_add;
        } else {
          const size_t task_index = tasks.get_free_element();
          Task &new_task = tasks[task_index];
          new_task.set_subgrid(new_buffers[new_index].get_subgrid_index());
          new_task.set_buffer(new_index);
          new_task.set_type(TASKTYPE_PHOTON_REEMIT);
          // a reemit task has no direct dependencies
          // add the task to the general queue
          queues_to_add[num_tasks_to_add] = -1;
          tasks_to_add[num_tasks_to_add] = task_index;
          ++num_tasks_to_add;
        }

        myassert(new_buffers[add_index].get_subgrid_index() == ngb,
                 "Wrong subgrid");
        myassert(new_buffers[add_index].get_direction() ==
                     TravelDirections::output_to_input_direction(i),
                 "Wrong direction");

      } // if (add_index != new_index)

    } // if (local_buffer_flags[i] &&
    //     local_buffers[i]._actual_size > 0)

    // we have to do this outside the other condition, as buffers to which
    // nothing was added can still be non-empty...
    if (local_buffer_flags[i]) {
      unsigned int new_index = this_grid.get_active_buffer(i);
      if (new_index != NEIGHBOUR_OUTSIDE &&
          new_buffers[new_index].size() > largest_size) {
        largest_index = i;
        largest_size = new_buffers[new_index].size();
      }
    }

  } // for (int i = TRAVELDIRECTION_NUMBER - 1; i >= 0; --i)

  this_grid.set_largest_buffer(largest_index, largest_size);

  // add photons that were absorbed (if reemission was disabled) or
  // that left the system to the global count
  num_photon_done.pre_add(num_photon_done_now);

  // delete the original buffer, as we are done with it
  new_buffers.free_buffer(current_buffer_index);

  myassert(num_active_buffers.value() > 0, "Number of active buffers < 0!");
  num_active_buffers.pre_decrement();

  // log the end time of the task
  task.stop();
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
 * @param num_photon_done Number of photon packets that has been completely
 * finished.
 * @param gridvec DensitySubGrids.
 * @param num_tasks_to_add Counter for the number of newly created tasks.
 * @param tasks_to_add List of tasks to add after this tasks finishes.
 * @param queues_to_add Queues to add new tasks to.
 */
inline void execute_photon_reemit_task(
    Task &task, const int thread_id, ThreadSafeVector< Task > &tasks,
    std::vector< Queue * > &new_queues, MemorySpace &new_buffers,
    RandomGenerator &random_generator, const double reemission_probability,
    Atomic< unsigned int > &num_photon_done,
    std::vector< DensitySubGrid * > &gridvec, unsigned int &num_tasks_to_add,
    unsigned int *tasks_to_add, int *queues_to_add) {

  // log the start of the task
  task.start(thread_id);

  // get the buffer
  const unsigned int current_buffer_index = task.get_buffer();
  PhotonBuffer &buffer = new_buffers[current_buffer_index];

  // keep track of the original number of photons in the buffer
  unsigned int num_photon_done_now = buffer.size();

  // reemit photon packets
  do_reemission(buffer, random_generator, reemission_probability);

  // find the number of photon packets that was absorbed and not
  // reemitted...
  num_photon_done_now -= buffer.size();
  // ...and add it to the global count
  num_photon_done.pre_add(num_photon_done_now);

  // the reemitted photon packets are ready to be propagated: create
  // a new propagation task
  DensitySubGrid &subgrid = *gridvec[task.get_subgrid()];
  const size_t task_index = tasks.get_free_element();
  Task &new_task = tasks[task_index];
  new_task.set_type(TASKTYPE_PHOTON_TRAVERSAL);
  new_task.set_subgrid(task.get_subgrid());
  new_task.set_buffer(current_buffer_index);

  // add dependency
  new_task.set_dependency(subgrid.get_dependency());

  // add it to the queue of the corresponding thread
  queues_to_add[num_tasks_to_add] =
      gridvec[buffer.get_subgrid_index()]->get_owning_thread();
  tasks_to_add[num_tasks_to_add] = task_index;
  ++num_tasks_to_add;

  // log the end time of the task
  task.stop();
}

/**
 * @brief Execute a single task.
 *
 * @param task_index Index of a task to execute.
 * @param thread_id Thread that will execute the task.
 * @param num_photon_local Total number of photon packets that needs to be
 * generated at the source on this process.
 * @param tasks Task space to create new tasks in.
 * @param new_queues Task queues for individual threads.
 * @param general_queue General queue shared by all threads.
 * @param new_buffers PhotonBuffer memory space.
 * @param random_generator Random_generator for the active thread.
 * @param gridvec List of all subgrids.
 * @param local_buffers Temporary buffers used in photon packet propagation
 * tasks on this thread.
 * @param local_buffer_flags Temporary flags used in photon packet propagation
 * tasks on this thread.
 * @param reemission_probability Reemission probability.
 * @param num_photon_done Number of photon packets that has been completely
 * finished.
 * @param MPI_lock Lock to guarantee thread-safe access to the MPI library.
 * @param new_MPI_buffer MPI communication buffer.
 * @param message_log MPI message log.
 * @param message_log_size Size of the MPI message log.
 * @param num_empty Number of empty buffers on this process.
 * @param num_active_buffers Number of active photon buffers on this process.
 * @param num_tasks_to_add Counter for the number of newly created tasks.
 * @param tasks_to_add List of tasks to add after this tasks finishes.
 * @param queues_to_add Queues to add new tasks to.
 * @param sources Photon sources.
 */
inline void execute_task(
    const unsigned int task_index, const int thread_id,
    const unsigned int num_photon_local, ThreadSafeVector< Task > &tasks,
    std::vector< Queue * > &new_queues, Queue &general_queue,
    MemorySpace &new_buffers, RandomGenerator &random_generator,
    std::vector< DensitySubGrid * > &gridvec, PhotonBuffer *local_buffers,
    bool *local_buffer_flags, const double reemission_probability,
    Atomic< unsigned int > &num_photon_done, Atomic< unsigned int > &num_empty,
    Atomic< unsigned int > &num_active_buffers, unsigned int &num_tasks_to_add,
    unsigned int *tasks_to_add, int *queues_to_add,
    const std::vector< Source * > &sources) {

  Task &task = tasks[task_index];

  myassert(!task.done(), "Task already executed!");

  // Different tasks are processed in different ways.
  switch (task.get_type()) {
  case TASKTYPE_SOURCE_PHOTON:

    /// generate random photon packets from the source

    execute_source_photon_task(task, thread_id, num_photon_local, tasks,
                               new_queues, new_buffers, random_generator,
                               gridvec, num_active_buffers, num_tasks_to_add,
                               tasks_to_add, queues_to_add, sources);
    break;

  case TASKTYPE_PHOTON_TRAVERSAL:

    /// propagate photon packets from a buffer through a subgrid

    execute_photon_traversal_task(
        task, thread_id, tasks, new_queues, general_queue, new_buffers, gridvec,
        local_buffers, local_buffer_flags, reemission_probability,
        num_photon_done, num_empty, num_active_buffers, num_tasks_to_add,
        tasks_to_add, queues_to_add);
    break;

  case TASKTYPE_PHOTON_REEMIT:

    /// reemit absorbed photon packets

    execute_photon_reemit_task(task, thread_id, tasks, new_queues, new_buffers,
                               random_generator, reemission_probability,
                               num_photon_done, gridvec, num_tasks_to_add,
                               tasks_to_add, queues_to_add);
    break;

  default:

    // should never happen
    cmac_error("Unknown task: %i!", task.get_type());
  }

  // we're done with the task, unlock its dependency
  task.unlock_dependency();
}

/**
 * @brief Prematurely activate a buffer to feed hungry threads.
 *
 * @param current_index Index of a task. Should be NO_TASK upon entry. Upon
 * exit, this variable will contain the index of a task that can be executed by
 * the current thread.
 * @param thread_id Thread that executes this code.
 * @param num_threads Number of threads that is currently running.
 * @param tasks Task space.
 * @param new_queues Per thread task queues.
 * @param general_queue General queue that is shared by all threads.
 * @param new_buffers PhotonBuffer memory space.
 * @param gridvec List of subgrids.
 * @param num_empty Number of empty buffers on this process.
 * @param num_active_buffers Number of active photon buffers on this process.
 */
inline void activate_buffer(unsigned int &current_index, const int thread_id,
                            const int num_threads,
                            ThreadSafeVector< Task > &tasks,
                            std::vector< Queue * > &new_queues,
                            Queue &general_queue, MemorySpace &new_buffers,
                            std::vector< DensitySubGrid * > &gridvec,
                            Atomic< unsigned int > &num_empty,
                            Atomic< unsigned int > &num_active_buffers) {

  // we first try to launch buffers that are nearly full, and then move on to
  // more empty buffers
  unsigned int threshold_size = PHOTONBUFFER_SIZE;
  while (threshold_size > 0) {
    threshold_size >>= 1;
    // loop over all subgrids
    for (unsigned int igrid = 0; igrid < gridvec.size(); ++igrid) {
      // try to lock the subgrid
      if (gridvec[igrid]->get_largest_buffer_size() > threshold_size &&
          gridvec[igrid]->get_dependency()->try_lock()) {

        // get the largest buffer
        const unsigned char largest_index =
            gridvec[igrid]->get_largest_buffer_index();
        if (largest_index != TRAVELDIRECTION_NUMBER) {

          // prematurely launch the buffer
          const unsigned int non_full_index =
              gridvec[igrid]->get_active_buffer(largest_index);
          // Create a new empty buffer and set it as active buffer for
          // this subgrid.
          const unsigned int new_index = new_buffers.get_free_buffer();
          new_buffers[new_index].set_subgrid_index(
              new_buffers[non_full_index].get_subgrid_index());
          new_buffers[new_index].set_direction(
              new_buffers[non_full_index].get_direction());
          gridvec[igrid]->set_active_buffer(largest_index, new_index);
          // we are creating a new active photon buffer
          num_active_buffers.pre_increment();
          // we created a new empty buffer
          num_empty.pre_increment();

          const size_t task_index = tasks.get_free_element();
          Task &new_task = tasks[task_index];
          new_task.set_subgrid(new_buffers[non_full_index].get_subgrid_index());
          new_task.set_buffer(non_full_index);
          if (largest_index > 0) {
            DensitySubGrid &subgrid =
                *gridvec[new_buffers[non_full_index].get_subgrid_index()];
            new_task.set_type(TASKTYPE_PHOTON_TRAVERSAL);

            // add dependency
            new_task.set_dependency(subgrid.get_dependency());

            const unsigned int queue_index =
                gridvec[new_buffers[non_full_index].get_subgrid_index()]
                    ->get_owning_thread();
            new_queues[queue_index]->add_task(task_index);
          } else {
            new_task.set_type(TASKTYPE_PHOTON_REEMIT);
            // a reemit task has no dependencies
            general_queue.add_task(task_index);
          }

          // set the new largest index
          unsigned char new_largest_index = TRAVELDIRECTION_NUMBER;
          unsigned int new_largest_size = 0;
          for (unsigned char ibuffer = 0; ibuffer < TRAVELDIRECTION_NUMBER;
               ++ibuffer) {
            if (gridvec[igrid]->get_active_buffer(ibuffer) !=
                    NEIGHBOUR_OUTSIDE &&
                new_buffers[gridvec[igrid]->get_active_buffer(ibuffer)].size() >
                    new_largest_size) {
              new_largest_index = ibuffer;
              new_largest_size =
                  new_buffers[gridvec[igrid]->get_active_buffer(ibuffer)]
                      .size();
            }
          }
          gridvec[igrid]->set_largest_buffer(new_largest_index,
                                             new_largest_size);

          // unlock the subgrid, we are done with it
          gridvec[igrid]->get_dependency()->unlock();

          // we managed to activate a buffer, we are done with this function
          return;
        } else {
          // no semi-full buffers for this subgrid: release the lock again
          gridvec[igrid]->get_dependency()->unlock();
        }
      }
    }
  }
}

/**
 * @brief Steal a task from another queue.
 *
 * @param thread_id Id of the active thread.
 * @param num_threads Total number of threads.
 * @param new_queues Thread queues.
 * @param tasks Task space.
 * @param gridvec Grid.
 * @return Index of an available task, or NO_TASK if no tasks are available.
 */
inline unsigned int steal_task(const int thread_id, const int num_threads,
                               std::vector< Queue * > &new_queues,
                               ThreadSafeVector< Task > &tasks,
                               std::vector< DensitySubGrid * > &gridvec) {

  // sort the queues by size
  std::vector< unsigned int > queue_sizes(new_queues.size(), 0);
  for (unsigned int i = 0; i < new_queues.size(); ++i) {
    queue_sizes[i] = new_queues[i]->size();
  }
  std::vector< size_t > sorti = Utilities::argsort(queue_sizes);

  // now try to steal from the largest queue first
  unsigned int current_index = NO_TASK;
  unsigned int i = 0;
  while (current_index == NO_TASK && i < queue_sizes.size() &&
         queue_sizes[sorti[queue_sizes.size() - i - 1]] > 0) {
    current_index =
        new_queues[sorti[queue_sizes.size() - i - 1]]->try_get_task(tasks);
    ++i;
  }
  if (current_index != NO_TASK) {
    // stealing means transferring ownership...
    if (tasks[current_index].get_type() == TASKTYPE_PHOTON_TRAVERSAL) {
      gridvec[tasks[current_index].get_subgrid()]->set_owning_thread(thread_id);
    }
  }

  return current_index;
}

/**
 * @brief Randomly distribute the given integer number over the given total
 * number of array elements so that on average every array element is equally
 * likely to receive a contribution.
 *
 * @param total_number Total number of array elements.
 * @param distribute_number Number to distribute.
 * @param random_generator RandomGenerator to use.
 * @return Vector of length total_number with elements that are randomly either
 * 0 or 1, but so that the total sum of all elements equals distribute_number.
 */
inline std::vector< unsigned int >
distribute(const unsigned int total_number,
           const unsigned int distribute_number,
           RandomGenerator &random_generator) {

  // create a vector of length total_number and fill it with random values
  std::vector< double > random_array(total_number);
  std::vector< unsigned int > index_array(total_number);
  std::vector< unsigned int > result_array(total_number, 0);
  for (unsigned int i = 0; i < total_number; ++i) {
    random_array[i] = random_generator.get_uniform_random_double();
    index_array[i] = i;
  }

  // index sort the random vector
  std::sort(index_array.begin(), index_array.end(),
            [random_array](unsigned int i0, unsigned int i1) {
              return random_array[i0] < random_array[i1];
            });

  // now set the elements corresponding to the first distribute_number elements
  // in index_array to 1
  for (unsigned int i = 0; i < distribute_number; ++i) {
    result_array[index_array[i]] = 1;
  }

  return result_array;
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

  //////////////////////////////////////////////////////////////////////////////
  /// Initialization
  //////////////////////////////////////////////////////////////////////////////

  memory_tracking_init();

  //////////////////////
  // MPI initialization
  /////////////////////

  Timer program_timer;
  program_timer.start();

  unsigned long program_start, program_end;
  cpucycle_tick(program_start);

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
  unsigned int num_photon, number_of_iterations, number_of_sources;
  bool random_density;

  unsigned int queue_size_per_thread, memoryspace_size, number_of_tasks,
      MPI_buffer_size, general_queue_size;
  double cost_copy_factor, edge_copy_factor;

  read_parameters(paramfile_name, box, reemission_probability, ncell,
                  num_subgrid, num_photon, number_of_iterations,
                  number_of_sources, random_density, queue_size_per_thread,
                  memoryspace_size, number_of_tasks, MPI_buffer_size,
                  cost_copy_factor, edge_copy_factor, general_queue_size);

  //////////////////////////

  ////////////////////////////////
  // Set up task based structures
  ///////////////////////////////

  // set up the queues used to queue tasks
  std::vector< Queue * > new_queues(num_threads, nullptr);
  for (int i = 0; i < num_threads; ++i) {
    new_queues[i] = new Queue(queue_size_per_thread);

    memory_tracking_log_size(new_queues[i]->get_memory_size());
  }

  Queue general_queue(general_queue_size);

  memory_tracking_log_size(general_queue.get_memory_size());

  // set up the task space used to store tasks
  ThreadSafeVector< Task > tasks(number_of_tasks);

  memory_tracking_log_size(tasks.get_memory_size());

  // set up the memory space used to store photon packet buffers
  MemorySpace new_buffers(memoryspace_size);

  memory_tracking_log_size(new_buffers.get_memory_size());

  // set up the cost vector used to load balance
  const unsigned int tot_num_subgrid =
      num_subgrid[0] * num_subgrid[1] * num_subgrid[2];

  ///////////////////////////////

  ///////////////////////////////////////
  // Set up the random number generators
  //////////////////////////////////////

  std::vector< RandomGenerator > random_generator(num_threads);
  for (int i = 0; i < num_threads; ++i) {
    // make sure every thread on every process has a different seed
    random_generator[i].set_seed(42 + num_threads + i);
  }

  //////////////////////////////////////

  ///////////////////
  // Set up the grid
  //////////////////

  // set up the grid of smaller grids used for the algorithm
  // each smaller grid stores a fraction of the total grid and has information
  // about the neighbouring subgrids
  std::vector< DensitySubGrid * > gridvec(tot_num_subgrid, nullptr);

  Atomic< size_t > grid_memory(0);

  const double subbox_side[3] = {box[3] / num_subgrid[0],
                                 box[4] / num_subgrid[1],
                                 box[5] / num_subgrid[2]};
  const int subbox_ncell[3] = {ncell[0] / num_subgrid[0],
                               ncell[1] / num_subgrid[1],
                               ncell[2] / num_subgrid[2]};

  // set up the subgrids (in parallel)
  Atomic< unsigned int > atomic_index(0);
#pragma omp parallel default(shared)
  {
    // id of this specific thread
    const int thread_id = omp_get_thread_num();
    while (atomic_index.value() < tot_num_subgrid) {
      const unsigned int index = atomic_index.post_increment();
      if (index < tot_num_subgrid) {
        const int ix = index / (num_subgrid[1] * num_subgrid[2]);
        const int iy =
            (index - ix * num_subgrid[1] * num_subgrid[2]) / num_subgrid[2];
        const int iz =
            index - ix * num_subgrid[1] * num_subgrid[2] - iy * num_subgrid[2];
        const double subbox[6] = {box[0] + ix * subbox_side[0],
                                  box[1] + iy * subbox_side[1],
                                  box[2] + iz * subbox_side[2],
                                  subbox_side[0],
                                  subbox_side[1],
                                  subbox_side[2]};
        gridvec[index] = new DensitySubGrid(subbox, subbox_ncell);
        grid_memory.pre_add(gridvec[index]->get_memory_size());
        DensitySubGrid &this_grid = *gridvec[index];
        this_grid.set_owning_thread(thread_id);
        // set up neighbouring information. We first make sure all
        // neighbours are initialized to NEIGHBOUR_OUTSIDE, indicating no
        // neighbour
        for (int i = 0; i < TRAVELDIRECTION_NUMBER; ++i) {
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
                  ciy < num_subgrid[1] && ciz >= 0 && ciz < num_subgrid[2]) {
                // we use get_output_direction() to get the correct index
                // for the neighbour
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
              } // if ci
            }   // for niz
          }     // for niy
        }       // for nix
      }         // if local index
    }
  } // end parallel region

  memory_tracking_log_size(grid_memory.value());

  if (random_density) {
    // generate random k-modes for the random density distribution
    const unsigned int nk = 10;
    std::vector< double > k(3 * nk);
    std::vector< double > phase(3 * nk);
    for (unsigned int i = 0; i < nk; ++i) {
      const double u[4] = {random_generator[0].get_uniform_random_double(),
                           random_generator[0].get_uniform_random_double(),
                           random_generator[0].get_uniform_random_double(),
                           random_generator[0].get_uniform_random_double()};

      const double kx =
          std::sqrt(-2. * std::log(u[0])) * std::cos(2. * M_PI * u[1]);
      const double ky =
          std::sqrt(-2. * std::log(u[0])) * std::sin(2. * M_PI * u[1]);
      const double kz =
          std::sqrt(-2. * std::log(u[2])) * std::cos(2. * M_PI * u[3]);

      k[3 * i] = 4. * M_PI * kx / box[3];
      k[3 * i + 1] = 4. * M_PI * ky / box[4];
      k[3 * i + 2] = 4. * M_PI * kz / box[5];

      phase[3 * i] =
          2. * M_PI * random_generator[0].get_uniform_random_double();
      phase[3 * i + 1] =
          2. * M_PI * random_generator[0].get_uniform_random_double();
      phase[3 * i + 2] =
          2. * M_PI * random_generator[0].get_uniform_random_double();
    }

// initialize the density field
#pragma omp parallel for default(shared)
    for (unsigned int igrid = 0; igrid < tot_num_subgrid; ++igrid) {
      const unsigned int ncell = gridvec[igrid]->get_number_of_cells();
      for (unsigned int icell = 0; icell < ncell; ++icell) {
        double midpoint[3];
        gridvec[igrid]->get_cell_midpoint(icell, midpoint);

        // get the density
        double ndens = 0.;
        for (unsigned int ik = 0; ik < nk; ++ik) {
          const double kx = k[3 * ik];
          const double ky = k[3 * ik + 1];
          const double kz = k[3 * ik + 2];

          ndens += 1.e8 * (std::cos(kx * midpoint[0] + phase[3 * ik]) *
                               std::cos(ky * midpoint[1] + phase[3 * ik + 1]) *
                               std::cos(kz * midpoint[2] + phase[3 * ik + 2]) +
                           1.);
        }
        ndens /= nk;

        gridvec[igrid]->set_number_density(icell, ndens);
      }
    }
  }

  //////////////////

  ////////////////////////////////////////////
  // Initialize the photon source(s)
  ///////////////////////////////////////////

  std::vector< Source * > sources;
  double total_luminosity = 0.;
  logmessage("Box anchor: " << box[0] << " " << box[1] << " " << box[2], 2);
  logmessage("Box sides: " << box[3] << " " << box[4] << " " << box[5], 2);
  if (number_of_sources > 1) {
    for (size_t isrc = 0; isrc < number_of_sources; ++isrc) {
      double position[3];
      position[0] =
          box[0] + random_generator[0].get_uniform_random_double() * box[3];
      position[1] =
          box[1] + random_generator[0].get_uniform_random_double() * box[4];
      position[2] =
          box[2] + random_generator[0].get_uniform_random_double() * box[5];

      logmessage("position: " << position[0] << " " << position[1] << " "
                              << position[2],
                 2);

      const unsigned int source_indices[3] = {
          static_cast< unsigned int >((position[0] - box[0]) / box[3] *
                                      num_subgrid[0]),
          static_cast< unsigned int >((position[1] - box[1]) / box[4] *
                                      num_subgrid[1]),
          static_cast< unsigned int >((position[2] - box[2]) / box[5] *
                                      num_subgrid[2])};

      const unsigned int source_index =
          source_indices[0] * num_subgrid[1] * num_subgrid[2] +
          source_indices[1] * num_subgrid[2] + source_indices[2];

      logmessage("source_indices: " << source_indices[0] << " "
                                    << source_indices[1] << " "
                                    << source_indices[2],
                 2);
      logmessage("source index: " << source_index, 2);

      sources.push_back(new Source(position, 4.26e48));
      sources[isrc]->add_subgrid_index(source_index);

      total_luminosity += sources[isrc]->get_luminosity();
    }
  } else {
    for (size_t isrc = 0; isrc < 1; ++isrc) {
      const double position[3] = {0., 0., 0.};

      logmessage("position: " << position[0] << " " << position[1] << " "
                              << position[2],
                 2);

      const unsigned int source_indices[3] = {
          static_cast< unsigned int >((position[0] - box[0]) / box[3] *
                                      num_subgrid[0]),
          static_cast< unsigned int >((position[1] - box[1]) / box[4] *
                                      num_subgrid[1]),
          static_cast< unsigned int >((position[2] - box[2]) / box[5] *
                                      num_subgrid[2])};

      const unsigned int source_index =
          source_indices[0] * num_subgrid[1] * num_subgrid[2] +
          source_indices[1] * num_subgrid[2] + source_indices[2];

      logmessage("source_indices: " << source_indices[0] << " "
                                    << source_indices[1] << " "
                                    << source_indices[2],
                 2);
      logmessage("source index: " << source_index, 2);

      sources.push_back(new Source(position, 4.26e49));
      sources[isrc]->add_subgrid_index(source_index);

      total_luminosity += sources[isrc]->get_luminosity();
    }
  }

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

  logmessage(
      "Running low resolution simulation to get initial subgrid costs...", 0);
  // we need a good cost estimate for each subgrid to load balance across
  // nodes
  // to obtain this, we run a low resolution Monte Carlo simulation on the
  // grid of subgrids (with each subgrid acting as a single cell)
  CoarseDensityGrid coarse_grid(box, num_subgrid);
  // run 10 photon buffers
  RandomGenerator coarse_rg(42);
  for (unsigned int iloop = 0; iloop < number_of_iterations; ++iloop) {
    for (unsigned int i = 0; i < 10; ++i) {
      for (unsigned int isrc = 0; isrc < sources.size(); ++isrc) {
        PhotonBuffer buffer;
        fill_buffer(buffer, PHOTONBUFFER_SIZE, coarse_rg, 0, *sources[isrc]);
        // now loop over the buffer photons and traverse them one by one
        for (unsigned int j = 0; j < buffer.size(); ++j) {

          // active photon
          Photon &photon = buffer[j];

          // traverse the photon through the active subgrid
          coarse_grid.interact(photon);
        }
      }
    }

    // get the maximal intensity
    double maxintensity = 0.;
    for (int ix = 0; ix < num_subgrid[0]; ++ix) {
      for (int iy = 0; iy < num_subgrid[1]; ++iy) {
        for (int iz = 0; iz < num_subgrid[2]; ++iz) {
          maxintensity = std::max(
              maxintensity, coarse_grid.get_intensity_integral(ix, iy, iz));
        }
      }
    }

    // now set the subgrid cost based on the intensity counters
    for (int ix = 0; ix < num_subgrid[0]; ++ix) {
      for (int iy = 0; iy < num_subgrid[1]; ++iy) {
        for (int iz = 0; iz < num_subgrid[2]; ++iz) {
          const unsigned int igrid =
              ix * num_subgrid[1] * num_subgrid[2] + iy * num_subgrid[2] + iz;
          const double intensity =
              coarse_grid.get_intensity_integral(ix, iy, iz);
          // pro tip: although it is nowhere explicitly stated, it turns out
          // that metis vertex weight values are only enforced properly if they
          // are small enough. I suspect that under the hood Metis sums the
          // weights for each partition, and if these sums cause overflows,
          // weird partitions are produced. A maximum weight value of 0xffff
          // seems to produce nice result.
          // Similar problems were observed when CPU cycles were used as weight,
          // because again the weights are too large.
          const unsigned long cost = 0xffff * (intensity / maxintensity);
          initial_photon_cost[igrid] += cost;
        }
      }
    }

    coarse_grid.compute_neutral_fraction(
        total_luminosity, 10 * sources.size() * PHOTONBUFFER_SIZE);
  }

  std::fill(initial_cost_vector.begin(), initial_cost_vector.end(), 1);

  logmessage("Done.", 0);

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
  std::vector< unsigned int > originals;
  std::vector< unsigned int > copies(tot_num_subgrid, 0xffffffff);
  std::vector< unsigned char > levels(tot_num_subgrid, 0);

  // get the average cost per thread and average edge cost
  unsigned long avg_cost_per_thread = 0;
  for (unsigned int i = 0; i < tot_num_subgrid; ++i) {
    avg_cost_per_thread += initial_photon_cost[i];
  }
  avg_cost_per_thread /= num_threads;
  // now set the levels accordingly
  for (unsigned int i = 0; i < tot_num_subgrid; ++i) {
    if (cost_copy_factor * initial_photon_cost[i] > avg_cost_per_thread) {
      // note that this in principle should be 1 higher. However, we do not
      // count the original.
      unsigned int number_of_copies = std::ceil(
          (cost_copy_factor * initial_photon_cost[i]) / avg_cost_per_thread);
      // make sure the number of copies of the source subgrid is at least
      // equal to the number of cores
      if (number_of_copies < static_cast< unsigned int >(num_threads)) {
        number_of_copies = num_threads;
      }
      // get the highest bit
      unsigned int level = 0;
      while (number_of_copies > 1) {
        number_of_copies >>= 1;
        ++level;
      }
      levels[i] = level;
    }
  }

  // impose copy restrictions
  copy_restrictions(levels, num_subgrid);

  // output the copy levels
  {
    std::ofstream ofile("copy_levels.txt");
    ofile << "# subgrid\tlevel\n";
    for (size_t i = 0; i < levels.size(); ++i) {
      unsigned int level = levels[i];
      ofile << i << "\t" << level << "\n";
    }
  }

  // now create the copies
  create_copies(gridvec, levels, new_buffers, originals, copies);

  // add copies of sources
  for (unsigned int isrc = 0; isrc < sources.size(); ++isrc) {
    const unsigned int original = sources[isrc]->get_original_subgrid_index();
    unsigned int copy = copies[original];
    if (copy < 0xffffffff) {
      while (copy < gridvec.size() &&
             originals[copy - tot_num_subgrid] == original) {
        sources[isrc]->add_subgrid_index(copy);
        ++copy;
      }
    }
  }

  // now compute the actual grid size
  size_t grid_memory_size = 0;
  for (unsigned int i = 0; i < gridvec.size(); ++i) {
    grid_memory_size += gridvec[i]->get_memory_size();

    // initialize the photon buffers
    for (int ingb = 0; ingb < TRAVELDIRECTION_NUMBER; ++ingb) {
      gridvec[i]->set_active_buffer(ingb, NEIGHBOUR_OUTSIDE);
    }
  }
  memory_tracking_unlog_size(grid_memory.value());
  memory_tracking_log_size(grid_memory_size);

  ////////////////////////////////////////////////

  ////////////////////
  // Actual main loop
  ///////////////////

  Timer iteration_timer;
  iteration_timer.start();

  Timer MCtimer;

  Lock decision_lock;

  std::vector< unsigned int > source_numbers(sources.size());
  unsigned int num_photon_add = 0;
  for (unsigned int isrc = 0; isrc < sources.size(); ++isrc) {
    source_numbers[isrc] = static_cast< unsigned int >(
        sources[isrc]->get_luminosity() / total_luminosity * num_photon);
    num_photon_add += source_numbers[isrc];
  }
  const unsigned int num_photon_rem = num_photon - num_photon_add;
  myassert(num_photon_rem >= 0, "Negative remaining photon number");
  myassert(num_photon_rem < sources.size(), "Wrong remaining photon number");

  // now for the main loop. This loop
  //  - shoots num_photon photons through the grid to get intensity estimates
  //  - computes the ionization equilibrium
  for (unsigned int iloop = 0; iloop < number_of_iterations; ++iloop) {

    // store the CPU cycle count at the start of the iteration for this node
    unsigned long iteration_start, iteration_end;
    cpucycle_tick(iteration_start);
    MCtimer.start();

    // STEP 0: log output
    logmessage("Loop " << iloop + 1, 0);

    // STEP 1: photon shooting
    logmessage("Starting photon shoot loop", 0);
    // GLOBAL control variables (these are shared and updated atomically):
    //  - number of photon packets that has been created at the source
    Atomic< unsigned int > num_photon_sourced(0);
    //  - number of photon packets that has left the system, either through
    //    absorption or by crossing a simulation box wall
    unsigned int num_photon_done = 0;
    bool global_run_flag = true;
    // local control variables
    const unsigned int num_empty_target =
        TRAVELDIRECTION_NUMBER * gridvec.size();
    Atomic< unsigned int > num_empty(TRAVELDIRECTION_NUMBER * gridvec.size());
    Atomic< unsigned int > num_active_buffers(0);
    // global control variable
    Atomic< unsigned int > num_photon_done_since_last(0);

    // distribute the photon packets over the sources
    // first randomly distributed the remainder
    std::vector< unsigned int > remdist =
        distribute(sources.size(), num_photon_rem, random_generator[0]);
    // now obtain the number of buffers and photon size per source
    std::vector< unsigned int > actual_source_numbers(sources.size());
    std::vector< unsigned int > number_of_source_tasks(sources.size());
    size_t num_photon_tasks = 0;
    for (unsigned int isrc = 0; isrc < sources.size(); ++isrc) {
      actual_source_numbers[isrc] = source_numbers[isrc] + remdist[isrc];
      number_of_source_tasks[isrc] =
          actual_source_numbers[isrc] / PHOTONBUFFER_SIZE +
          (actual_source_numbers[isrc] % PHOTONBUFFER_SIZE > 0);
      num_photon_tasks += number_of_source_tasks[isrc];
    }

    // preallocate photon creation tasks for a faster iteration start
    tasks.get_free_elements(num_photon_tasks);
    size_t task_offset = 0;
    for (unsigned int isrc = 0; isrc < sources.size(); ++isrc) {
#pragma omp parallel for
      for (size_t i = 0; i < number_of_source_tasks[isrc]; ++i) {
        tasks[task_offset + i].set_type(TASKTYPE_SOURCE_PHOTON);
        tasks[task_offset + i].set_buffer(PHOTONBUFFER_SIZE);
        tasks[task_offset + i].set_subgrid(isrc);
      }
      if (actual_source_numbers[isrc] % PHOTONBUFFER_SIZE > 0) {
        tasks[task_offset + number_of_source_tasks[isrc] - 1].set_buffer(
            PHOTONBUFFER_SIZE -
            (actual_source_numbers[isrc] % PHOTONBUFFER_SIZE));
      }
      task_offset += number_of_source_tasks[isrc];
    }
    general_queue.add_tasks(0, num_photon_tasks);
#pragma omp parallel default(shared)
    {
      // id of this specific thread
      const int thread_id = omp_get_thread_num();
      PhotonBuffer local_buffers[TRAVELDIRECTION_NUMBER];
      bool local_buffer_flags[TRAVELDIRECTION_NUMBER];
      for (int i = 0; i < TRAVELDIRECTION_NUMBER; ++i) {
        local_buffers[i].set_direction(
            TravelDirections::output_to_input_direction(i));
        local_buffers[i].reset();
        local_buffer_flags[i] = true;
      }

      // this loop is repeated until all local photons have been propagated
      // note that this condition automatically covers the condition
      //  num_photon_sourced < num_photon
      // as unsourced photons cannot contribute to num_photon_done
      // this condition definitely needs to change for MPI...
      while (global_run_flag) {

        // get a first task
        // upon first entry of the while loop, this will be one of the photon
        // source tasks we just created
        unsigned int current_index = new_queues[thread_id]->get_task(tasks);
        if (current_index == NO_TASK) {
          current_index =
              steal_task(thread_id, num_threads, new_queues, tasks, gridvec);
          // still no task: take one from the general queue
          if (current_index == NO_TASK) {
            current_index = general_queue.get_task(tasks);
          }
        }

        // task activation: if no task is found, try to launch a photon buffer
        // that is not yet full and prematurely schedule it
        if (current_index == NO_TASK) {
          activate_buffer(current_index, thread_id, num_threads, tasks,
                          new_queues, general_queue, new_buffers, gridvec,
                          num_empty, num_active_buffers);
          current_index = new_queues[thread_id]->get_task(tasks);
          if (current_index == NO_TASK) {
            current_index =
                steal_task(thread_id, num_threads, new_queues, tasks, gridvec);
            // still no task: take one from the general queue
            if (current_index == NO_TASK) {
              current_index = general_queue.get_task(tasks);
            }
          }
        }

        // Keep processing tasks until the queue is empty.
        while (current_index != NO_TASK) {

          // we can maximally create TRAVELDIRECTION_NUMBER new tasks during
          // the execution of a task
          unsigned int num_tasks_to_add = 0;
          unsigned int tasks_to_add[TRAVELDIRECTION_NUMBER];
          int queues_to_add[TRAVELDIRECTION_NUMBER];

          execute_task(current_index, thread_id, num_photon, tasks, new_queues,
                       general_queue, new_buffers, random_generator[thread_id],
                       gridvec, local_buffers, local_buffer_flags,
                       reemission_probability, num_photon_done_since_last,
                       num_empty, num_active_buffers, num_tasks_to_add,
                       tasks_to_add, queues_to_add, sources);

          // add new tasks to their respective queues
          for (unsigned int itask = 0; itask < num_tasks_to_add; ++itask) {
            if (queues_to_add[itask] < 0) {
              // general queue
              general_queue.add_task(tasks_to_add[itask]);
            } else {
              new_queues[queues_to_add[itask]]->add_task(tasks_to_add[itask]);
            }
          }

// this would be the right place to delete the task (if we don't want
// to output it)
#ifndef TASK_OUTPUT
          tasks.free_element(current_index);
#endif

          // We finished a task: try to get a new task from the local queue
          current_index = new_queues[thread_id]->get_task(tasks);
          if (current_index == NO_TASK) {
            current_index =
                steal_task(thread_id, num_threads, new_queues, tasks, gridvec);
            // still no task: take one from the general queue
            if (current_index == NO_TASK) {
              current_index = general_queue.get_task(tasks);
            }
          }

        } // while (current_index != NO_TASK)

        // check if the local process finished
        if (num_empty.value() == num_empty_target &&
            num_active_buffers.value() == 0) {

          logmessage_lockfree("thread " << thread_id << " thinks we're ready!",
                              1);
          // we use a decision lock so that we know for sure we are the only
          // thread that has access to num_photon_done_since_last
          // note that for a pure shared memory run we could slightly alter
          // the decision making compared to the hybrid algorithm and not use
          // a lock at all
          if (decision_lock.try_lock()) {
            if (num_empty.value() == num_empty_target &&
                num_active_buffers.value() == 0 &&
                num_photon_done_since_last.value() > 0) {
              num_photon_done += num_photon_done_since_last.value();
              num_photon_done_since_last.set(0);
              logmessage_lockfree("num_photon_done = " << num_photon_done
                                                       << " (" << num_photon
                                                       << ")",
                                  1);
              if (num_photon_done == num_photon) {
                // make sure the master process also stops
                global_run_flag = false;
              }
            }
            decision_lock.unlock();
          }
        }

      } // while (global_run_flag)

    } // parallel region

    logmessage("Updating copies...", 0);

    for (unsigned int i = 0; i < originals.size(); ++i) {
      const unsigned int original = originals[i];
      const unsigned int copy = tot_num_subgrid + i;
      // no communication required
      gridvec[original]->update_intensities(*gridvec[copy]);
    }

    logmessage("Done.", 0);

    // STEP 2: update the ionization structure for each subgrid
    logmessage("Updating ionisation structure", 0);
    Atomic< unsigned int > igrid(0);
#pragma omp parallel default(shared)
    {
      while (igrid.value() < tot_num_subgrid) {
        const unsigned int current_igrid = igrid.post_increment();
        if (current_igrid < tot_num_subgrid) {
          gridvec[current_igrid]->compute_neutral_fraction(total_luminosity,
                                                           num_photon);
        }
      }
    }

    // update the neutral fractions for copies
    for (unsigned int i = 0; i < originals.size(); ++i) {
      const unsigned int original = originals[i];
      const unsigned int copy = tot_num_subgrid + i;
      // no communication required
      gridvec[copy]->update_neutral_fractions(*gridvec[original]);
    }
    cpucycle_tick(iteration_end);
    MCtimer.stop();

    // output useful information about this iteration (if enabled)
    logmessage("Writing task information", 0);
    output_tasks(iloop, tasks, iteration_start, iteration_end);
    output_queues(iloop, new_queues, general_queue);
    output_memoryspace(iloop, new_buffers);

    // clear task buffer
    tasks.clear();

  } // main loop

  iteration_timer.stop();

///////////////////

//////////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////////
/// Clean up
//////////////////////////////////////////////////////////////////////////////

///////////////////////
// Output final result
//////////////////////

#ifdef MM_FILE
  Timer mm_timer;
  mm_timer.start();
  size_t mm_size = output_neutral_fractions(gridvec, tot_num_subgrid);
  mm_timer.stop();
  logmessage("Writing memory-mapped file took " << mm_timer.value() << " s.",
             0);
  double mm_speed = mm_size / mm_timer.value() / (1024. * 1024.);
  logmessage("Writing speed: " << mm_speed << " MB/s.", 0);
#endif

#ifdef BIN_FILE
  Timer bin_timer;
  bin_timer.start();
  size_t bin_size = output_neutral_fractions_binary(gridvec, tot_num_subgrid);
  bin_timer.stop();
  logmessage("Writing binary file took " << bin_timer.value() << " s.", 0);
  double bin_speed = bin_size / bin_timer.value() / (1024. * 1024.);
  logmessage("Writing speed: " << bin_speed << " MB/s.", 0);
#endif

#ifdef HDF5_FILE
  Timer hdf5_timer;
  hdf5_timer.start();
  size_t hdf5_size = output_neutral_fractions_hdf5(gridvec, tot_num_subgrid);
  hdf5_timer.stop();
  logmessage("Writing HDF5 file took " << hdf5_timer.value() << " s.", 0);
  double hdf5_speed = hdf5_size / hdf5_timer.value() / (1024. * 1024.);
  logmessage("Writing speed: " << hdf5_speed << " MB/s.", 0);
#endif

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

  memory_tracking_report();

  logmessage("Basic grid size: "
                 << Utilities::human_readable_bytes(grid_memory.value()),
             0);
  output_memory_size("Actual grid size", grid_memory_size);

  output_memory_size("Memory space size", new_buffers.get_memory_size());
  output_memory_size("Task space size", tasks.get_memory_size());

  size_t queue_memory_size = general_queue.get_memory_size();
  for (int i = 0; i < num_threads; ++i) {
    queue_memory_size += new_queues[i]->get_memory_size();
  }
  output_memory_size("Queue size", queue_memory_size);

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

  for (unsigned int isrc = 0; isrc < sources.size(); ++isrc) {
    delete sources[isrc];
  }

  /////////////////////

  //////////////////////////////////////////////////////////////////////////////

  // stop the timers
  program_timer.stop();
  cpucycle_tick(program_end);

  // write the start and end CPU cycle count for each process, and the total
  // program time (for tick to time conversion)
  {
    std::string filename = "program_time.txt";
    // now open the file
    std::ofstream ofile(filename, std::ofstream::trunc);
    ofile << "# rank\tstart\tstop\ttime\n";
    ofile << "0\t" << program_start << "\t" << program_end << "\t"
          << program_timer.value() << "\n";
  }

  ///////////////

  // output timing results
  logmessage("Total Monte Carlo time: " << MCtimer.value() << " s.", 0);
  logmessage("Total photoionization loop time: " << iteration_timer.value()
                                                 << " s.",
             0);
  logmessage("Total program time: " << program_timer.value() << " s.", 0);

  return 0;
}
