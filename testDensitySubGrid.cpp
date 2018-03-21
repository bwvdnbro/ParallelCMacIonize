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

/*! @brief Uncomment this to enable run time assertions. */
#define DO_ASSERTS

/*! @brief Enable this to activate task output. */
#define TASK_OUTPUT

/*! @brief Enable this to activate cost output. */
#define COST_OUTPUT

#ifdef TASK_OUTPUT
// activate task output in Task.hpp
#define TASK_PLOT
#endif

// Project includes
#include "CommandLineParser.hpp"
#include "CostVector.hpp"
#include "DensitySubGrid.hpp"
#include "Log.hpp"
#include "MemorySpace.hpp"
#include "NewQueue.hpp"
#include "PhotonBuffer.hpp"
#include "RandomGenerator.hpp"
#include "Task.hpp"
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
inline void output_tasks(const unsigned int iloop,
                         const ThreadSafeVector< Task > &tasks) {
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
 * @param copies List that links subgrids to copies.
 * @param original List that links subgrid copies to originals.
 */
inline void output_costs(const unsigned int iloop, const unsigned int ngrid,
                         const int nthread, const CostVector &costs,
                         const std::vector< unsigned int > &copies,
                         const std::vector< unsigned int > &originals) {
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
    ofile << i << "\t" << costs.get_cost(i) << "\t" << costs.get_process(i)
          << "\t" << costs.get_thread(i) << "\n";
    if (copies[i] < 0xffffffff) {
      unsigned int copy = copies[i];
      while (copy - ngrid < originals.size() && originals[copy - ngrid] == i) {
        ofile << i << "\t" << costs.get_cost(copy) << "\t"
              << costs.get_process(copy) << "\t" << costs.get_thread(copy)
              << "\n";
        ++copy;
      }
    }
  }
#endif
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
inline void create_copies(std::vector< SubGrid * > &gridvec,
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
      gridvec.push_back(
          new DensitySubGrid(*static_cast< DensitySubGrid * >(gridvec[i])));
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
 */
inline void read_parameters(
    std::string paramfile_name, double box[6], double &reemission_probability,
    int ncell[3], int num_subgrid[3], unsigned int &num_photon,
    unsigned int &number_of_iterations, unsigned int &queue_size_per_thread,
    unsigned int &memoryspace_size, unsigned int &number_of_tasks,
    unsigned int &MPI_buffer_size, double &copy_factor) {

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

  logmessage("\n##\n# Parameters:\n##", 0);
  parameters.print_contents(std::cout, true);
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

  int MPI_rank, MPI_size;
  initialize_MPI(argc, argv, MPI_rank, MPI_size);

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
      MPI_buffer_size;
  double copy_factor;

  read_parameters(paramfile_name, box, reemission_probability, ncell,
                  num_subgrid, num_photon, number_of_iterations,
                  queue_size_per_thread, memoryspace_size, number_of_tasks,
                  MPI_buffer_size, copy_factor);

  //////////////////////////

  ////////////////////////////////
  // Set up task based structures
  ///////////////////////////////

  // set up the queues used to queue tasks
  std::vector< NewQueue * > new_queues(num_threads, nullptr);
  for (int i = 0; i < num_threads; ++i) {
    new_queues[i] = new NewQueue(queue_size_per_thread);
  }

  // set up the task space used to store tasks
  ThreadSafeVector< Task > tasks(number_of_tasks);

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

  char *MPI_buffer = new char[MPI_buffer_size];

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
  std::vector< SubGrid * > gridvec(tot_num_subgrid, nullptr);

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
              DensitySubGrid &this_grid =
                  *static_cast< DensitySubGrid * >(gridvec[index]);
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

  } // end MPI_rank == 0

  //////////////////

  ////////////////////////////////////////////
  // Initialize the photon source information
  ///////////////////////////////////////////

  // Get the index of the (one) subgrid that contains the source position
  const unsigned int source_indices[3] = {
      (unsigned int)((-box[0] / box[3]) * num_subgrid[0]),
      (unsigned int)((-box[1] / box[4]) * num_subgrid[1]),
      (unsigned int)((-box[2] / box[5]) * num_subgrid[2])};

  ///////////////////////////////////////////

  //////////////////////////////
  // Initialize the cost vector
  /////////////////////////////

  std::vector< unsigned long > initial_cost_vector(tot_num_subgrid, 0);
  std::ifstream initial_costs("costs_00.txt");
  if (initial_costs.good()) {
    // use cost information from a previous run as initial guess for the cost
    // skip the initial comment line
    std::string line;
    std::getline(initial_costs, line);
    unsigned int index;
    unsigned long cost;
    int rank, thread;
    while (std::getline(initial_costs, line)) {
      std::istringstream lstream(line);
      lstream >> index >> cost >> rank >> thread;
      if (index < tot_num_subgrid) {
        initial_cost_vector[index] += cost;
      }
    }
  } else {
    // no initial cost information: assume a uniform cost
    for (unsigned int i = 0; i < tot_num_subgrid; ++i) {
      initial_cost_vector[i] = 1;
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
  // note that we will need to communicate the originals vector somehow
  std::vector< unsigned int > originals;
  std::vector< unsigned int > copies(tot_num_subgrid, 0xffffffff);
  std::vector< unsigned char > levels(tot_num_subgrid, 0);
  if (MPI_rank == 0) {

    // get the average cost per thread
    unsigned long avg_cost_per_thread = 0;
    for (unsigned int i = 0; i < tot_num_subgrid; ++i) {
      avg_cost_per_thread += initial_cost_vector[i];
    }
    avg_cost_per_thread /= num_threads;
    // now set the levels accordingly
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

    create_copies(gridvec, levels, new_buffers, originals, copies);

    // we keep this code for later debugging when we have more complex copy
    // hiearchies
    //  std::ofstream ngbfile("initial_ngbs.txt");
    //  for(unsigned int i = 0; i < gridvec.size(); ++i){
    //    ngbfile << i << "\n";
    //    for(int j = 0; j < 27; ++j){
    //      ngbfile << gridvec[i]->get_neighbour(j) << "\n";
    //    }
    //    ngbfile << "\n";
    //  }
    //  ngbfile.close();
  }

  // communicate the new size of the grid to all processes and make sure the
  // local gridvec is up to date (all processes other than rank 0 still have a
  // completely empty gridvec)
  unsigned int new_size = gridvec.size();
  MPI_Bcast(&new_size, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
  gridvec.resize(new_size, nullptr);

  // initialize the actual cost vector
  costs.reset(gridvec.size());

  // no initial cost information: assume a uniform cost
  for (unsigned int i = 0; i < tot_num_subgrid; ++i) {
    costs.add_cost(i, initial_cost_vector[i]);
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
          costs.get_cost(igrid) / (this_copies.size() + 1);
      costs.set_cost(igrid, cost);
      for (unsigned int i = 0; i < this_copies.size(); ++i) {
        costs.set_cost(this_copies[i], cost);
      }
    }
  }

  costs.redistribute();

  // now it is time to move the subgrids to the process where they belong

  if (MPI_rank == 0) {
    unsigned int buffer_position = 0;
    for (int irank = 1; irank < MPI_size; ++irank) {
      unsigned int rank_size = 0;
      for (unsigned int igrid = 0; igrid < gridvec.size(); ++igrid) {
        if (costs.get_process(igrid) == irank) {
          gridvec[igrid]->pack(&MPI_buffer[buffer_position + rank_size],
                               MPI_buffer_size);
          rank_size += gridvec[igrid]->get_MPI_size();
        }
      }
      MPI_Send(&rank_size, 1, MPI_UNSIGNED, irank, 0, MPI_COMM_WORLD);
      MPI_Send(&MPI_buffer[buffer_position], rank_size, MPI_PACKED, irank, 1,
               MPI_COMM_WORLD);
      buffer_position += rank_size;
    }
  } else {
    unsigned int rank_size;
    MPI_Recv(&rank_size, 1, MPI_UNSIGNED, 0, 0, MPI_COMM_WORLD,
             MPI_STATUS_IGNORE);
    MPI_Recv(MPI_buffer, rank_size, MPI_PACKED, 0, 1, MPI_COMM_WORLD,
             MPI_STATUS_IGNORE);
    // we need to somehow figure out which subgrid belongs to which index...
  }

  ////////////////////////////////////////////////

  ////////////////////
  // Actual main loop
  ///////////////////

  // now for the main loop. This loop
  //  - shoots num_photon photons through the grid to get intensity estimates
  //  - computes the ionization equilibrium
  for (unsigned int iloop = 0; iloop < number_of_iterations; ++iloop) {

    // make a global list of all subgrids that contain the (single) photon
    // source position, we will distribute the initial propagation tasks
    // evenly across these subgrids
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

    // get the corresponding thread ranks
    std::vector< int > central_queue(central_index.size());
    for (unsigned int i = 0; i < central_index.size(); ++i) {
      central_queue[i] = costs.get_thread(central_index[i]);
    }

    // STEP 0: log output
    logmessage("Loop " << iloop + 1, 0);

    // make sure all copies have the same neutral fraction as their original
    // subgrid
    for (unsigned int i = 0; i < originals.size(); ++i) {
      const unsigned int original = originals[i];
      const unsigned int copy = tot_num_subgrid + i;
      logmessage("Updating neutral fractions for " << copy << ", copy of "
                                                   << original,
                 1);
      static_cast< DensitySubGrid * >(gridvec[copy])
          ->update_neutral_fractions(
              *static_cast< DensitySubGrid * >(gridvec[original]));
    }

    // STEP 1: photon shooting
    logmessage("Starting photon shoot loop", 0);
    // GLOBAL control variables (these are shared and updated atomically):
    //  - number of photon packets that has been created at the source
    unsigned int num_photon_sourced = 0;
    //  - number of photon packets that has left the system, either through
    //    absorption or by crossing a simulation box wall
    unsigned int num_photon_done = 0;
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
      // set up the initial source photon task
      // this task will respawn itself as long as
      //   num_photon_sourced < num_photon
      // (this automatically means that threads with less propagation tasks will
      //  handle more source tasks)
      {
        const size_t task_index = tasks.get_free_element_safe();
        myassert(task_index < tasks.max_size(), "Task buffer overflow!");
        tasks[task_index]._type = TASKTYPE_SOURCE_PHOTON;
        // buffer is ready to be processed: add to the queue
        new_queues[thread_id]->add_task(task_index);
      }
      // this loop is repeated until all photons have been propagated
      // note that this condition automatically covers the condition
      //  num_photon_sourced < num_photon
      // as unsourced photons cannot contribute to num_photon_done
      while (num_photon_done < num_photon) {

        // get a first task
        // upon first entry of the while loop, this will be the photon source
        // task we just created
        unsigned int current_index = new_queues[thread_id]->get_task();

        // task activation: if no task is found, try to launch a photon buffer
        // that is not yet full and prematurely schedule it
        if (current_index == NO_TASK) {
          // try to activate a non-full buffer
          // note that we only try to access thread-local information, so as
          // long as we don't allow task stealing, this will be thread-safe
          unsigned int i = 0;
          // loop over all subgrids
          while (i < gridvec.size() && current_index == NO_TASK) {
            // we only activate subgrids that belong to this thread to make
            // sure we don't create conflicts
            // Note that this could mean we prematurely activate tasks for
            // another thread.
            if (costs.get_thread(i) == thread_id) {
              int j = 0;
              // loop over all buffers of this subgrid
              while (j < 27 && current_index == NO_TASK) {
                // only process existing buffers that are non empty
                if (gridvec[i]->get_neighbour(j) != NEIGHBOUR_OUTSIDE &&
                    new_buffers[gridvec[i]->get_active_buffer(j)]._actual_size >
                        0) {
                  // found one! Prematurely activate this subgrid.
                  const unsigned int non_full_index =
                      gridvec[i]->get_active_buffer(j);
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
                  // Add the buffer to the queue of the corresponding thread.
                  // Note that this could be another thread than this thread, in
                  // which case this thread will still be hungry (and we might
                  // be over-feeding the other thread).
                  // However, this is the only mechanism through which we can
                  // feed hungry threads with food from this thread, as we do
                  // not allow threads to feed themselves with food that does
                  // not belong to them.
                  const size_t task_index = tasks.get_free_element();
                  if (j > 0) {
                    tasks[task_index]._type = TASKTYPE_PHOTON_TRAVERSAL;
                  } else {
                    tasks[task_index]._type = TASKTYPE_PHOTON_REEMIT;
                  }
                  tasks[task_index]._cell =
                      new_buffers[non_full_index]._sub_grid_index;
                  tasks[task_index]._buffer = non_full_index;
                  // note that this statement should be last, as the task might
                  // be executed as soon as this statement is executed
                  new_queues[costs.get_thread(
                                 new_buffers[non_full_index]._sub_grid_index)]
                      ->add_task(task_index);

                  // Try again to get a task. This could be the task we just
                  // created, a task that was added by another thread while we
                  // were doing the task activation, or still NO_TASK, in which
                  // case we continue waking up buffers.
                  current_index = new_queues[thread_id]->get_task();
                }
                ++j;
              }
            }
            ++i;
          }
        }

        // Keep processing tasks until the queue is empty.
        while (current_index != NO_TASK) {

          Task &task = tasks[current_index];

          // Different tasks are processed in different ways.
          if (task._type == TASKTYPE_SOURCE_PHOTON) {

            /// generate random photon packets from the source

            // log the start time of the task (if task output is enabled)
            task.start(thread_id);

            // check if we should still execute this task (as another thread
            // could have depleted the photon source by now)
            if (num_photon_sourced < num_photon) {

              // atomically increment the number of photons that was sourced
              // if this works, then this thread gets the unique right and
              // obligation to generate the next batch of photons
              const unsigned int num_photon_sourced_now =
                  atomic_post_add(num_photon_sourced, PHOTONBUFFER_SIZE);

              // the statement above could have been executed by multiple
              // threads simultaneously, so we need to check that it worked for
              // this particular thread before we actually continue.
              if (num_photon_sourced_now < num_photon) {

                // OK, we can (/have to) generate some photons!

                // Spawn a new source photon task for this thread and add it to
                // the end of its queue.
                // This guarantess we cannot exit the task loop as long as the
                // source still has photon packets.
                {
                  const size_t task_index = tasks.get_free_element_safe();
                  myassert(task_index < tasks.max_size(),
                           "Task buffer overflow!");
                  tasks[task_index]._type = TASKTYPE_SOURCE_PHOTON;
                  // buffer is ready to be processed: add to the queue
                  // note that in this case, the statement order does not really
                  // matter, as this task can only be executed by the same
                  // thread that executes the statement
                  new_queues[thread_id]->add_task(task_index);
                }

                // if this is the last buffer: cap the total number of photons
                // to the requested value
                // (note that num_photon_done_now is the number of photons that
                //  was generated BEFORE this task, after this task,
                //  num_photon_done_now + PHOTONBUFFER_SIZE photons will have
                //  been generated, unless this number is capped)
                unsigned int num_photon_this_loop = PHOTONBUFFER_SIZE;
                if (num_photon_sourced_now + PHOTONBUFFER_SIZE > num_photon) {
                  num_photon_this_loop += (num_photon - num_photon_sourced_now);
                }

                // get a free photon buffer in the central queue
                unsigned int buffer_index = new_buffers.get_free_buffer();
                PhotonBuffer &input_buffer = new_buffers[buffer_index];
                // assign the buffer to a random thread that has a copy of the
                // subgrid that contains the source position. This should ensure
                // a balanced load for these threads.
                unsigned int which_central_index =
                    random_generator[thread_id].get_uniform_random_double() *
                    central_index.size();
                myassert(which_central_index >= 0 &&
                             which_central_index < central_index.size(),
                         "Invalid source subgrid thread index!");
                unsigned int this_central_index =
                    central_index[which_central_index];

                // now actually fill the buffer with random photon packets
                fill_buffer(input_buffer, num_photon_this_loop,
                            random_generator[thread_id], this_central_index);

                // add to the queue of the corresponding thread
                const size_t task_index = tasks.get_free_element_safe();
                myassert(task_index < tasks.max_size(),
                         "Task buffer overflow!");
                tasks[task_index]._type = TASKTYPE_PHOTON_TRAVERSAL;
                tasks[task_index]._cell = this_central_index;
                tasks[task_index]._buffer = buffer_index;
                // note that this statement should be last, as the buffer might
                // be processed as soon as this statement is executed
                new_queues[central_queue[which_central_index]]->add_task(
                    task_index);

              } // if (num_photon_sourced_now < num_photon)

            } // if (num_photon_sourced < num_photon)

            // log the end time of the task
            task.stop();

          } else if (task._type == TASKTYPE_PHOTON_TRAVERSAL) {

            /// propagate photon packets from a buffer through a subgrid

            // variables used to determine the cost of photon traversal tasks
            unsigned long task_start, task_end;
            task_tick(task_start);

            // log the start of the task
            task.start(thread_id);

            const unsigned int current_buffer_index = task._buffer;
            PhotonBuffer &buffer = new_buffers[current_buffer_index];
            const unsigned int igrid = buffer._sub_grid_index;
            DensitySubGrid &this_grid = *static_cast< DensitySubGrid * >(
                gridvec[buffer._sub_grid_index]);

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

            // now do the actual photon traversal
            do_photon_traversal(buffer, this_grid, local_buffers,
                                local_buffer_flags);

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
                unsigned int add_index =
                    new_buffers.add_photons(new_index, local_buffers[i]);

                // check if the original buffer is full
                if (add_index != new_index) {

                  // YES: create a task for the buffer and add it to the queue
                  const size_t task_index = tasks.get_free_element_safe();
                  myassert(task_index < tasks.max_size(),
                           "Task buffer overflow!");
                  // the task type depends on the buffer: photon packets in the
                  // internal buffer were absorbed and could be reemitted,
                  // photon packets in the other buffers left the subgrid and
                  // need to be traversed in the neighbouring subgrid
                  if (i > 0) {
                    tasks[task_index]._type = TASKTYPE_PHOTON_TRAVERSAL;
                  } else {
                    tasks[task_index]._type = TASKTYPE_PHOTON_REEMIT;
                  }
                  tasks[task_index]._cell =
                      new_buffers[new_index]._sub_grid_index;
                  tasks[task_index]._buffer = new_index;
                  // add the task to the queue of the corresponding thread
                  new_queues[costs.get_thread(ngb)]->add_task(task_index);
                  myassert(new_buffers[add_index]._sub_grid_index == ngb,
                           "Wrong subgrid");
                  myassert(new_buffers[add_index]._direction ==
                               output_to_input_direction(i),
                           "Wrong direction");
                  // new_buffers.add_photons already created a new empty
                  // buffer, set it as the active buffer for this output
                  // direction
                  this_grid.set_active_buffer(i, add_index);

                } // if (add_index != new_index)

              } // if (local_buffer_flags[i] &&
                //     local_buffers[i]._actual_size > 0)

            } // for (int i = 26; i >= 0; --i)

            // add photons that were absorbed (if reemission was disabled) or
            // that left the system to the global count
            atomic_pre_add(num_photon_done, num_photon_done_now);

            // delete the original buffer, as we are done with it
            new_buffers.free_buffer(current_buffer_index);

            // log the end time of the task
            task.stop();

            // update the cost computation for this subgrid
            task_tick(task_end);
            costs.add_cost(igrid, task_end - task_start);

          } else if (task._type == TASKTYPE_PHOTON_REEMIT) {

            /// reemit absorbed photon packets

            // variables used to determine the cost of photon traversal tasks
            unsigned long task_start, task_end;
            task_tick(task_start);

            // log the start of the task
            task.start(thread_id);

            // get the buffer
            const unsigned int current_buffer_index = task._buffer;
            PhotonBuffer &buffer = new_buffers[current_buffer_index];

            // keep track of the original number of photons in the buffer
            unsigned int num_photon_done_now = buffer._actual_size;

            // reemit photon packets
            do_reemission(buffer, random_generator[thread_id],
                          reemission_probability);

            // find the number of photon packets that was absorbed and not
            // reemitted...
            num_photon_done_now -= buffer._actual_size;
            // ...and add it to the global count
            atomic_pre_add(num_photon_done, num_photon_done_now);

            // the reemitted photon packets are ready to be propagated: create
            // a new propagation task
            const size_t task_index = tasks.get_free_element_safe();
            myassert(task_index < tasks.max_size(), "Task buffer overflow!");
            tasks[task_index]._type = TASKTYPE_PHOTON_TRAVERSAL;
            tasks[task_index]._cell = task._cell;
            tasks[task_index]._buffer = current_buffer_index;
            // add it to the local queue
            new_queues[thread_id]->add_task(task_index);

            // log the end time of the task
            task.stop();

            // update the cost computation for this subgrid
            task_tick(task_end);
            costs.add_cost(task._cell, task_end - task_start);

          } else {

            // should never happen
            logmessage("Unknown task!", 0);
          }

          // We finished as task: try to get a new task from the local queue
          current_index = new_queues[thread_id]->get_task();

          // this would be the right place to delete the task (if we don't want
          // to output it)

        } // while (current_index != NO_TASK)

      } // while (num_photon_done < num_photon)

    } // parallel region

    // some useful log output to help us determine a good value for the queue
    // and task space sizes
    logmessage("Total number of tasks: " << tasks.size(), 0);

    // combine the counter values for subgrids with copies
    for (unsigned int i = 0; i < originals.size(); ++i) {
      const unsigned int original = originals[i];
      const unsigned int copy = tot_num_subgrid + i;
      logmessage("Updating ionization integrals for " << original
                                                      << " using copy " << copy,
                 1);
      static_cast< DensitySubGrid * >(gridvec[original])
          ->update_intensities(*static_cast< DensitySubGrid * >(gridvec[copy]));
    }

    // STEP 2: update the ionization structure for each subgrid
    logmessage("Updating ionisation structure", 0);
    unsigned int igrid = 0;
#pragma omp parallel default(shared)
    {
      while (igrid < tot_num_subgrid) {
        const unsigned int current_igrid = atomic_post_increment(igrid);
        if (current_igrid < tot_num_subgrid) {
          static_cast< DensitySubGrid * >(gridvec[current_igrid])
              ->compute_neutral_fraction(num_photon);
        }
      }
    }

    // output useful information about this iteration (if enabled)
    logmessage("Writing task and cost information", 0);
    output_tasks(iloop, tasks);
    output_costs(iloop, tot_num_subgrid, num_threads, costs, copies, originals);

    // clear task buffer
    tasks.clear();

    // rebalance the work based on the cost information during the last
    // iteration
    logmessage("Rebalancing", 0);
    // fill initial_cost_vector with costs
    unsigned long avg_cost_per_thread = 0;
    for (unsigned int i = 0; i < tot_num_subgrid; ++i) {
      initial_cost_vector[i] = costs.get_cost(i);
      avg_cost_per_thread += initial_cost_vector[i];
    }
    for (unsigned int i = 0; i < originals.size(); ++i) {
      initial_cost_vector[originals[i]] += costs.get_cost(tot_num_subgrid + i);
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
        costs.set_cost(igrid, cost);
        for (unsigned int i = 0; i < this_copies.size(); ++i) {
          costs.set_cost(this_copies[i], cost);
        }
      } else {
        costs.set_cost(igrid, initial_cost_vector[igrid]);
      }
    }

    // redistribute the subgrids among the threads to balance the computational
    // costs (based on this iteration)
    costs.redistribute();
  } // main loop

  ///////////////////

  //////////////////////////////////////////////////////////////////////////////

  //////////////////////////////////////////////////////////////////////////////
  /// Clean up
  //////////////////////////////////////////////////////////////////////////////

  ///////////////////////
  // Output final result
  //////////////////////

  //  - ASCII output (for the VisIt plot script)
  std::ofstream ofile("intensities.txt");
  for (unsigned int igrid = 0; igrid < tot_num_subgrid; ++igrid) {
    static_cast< DensitySubGrid * >(gridvec[igrid])->print_intensities(ofile);
  }
  ofile.close();

  //  - binary output (for the Python plot script)
  std::ofstream bfile("intensities.dat");
  for (unsigned int igrid = 0; igrid < tot_num_subgrid; ++igrid) {
    static_cast< DensitySubGrid * >(gridvec[igrid])->output_intensities(bfile);
  }
  bfile.close();

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
