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
 * @file testDensitySubGrid_notasks.cpp
 *
 * @brief testDensitySubGrid version without task-based parallelism.
 *
 * @author Bert Vandenbroucke (bv7@st-andrews.ac.uk)
 */

/*! @brief We define this to make sure cell locking is enabled. */
#define SUBGRID_CELL_LOCK

#include "Atomic.hpp"
#include "DensitySubGrid.hpp"
#include "RandomGenerator.hpp"
#include "Timer.hpp"

#include <cmath>
#include <fstream>
#include <omp.h>
#include <sstream>
#include <sys/resource.h>

/*! @brief Output log level. The higher the value, the more stuff is printed to
 *  the stderr. Comment to disable logging altogether. */
#define LOG_OUTPUT 1

/*! @brief Activate this to unit test the directional algorithms. */
//#define TEST_DIRECTIONS

/*! @brief Activate this to unit test the CostVector. */
//#define TEST_COSTVECTOR

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
 * @brief testDensitySubGrid version without task-based parallelism.
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

  const double box[6] = {-1.543e17, -1.543e17, -1.543e17,
                         3.086e17,  3.086e17,  3.086e17};
  const int ncell[3] = {60, 60, 60};
  DensitySubGrid grid(box, ncell);
  const unsigned int num_photon = 1000000;

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

  Timer program_timer;
  program_timer.start();

  // set up the random number generators
  std::vector< RandomGenerator > random_generator(num_threads);
  for (int i = 0; i < num_threads; ++i) {
    // make sure every thread on every process has a different seed
    random_generator[i].set_seed(42 + i);
  }

  for (unsigned int iloop = 0; iloop < 10; ++iloop) {
    logmessage("Loop " << iloop + 1, 0);

    unsigned int iglobal = 0;
#pragma omp parallel default(shared)
    {
      // id of this specific thread
      const int thread_id = omp_get_thread_num();
      unsigned int i = atomic_post_increment(iglobal);
      while (i < num_photon) {
        Photon photon;
        photon._position[0] = 0.;
        photon._position[1] = 0.;
        photon._position[2] = 0.;
        const double cost =
            2. * random_generator[thread_id].get_uniform_random_double() - 1.;
        const double sint = std::sqrt(std::max(1. - cost * cost, 0.));
        const double phi =
            2. * M_PI * random_generator[thread_id].get_uniform_random_double();
        const double cosp = std::cos(phi);
        const double sinp = std::sin(phi);
        photon._direction[0] = sint * cosp;
        photon._direction[1] = sint * sinp;
        photon._direction[2] = cost;
        photon._inverse_direction[0] = 1. / photon._direction[0];
        photon._inverse_direction[1] = 1. / photon._direction[1];
        photon._inverse_direction[2] = 1. / photon._direction[2];
        photon._weight = 1.;
        photon._current_optical_depth = 0.;
        photon._target_optical_depth =
            -std::log(random_generator[thread_id].get_uniform_random_double());
        photon._photoionization_cross_section = 6.3e-22;
        grid.interact(photon, TRAVELDIRECTION_INSIDE);
        i = atomic_post_increment(iglobal);
      }
    }
    grid.compute_neutral_fraction(num_photon);
  }

  program_timer.stop();
  logmessage("Total program time: " << program_timer.value() << " s.", 0);

  struct rusage resource_usage;
  getrusage(RUSAGE_SELF, &resource_usage);
  size_t max_memory = static_cast< size_t >(resource_usage.ru_maxrss) *
                      static_cast< size_t >(1024);
  logmessage("Maximum memory usage: " << human_readable_bytes(max_memory), 0);

  std::ofstream bfile("intensities.dat");
  grid.output_intensities(bfile);
  bfile.close();

  return 0;
}
