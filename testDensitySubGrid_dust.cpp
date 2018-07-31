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

/*! @brief Activate this to add direct photons from the source to the image. */
//#define ADD_DIRECT_LIGHT

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

// global variables, as we need them in the log macro
int MPI_rank, MPI_size;

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

  initialize_MPI(argc, argv, MPI_rank, MPI_size);

  int num_threads_request = 1000;
  if (argc > 1) {
    num_threads_request = atoi(argv[1]);
  }
  int num_threads;
  set_number_of_threads(num_threads_request, num_threads);

  const double box[6] = {-1., -1., -1., 2., 2., 2.};
  const int ncell[3] = {64, 64, 64};
  DensitySubGrid grid(box, ncell);
  const unsigned int num_photon = 1e8;

  for (int icell = 0; icell < ncell[0] * ncell[1] * ncell[2]; ++icell) {
    double cell_midpoint[3];
    grid.get_cell_midpoint(icell, cell_midpoint);
    const double cell_radius = std::sqrt(cell_midpoint[0] * cell_midpoint[0] +
                                         cell_midpoint[1] * cell_midpoint[1] +
                                         cell_midpoint[2] * cell_midpoint[2]);

    if (cell_radius < 1.) {
      grid.set_number_density(icell, 1.);
      grid.set_neutral_fraction(icell, 10.);
    } else {
      grid.set_number_density(icell, 0.);
      grid.set_neutral_fraction(icell, 10.);
    }
  }

  Timer program_timer;
  program_timer.start();

  // set up the random number generators
  std::vector< RandomGenerator > random_generator(num_threads);
  for (int i = 0; i < num_threads; ++i) {
    // make sure every thread on every process has a different seed
    random_generator[i].set_seed(42 + i);
  }

  double image[100][100];
  for (int ix = 0; ix < 100; ++ix) {
    for (int iy = 0; iy < 100; ++iy) {
      image[ix][iy] = 0.;
    }
  }

  Atomic< unsigned int > iglobal(0);
#ifdef ADD_DIRECT_LIGHT
  double source_photon_weight;
  // compute source photon weight
  // this is the optical extinction for a photon that starts from the source
  // and is scattered towards the observer
  // this weight is used to add direct emission to the image for every photon
  // packet. It is the same for all source photons.
  {
    Photon photon;
    photon.set_position(0., 0., 0.);
    photon.set_direction(0., 0., 1.);
    grid.compute_optical_depth(photon, TRAVELDIRECTION_INSIDE);
    source_photon_weight = std::exp(-photon.get_target_optical_depth());
  }
#endif
#pragma omp parallel default(shared)
  {
    // id of this specific thread
    const int thread_id = omp_get_thread_num();

    double local_image[100][100];
    for (int ix = 0; ix < 100; ++ix) {
      for (int iy = 0; iy < 100; ++iy) {
        local_image[ix][iy] = 0.;
      }
    }

    unsigned int i = iglobal.post_increment();
    while (i < num_photon) {

      Photon photon;

      // initial position: we currently assume a single source at the origin
      photon.set_position(0., 0., 0.);
// note that we do not include the normalization factor for the assumed
// isotropic emission phase function; it contributes a constant factor
// to our pixels
#ifdef ADD_DIRECT_LIGHT
      local_image[50][50] += source_photon_weight;
#endif

      // initial direction: isotropic distribution
      get_random_direction(random_generator[thread_id], photon.get_direction());

      // target optical depth (exponential distribution)
      photon.set_target_optical_depth(
          -std::log(random_generator[thread_id].get_uniform_random_double()));

      // this is the fixed cross section we use for the moment
      photon.set_photoionization_cross_section(1.);

      if (grid.propagate(photon, TRAVELDIRECTION_INSIDE) ==
          TRAVELDIRECTION_INSIDE) {
        // if the photon packet is still inside, we add its contribution to the
        // image using forced scattering
        photon.set_direction(0., 0., 1.);
        grid.compute_optical_depth(photon, TRAVELDIRECTION_INSIDE);
        const double *position = photon.get_position();
        const unsigned int ix = 0.5 * (position[0] + 1.) * 100;
        const unsigned int iy = 0.5 * (position[1] + 1.) * 100;
        // again, we omit the normalization factor for isotropic scattering
        local_image[ix][iy] += std::exp(-photon.get_target_optical_depth());
      }

      i = iglobal.post_increment();
    }

#pragma omp critical
    {
      // add image contribution from this thread
      for (int ix = 0; ix < 100; ++ix) {
        for (int iy = 0; iy < 100; ++iy) {
          image[ix][iy] += local_image[ix][iy];
        }
      }
    }
  }

  std::ofstream ifile("image.dat");
  for (int ix = 0; ix < 100; ++ix) {
    for (int iy = 0; iy < 100; ++iy) {
      ifile.write(reinterpret_cast< char * >(&image[ix][iy]), sizeof(double));
    }
  }
  ifile.close();

  program_timer.stop();
  logmessage("Total program time: " << program_timer.value() << " s.", 0);

  struct rusage resource_usage;
  getrusage(RUSAGE_SELF, &resource_usage);
  size_t max_memory = static_cast< size_t >(resource_usage.ru_maxrss) *
                      static_cast< size_t >(1024);
  logmessage("Maximum memory usage: " << human_readable_bytes(max_memory), 0);

  return MPI_Finalize();
}
