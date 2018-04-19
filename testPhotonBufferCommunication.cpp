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
 * @file testPhotonBufferCommunication.cpp
 *
 * @brief Unit test for PhotonBuffer MPI communication.
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
#include "Log.hpp"
#include "PhotonBuffer.hpp"
#include "RandomGenerator.hpp"
#include "TravelDirections.hpp"

#include <mpi.h>

/**
 * @brief Unit test for PhotonBuffer MPI communication.
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

  // make sure we have at least 2 processes
  myassert(MPI_size > 1, "Not running in MPI mode!");

  // set up a random PhotonBuffer
  PhotonBuffer test_buffer;

  // fill the test buffer with random content
  RandomGenerator random_generator(42);
  test_buffer.grow(random_generator.get_random_integer() % PHOTONBUFFER_SIZE);
  test_buffer.set_subgrid_index(random_generator.get_random_integer());
  test_buffer.set_direction(random_generator.get_random_integer() %
                            TRAVELDIRECTION_NUMBER);
  for (unsigned int i = 0; i < test_buffer.size(); ++i) {
    Photon &photon = test_buffer[i];
    photon._position[0] = random_generator.get_uniform_random_double();
    photon._position[1] = random_generator.get_uniform_random_double();
    photon._position[2] = random_generator.get_uniform_random_double();
    photon._direction[0] = random_generator.get_uniform_random_double();
    photon._direction[1] = random_generator.get_uniform_random_double();
    photon._direction[2] = random_generator.get_uniform_random_double();
    photon._inverse_direction[0] = random_generator.get_uniform_random_double();
    photon._inverse_direction[1] = random_generator.get_uniform_random_double();
    photon._inverse_direction[2] = random_generator.get_uniform_random_double();
    photon._weight = random_generator.get_uniform_random_double();
    photon._current_optical_depth =
        random_generator.get_uniform_random_double();
    photon._target_optical_depth = random_generator.get_uniform_random_double();
    photon._photoionization_cross_section =
        random_generator.get_uniform_random_double();
  }

  // now communicate:
  //  - rank 0 sends the buffer
  //  - rank 1 receives and checks if the buffer is what it should be
  char MPI_buffer[PHOTONBUFFER_MPI_SIZE];
  if (MPI_rank == 0) {
    // pack...
    test_buffer.pack(MPI_buffer);
    // ...and send
    MPI_Send(MPI_buffer, PHOTONBUFFER_MPI_SIZE, MPI_PACKED, 1, 101010,
             MPI_COMM_WORLD);
  } else if (MPI_rank == 1) {
    // receive...
    MPI_Status status;
    MPI_Recv(MPI_buffer, PHOTONBUFFER_MPI_SIZE, MPI_PACKED, 0, 101010,
             MPI_COMM_WORLD, &status);
    // ...and unpack
    PhotonBuffer recv_buffer;
    recv_buffer.unpack(MPI_buffer);

    // check if the result is what it should be
    test_buffer.check_equal(recv_buffer);
  } // other ranks do nothing

  if (MPI_rank == 0) {
    logmessage("Test successful.", 0);
  }

  return MPI_Finalize();
}
