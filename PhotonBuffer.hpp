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
 * @file PhotonBuffer.hpp
 *
 * @brief Photon buffer.
 *
 * @author Bert Vandenbroucke (bv7@st-andrews.ac.uk)
 */

#ifndef PHOTONBUFFER_HPP
#define PHOTONBUFFER_HPP

// project includes
#include "Assert.hpp"
#include "Atomic.hpp"
#include "Photon.hpp"

// standard library includes
#include <mpi.h>

/*! @brief Number of photons that can be stored in a single buffer. */
#define PHOTONBUFFER_SIZE 500u

/*! @brief Size of the MPI buffer necessary to store a PhotonBuffer. */
#define PHOTONBUFFER_MPI_SIZE                                                  \
  (2 * sizeof(unsigned int) + sizeof(int) + PHOTONBUFFER_SIZE * PHOTON_MPI_SIZE)

/**
 * @brief Check if the given PhotonBuffers are equal.
 *
 * @param a First PhotonBuffer.
 * @param b Second PhotonBuffer.
 */
#define photonbuffer_check_equal(a, b)                                         \
  myassert(a._sub_grid_index == b._sub_grid_index,                             \
           "Subgrid indices do not match!");                                   \
  myassert(a._direction == b._direction, "Directions do not match!");          \
  myassert(a._actual_size == b._actual_size, "Sizes do not match!");           \
  for (unsigned int i = 0; i < PHOTONBUFFER_SIZE; ++i) {                       \
    photon_check_equal(a._photons[i], b._photons[i]);                          \
  }

/**
 * @brief Photon buffer.
 *
 * All members are public for now.
 */
class PhotonBuffer {
public:
  /*! @brief Subgrid with which this buffer is associated. */
  unsigned int _sub_grid_index;

  /*! @brief TravelDirection of the photons in the buffer. */
  int _direction;

  /*! @brief Number of photons in the buffer. */
  unsigned int _actual_size;

  /*! @brief Actual photon buffer. */
  Photon _photons[PHOTONBUFFER_SIZE];

  /**
   * @brief Store the contents of the PhotonBuffer in the given MPI
   * communication buffer.
   *
   * @param buffer Buffer to use (should be preallocated and have at least size
   * PHOTONBUFFER_MPI_SIZE).
   */
  inline void pack(char buffer[PHOTONBUFFER_MPI_SIZE]) {
    int buffer_position = 0;
    MPI_Pack(&_sub_grid_index, 1, MPI_UNSIGNED, buffer, PHOTONBUFFER_MPI_SIZE,
             &buffer_position, MPI_COMM_WORLD);
    MPI_Pack(&_direction, 1, MPI_INT, buffer, PHOTONBUFFER_MPI_SIZE,
             &buffer_position, MPI_COMM_WORLD);
    MPI_Pack(&_actual_size, 1, MPI_UNSIGNED, buffer, PHOTONBUFFER_MPI_SIZE,
             &buffer_position, MPI_COMM_WORLD);
    for (unsigned int i = 0; i < PHOTONBUFFER_SIZE; ++i) {
      _photons[i].pack(&buffer[buffer_position]);
      buffer_position += PHOTON_MPI_SIZE;
    }
  }

  /**
   * @brief Copy the contents of the given MPI communication buffer into this
   * PhotonBuffer.
   *
   * @param buffer MPI communication buffer (should have at least size
   * PHOTONBUFFER_MPI_SIZE).
   */
  inline void unpack(char buffer[PHOTONBUFFER_MPI_SIZE]) {
    int buffer_position = 0;
    MPI_Unpack(buffer, PHOTONBUFFER_MPI_SIZE, &buffer_position,
               &_sub_grid_index, 1, MPI_UNSIGNED, MPI_COMM_WORLD);
    MPI_Unpack(buffer, PHOTONBUFFER_MPI_SIZE, &buffer_position, &_direction, 1,
               MPI_INT, MPI_COMM_WORLD);
    MPI_Unpack(buffer, PHOTONBUFFER_MPI_SIZE, &buffer_position, &_actual_size,
               1, MPI_UNSIGNED, MPI_COMM_WORLD);
    for (unsigned int i = 0; i < PHOTONBUFFER_SIZE; ++i) {
      _photons[i].unpack(&buffer[buffer_position]);
      buffer_position += PHOTON_MPI_SIZE;
    }
  }
};

#endif // PHOTONBUFFER_HPP
