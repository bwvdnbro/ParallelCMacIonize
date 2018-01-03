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
 * @file Photon.hpp
 *
 * @brief Photon packet.
 *
 * @author Bert Vandenbroucke (bv7@st-andrews.ac.uk)
 */
#ifndef PHOTON_HPP
#define PHOTON_HPP

/*! @brief Size of the MPI buffer necessary to store a single Photon. */
#define PHOTON_MPI_SIZE (13 * sizeof(double))

/**
 * @brief Check if the given Photons are the same.
 *
 * @param a First Photon.
 * @param b Second Photon.
 */
#define photon_check_equal(a, b)                                               \
  myassert(a._position[0] == b._position[0], "Positions do not match!");       \
  myassert(a._position[1] == b._position[1], "Positions do not match!");       \
  myassert(a._position[2] == b._position[2], "Positions do not match!");       \
  myassert(a._direction[0] == b._direction[0], "Directions do not match!");    \
  myassert(a._direction[1] == b._direction[1], "Directions do not match!");    \
  myassert(a._direction[2] == b._direction[2], "Directions do not match!");    \
  myassert(a._inverse_direction[0] == b._inverse_direction[0],                 \
           "Inverse directions do not match!");                                \
  myassert(a._inverse_direction[1] == b._inverse_direction[1],                 \
           "Inverse directions do not match!");                                \
  myassert(a._inverse_direction[2] == b._inverse_direction[2],                 \
           "Inverse directions do not match!");                                \
  myassert(a._current_optical_depth == b._current_optical_depth,               \
           "Current optical depths do not match!");                            \
  myassert(a._target_optical_depth == b._target_optical_depth,                 \
           "Target optical depths do not match!");                             \
  myassert(a._photoionization_cross_section ==                                 \
               b._photoionization_cross_section,                               \
           "Cross sections do not match!");                                    \
  myassert(a._weight == b._weight, "Weights do not match!");

#include <mpi.h>

/**
 * @brief Photon packet.
 *
 * All members are public for now.
 */
class Photon {
public:
  /*! @brief Current position of the photon packet (in m). */
  double _position[3];

  /*! @brief Propagation direction of the photon packet. */
  double _direction[3];

  /*! @brief Inverse of the propagation direction (used to avoid expensive
   *  divisions in the photon packet traversal algorithm). */
  double _inverse_direction[3];

  /*! @brief Current optical depth of the photon packet. */
  double _current_optical_depth;

  /*! @brief Target optical depth for the photon packet. */
  double _target_optical_depth;

  /*! @brief Photoionization cross section of the photons in the photon packet
   *  (in m^2). */
  double _photoionization_cross_section;

  /*! @brief Weight of the photon packet. */
  double _weight;

  /**
   * @brief Store the contents of the Photon in the given MPI communication
   * buffer.
   *
   * @param buffer Buffer to use (should be preallocated and have at least size
   * PHOTON_MPI_SIZE).
   */
  inline void pack(char buffer[PHOTON_MPI_SIZE]) const {
    int buffer_position = 0;
    MPI_Pack(_position, 3, MPI_DOUBLE, buffer, PHOTON_MPI_SIZE,
             &buffer_position, MPI_COMM_WORLD);
    MPI_Pack(_direction, 3, MPI_DOUBLE, buffer, PHOTON_MPI_SIZE,
             &buffer_position, MPI_COMM_WORLD);
    MPI_Pack(_inverse_direction, 3, MPI_DOUBLE, buffer, PHOTON_MPI_SIZE,
             &buffer_position, MPI_COMM_WORLD);
    MPI_Pack(&_current_optical_depth, 1, MPI_DOUBLE, buffer, PHOTON_MPI_SIZE,
             &buffer_position, MPI_COMM_WORLD);
    MPI_Pack(&_target_optical_depth, 1, MPI_DOUBLE, buffer, PHOTON_MPI_SIZE,
             &buffer_position, MPI_COMM_WORLD);
    MPI_Pack(&_photoionization_cross_section, 1, MPI_DOUBLE, buffer,
             PHOTON_MPI_SIZE, &buffer_position, MPI_COMM_WORLD);
    MPI_Pack(&_weight, 1, MPI_DOUBLE, buffer, PHOTON_MPI_SIZE, &buffer_position,
             MPI_COMM_WORLD);
  }

  /**
   * @brief Copy the contents of the given MPI communication buffer into this
   * Photon.
   *
   * @param buffer MPI communication buffer (should have at least size
   * PHOTON_MPI_SIZE).
   */
  inline void unpack(char buffer[PHOTON_MPI_SIZE]) {
    int buffer_position = 0;
    MPI_Unpack(buffer, PHOTON_MPI_SIZE, &buffer_position, _position, 3,
               MPI_DOUBLE, MPI_COMM_WORLD);
    MPI_Unpack(buffer, PHOTON_MPI_SIZE, &buffer_position, _direction, 3,
               MPI_DOUBLE, MPI_COMM_WORLD);
    MPI_Unpack(buffer, PHOTON_MPI_SIZE, &buffer_position, _inverse_direction, 3,
               MPI_DOUBLE, MPI_COMM_WORLD);
    MPI_Unpack(buffer, PHOTON_MPI_SIZE, &buffer_position,
               &_current_optical_depth, 1, MPI_DOUBLE, MPI_COMM_WORLD);
    MPI_Unpack(buffer, PHOTON_MPI_SIZE, &buffer_position,
               &_target_optical_depth, 1, MPI_DOUBLE, MPI_COMM_WORLD);
    MPI_Unpack(buffer, PHOTON_MPI_SIZE, &buffer_position,
               &_photoionization_cross_section, 1, MPI_DOUBLE, MPI_COMM_WORLD);
    MPI_Unpack(buffer, PHOTON_MPI_SIZE, &buffer_position, &_weight, 1,
               MPI_DOUBLE, MPI_COMM_WORLD);
  }
};

#endif // PHOTON_HPP
