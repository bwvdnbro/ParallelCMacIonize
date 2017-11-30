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
#include "Atomic.hpp"
#include "Photon.hpp"

/*! @brief Number of photons that can be stored in a single buffer. */
#define PHOTONBUFFER_SIZE 1000u

/**
 * @brief Photon buffer.
 *
 * All members are public for now.
 */
class PhotonBuffer {
public:
  /*! @brief Atomic lock. */
  bool _lock;

  /*! @brief Subgrid with which this buffer is associated. */
  unsigned int _sub_grid_index;

  /*! @brief TravelDirection of the photons in the buffer. */
  int _direction;

  /*! @brief Number of photons in the buffer. */
  unsigned int _actual_size;

  /*! @brief Actual photon buffer. */
  Photon _photons[PHOTONBUFFER_SIZE];

  /*! @brief Flag indicating if this buffer is in use. */
  bool _is_in_use;

  inline void lock() {
    while (!atomic_lock(_lock)) {
    }
  }

  inline bool try_lock() { return atomic_lock(_lock); }

  inline void unlock() { atomic_unlock(_lock); }
};

#endif // PHOTONBUFFER_HPP
