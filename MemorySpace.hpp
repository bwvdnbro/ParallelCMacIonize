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
 * @file MemorySpace.hpp
 *
 * @brief Buffer with pre-allocated PhotonBuffers that can be used.
 *
 * @author Bert Vandenbroucke (bv7@st-andrews.ac.uk)
 */
#ifndef MEMORYSPACE_HPP
#define MEMORYSPACE_HPP

#include "Atomic.hpp"
#include "Lock.hpp"
#include "PhotonBuffer.hpp"
#include "ThreadSafeVector.hpp"

/**
 * @brief Buffer with pre-allocated PhotonBuffers that can be used.
 */
class MemorySpace {
private:
  /*! @brief Memory space itself. */
  ThreadSafeVector< PhotonBuffer > _memory_space;

public:
  /**
   * @brief Constructor.
   *
   * @param size Size of the memory space.
   */
  inline MemorySpace(const size_t size) : _memory_space(size) {}

  /**
   * @brief Access the buffer with the given index.
   */
  inline PhotonBuffer &operator[](const size_t index) {
    return _memory_space[index];
  }

  /**
   * @brief Get the index of a free buffer in the memory space.
   *
   * @return Index of a free buffer.
   */
  inline size_t get_free_buffer() { return _memory_space.get_free_element(); }

  /**
   * @brief Free the buffer with the given index.
   *
   * @param index Index of a buffer that was in use.
   */
  inline void free_buffer(const size_t index) {
    _memory_space[index]._actual_size = 0;
    _memory_space.free_element(index);
  }

  /**
   * @brief Copy photons from the given buffer into the buffer with the given
   * index.
   *
   * This method assumes the buffer was successfully locked before it was
   * called. It copies photons until the buffer is full, and starts a new buffer
   * if that happens.
   *
   * @param index Index of the buffer we want to add to.
   * @param buffer Buffer to copy over.
   * @return The index of the last buffer we copied into. If this index is not
   * the same as the original index, the original index should be queued.
   */
  inline size_t add_photons(const size_t index, PhotonBuffer &buffer) {
    PhotonBuffer &buffer_target = _memory_space[index];
    unsigned int &counter_target = buffer_target._actual_size;
    const unsigned int size_in = buffer._actual_size;
    unsigned int counter_in = 0;
    while (counter_target < PHOTONBUFFER_SIZE && counter_in < size_in) {
      buffer_target._photons[counter_target] = buffer._photons[counter_in];
      ++counter_target;
      ++counter_in;
    }
    size_t index_out = index;
    if (counter_target == PHOTONBUFFER_SIZE) {
      index_out = get_free_buffer();
      // note that the hungry other threads might already be attacking this
      // buffer. We need to make sure they release it without deleting it if
      // it does not (yet) contain any photons.
      PhotonBuffer &buffer_target_new = _memory_space[index_out];
      // copy the old buffer properties
      buffer_target_new._sub_grid_index = buffer_target._sub_grid_index;
      buffer_target_new._direction = buffer_target._direction;
      unsigned int &counter_target_new = buffer_target_new._actual_size;
      // maybe assert that this value is indeed 0?
      while (counter_in < size_in) {
        buffer_target_new._photons[counter_target_new] =
            buffer._photons[counter_in];
        ++counter_target_new;
        ++counter_in;
      }
    }
    return index_out;
  }
};

#endif // MEMORYSPACE_HPP
