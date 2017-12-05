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

/**
 * @brief Buffer with pre-allocated PhotonBuffers that can be used.
 */
class MemorySpace {
private:
  /*! @brief Current active index in the memory space. */
  size_t _current_index;

  /*! @brief Size of the memory space. */
  const size_t _size;

  /*! @brief Number of buffers that have been taken. */
  size_t _number_taken;

  /*! @brief Memory space itself. */
  PhotonBuffer *_memory_space;

  /*! @brief Lock protecting the memory space. */
  Lock _lock;

public:
  /**
   * @brief Constructor.
   *
   * @param size Size of the memory space.
   */
  inline MemorySpace(const size_t size)
      : _current_index(0), _size(size), _number_taken(0) {
    _memory_space = new PhotonBuffer[size];
    for (size_t i = 0; i < size; ++i) {
      _memory_space[i]._actual_size = 0;
      _memory_space[i]._is_in_use = false;
    }
  }

  /**
   * @brief Destructor.
   */
  inline ~MemorySpace() { delete[] _memory_space; }

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
  inline size_t get_free_buffer() {
    //    myassert(_number_taken < _size, "No more free buffers in memory
    //    space!");
    //    size_t index = atomic_post_increment(_current_index) % _size;
    //    while (!atomic_lock(_memory_space[index]._is_in_use)) {
    //      index = atomic_post_increment(_current_index) % _size;
    //    }
    //    atomic_pre_increment(_number_taken);
    //    return index;

    myassert(_number_taken < _size, "No more free buffers in memory space!");
    _lock.lock();
    while (_memory_space[_current_index]._is_in_use) {
      _current_index = (_current_index + 1) % _size;
    }
    const size_t index = _current_index;
    _memory_space[index]._is_in_use = true;
    ++_number_taken;
    _lock.unlock();
    return index;
  }

  /**
   * @brief Free the buffer with the given index.
   *
   * @param index Index of a buffer that was in use.
   */
  inline void free_buffer(const size_t index) {
    //    _memory_space[index]._actual_size = 0;
    //    atomic_unlock(_memory_space[index]._is_in_use);
    //    atomic_post_increment(_number_taken);

    _lock.lock();
    _memory_space[index]._actual_size = 0;
    _memory_space[index]._is_in_use = false;
    --_number_taken;
    _lock.unlock();
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
