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
 * @file Queue.hpp
 *
 * @brief PhotonBuffer queue.
 *
 * @author Bert Vandenbroucke (bv7@st-andrews.ac.uk)
 */
#ifndef QUEUE_HPP
#define QUEUE_HPP

#include "Assert.hpp"
#include "Atomic.hpp"
#include "PhotonBuffer.hpp"

#include <fstream>
#include <sstream>
#include <vector>

#define QUEUE_SIZE 2000

//#define QUEUE_LOG

#ifdef QUEUE_LOG
#define queue_log_variable std::ofstream *_log
#else
#define queue_log_variable
#endif

#ifdef QUEUE_LOG
#define queue_init(name) _log = new std::ofstream(name)
#else
#define queue_init(name)
#endif

#ifdef QUEUE_LOG
#define queue_free() delete _log
#else
#define queue_free()
#endif

#ifdef QUEUE_LOG
#define queue_log(message)                                                     \
  *_log << message << "\n";                                                    \
  _log->flush();
#else
#define queue_log(message)
#endif

/**
 * @brief PhotonBuffer queue.
 */
class Queue {
private:
  /*! @brief PhotonBuffers stored in this queue. */
  unsigned int _buffers[QUEUE_SIZE];

  /*! @brief Last element in the queue. */
  unsigned int _last_element;

  /*! @brief Memory space of the queue. */
  PhotonBuffer _memory_space[QUEUE_SIZE];

  /*! @brief Flags telling us if memory is available or not. */
  bool _memory_taken[QUEUE_SIZE];

  /*! @brief Flags telling us if memory is queued or not. */
  bool _memory_queued[QUEUE_SIZE];

  /*! @brief Last used memory block. */
  unsigned int _memory_index;

  /*! @brief Number of free memory spaces. */
  unsigned int _memory_free;

  /*! @brief Lock to protect the buffer queue. */
  bool _buffer_lock;

  /*! @brief Log to write task info to. */
  queue_log_variable;

  inline static std::string get_logname(int rank) {
    std::stringstream logname;
    logname << "thread_" << rank;
    return logname.str();
  }

public:
  /**
   * @brief Constructor.
   */
  inline Queue()
      : _last_element(0), _memory_index(0), _memory_free(QUEUE_SIZE),
        _buffer_lock(false) {
    for (unsigned int i = 0; i < QUEUE_SIZE; ++i) {
      _memory_taken[i] = false;
      _memory_queued[i] = false;
    }
  }

  inline ~Queue() { queue_free(); }

  inline void set_rank(int rank) { queue_init(get_logname(rank)); }

  /**
   * @brief Add the buffer with the given index in the internal memory space to
   * the queue.
   *
   * The buffer can only be processed once it has been added. This method locks
   * the queue.
   *
   * @param buffer Index of the buffer in the internal memory space.
   */
  inline void add_buffer(const unsigned int buffer) {
    while (!atomic_lock(_buffer_lock)) {
    }
    atomic_lock(_memory_queued[buffer]);
    _buffers[_last_element] = buffer;
    queue_log("added buffer "
              << buffer << " to queue index " << _last_element << ", contains "
              << _memory_space[buffer]._actual_size << " photons");
    ++_last_element;
    myassert(_last_element != QUEUE_SIZE, "queue size overflow!");
    atomic_unlock(_buffer_lock);
  }

  /**
   * @brief Get a buffer from the queue.
   *
   * This function locks the queue.
   *
   * @param index Variable to store the resulting buffer index in.
   * @return Pointer to the buffer.
   */
  inline PhotonBuffer *get_buffer(unsigned int &index) {
    PhotonBuffer *result = nullptr;
    while (!atomic_lock(_buffer_lock)) {
    }
    if (_last_element > 0) {
      --_last_element;
      index = _buffers[_last_element];
      result = &_memory_space[index];
      queue_log("retrieved buffer " << index << " from queue index "
                                    << _last_element << ", contains "
                                    << result->_actual_size << " photons");
    } else {
      queue_log("refused buffer request because of empty queue");
    }
    atomic_unlock(_buffer_lock);
    return result;
  }

  /**
   * @brief Get a free buffer in the internal memory space.
   *
   * This function is lock-free, but can deadlock if there are no available
   * buffers.
   *
   * @param index Variable to store the resulting buffer index in.
   * @return Pointer to the free memory buffer.
   */
  inline PhotonBuffer *get_free_buffer(unsigned int &index) {
    myassert(_memory_free > 0, "No more free memory!");
    // if _memory_index overflows, it is reset to 0. This is what we want.
    index = atomic_post_increment(_memory_index) % QUEUE_SIZE;
    // loop over the buffers until we find one that can be locked
    unsigned int num_attempt = 0;
    while (!atomic_lock(_memory_taken[index])) {
      ++num_attempt;
      myassert(num_attempt < 2 * QUEUE_SIZE, "cannot obtain free buffer!");
      index = atomic_post_increment(_memory_index) % QUEUE_SIZE;
    }
    atomic_pre_decrement(_memory_free);
    return &_memory_space[index];
  }

  /**
   * @brief Free the buffer with the given index.
   *
   * Only buffers that have been freed in this way can be used again.
   *
   * @param index Index of the buffer to free.
   */
  inline void free_buffer(const unsigned int index) {
    atomic_unlock(_memory_taken[index]);
    atomic_unlock(_memory_queued[index]);
    atomic_pre_increment(_memory_free);
  }
};

#endif // QUEUE_HPP
