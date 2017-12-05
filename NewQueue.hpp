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
 * @file NewQueue.hpp
 *
 * @brief PhotonBuffer queue for a single thread.
 *
 * @author Bert Vandenbroucke (bv7@st-andrews.ac.uk)
 */
#ifndef NEWQUEUE_HPP
#define NEWQUEUE_HPP

#include "Assert.hpp"
#include "Atomic.hpp"
#include "Lock.hpp"

#define NO_TASK 0xffffffff

/**
 * @brief PhotonBuffer queue for a single thread.
 */
class NewQueue {
private:
  /*! @brief Queue. */
  size_t *_queue;

  /*! @brief Last element in the queue. */
  size_t _current_queue_index;

  /*! @brief Size of the queues. */
  const size_t _size;

  /*! @brief Lock that protects the queue. */
  Lock _queue_lock;

public:
  /**
   * @brief Constructor.
   *
   * @param size Size of the queue.
   */
  inline NewQueue(const size_t size) : _current_queue_index(0), _size(size) {
    _queue = new size_t[size];
  }

  /**
   * @brief Destructor.
   */
  inline ~NewQueue() { delete[] _queue; }

  /**
   * @brief Add a task to the queue.
   *
   * @param task Task to add.
   */
  inline void add_task(const size_t task) {
    _queue_lock.lock();
    _queue[_current_queue_index] = task;
    ++_current_queue_index;
    myassert(_current_queue_index < _size, "Too many tasks in queue!");
    _queue_lock.unlock();
  }

  /**
   * @brief Get a task from the queue.
   *
   * @return Task, or NO_TASK if no task is available.
   */
  inline size_t get_task() {
    _queue_lock.lock();
    size_t task = NO_TASK;
    if (_current_queue_index > 0) {
      --_current_queue_index;
      task = _queue[_current_queue_index];
    }
    _queue_lock.unlock();
    return task;
  }
};

#endif // NEWQUEUE_HPP
