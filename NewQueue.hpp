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

#define NO_TASK 0xffffffff

/**
 * @brief PhotonBuffer queue for a single thread.
 */
class NewQueue {
private:
  /*! @brief Queue. */
  size_t *_queue;

  /*! @brief Last element in the queue. */
  size_t _current_index;

  /*! @brief Size of the queue. */
  const size_t _size;

  /*! @brief Lock that protects the queue. */
  bool _lock;

public:
  /**
   * @brief Constructor.
   *
   * @param size Size of the queue.
   */
  inline NewQueue(const size_t size) : _current_index(0), _size(size) {
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
    while (!atomic_lock(_lock)) {
    }
    _queue[_current_index] = task;
    ++_current_index;
    myassert(_current_index < _size, "Too many tasks in queue!");
    atomic_unlock(_lock);
  }

  /**
   * @brief Get a task from the queue.
   *
   * @return Task, or NO_TASK if no task is available.
   */
  inline size_t get_task() {
    while (!atomic_lock(_lock)) {
    }
    size_t task = NO_TASK;
    if (_current_index > 0) {
      --_current_index;
      task = _queue[_current_index];
    }
    atomic_unlock(_lock);
    return task;
  }
};

#endif // NEWQUEUE_HPP
