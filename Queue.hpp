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
 * @brief Task queue.
 *
 * @author Bert Vandenbroucke (bv7@st-andrews.ac.uk)
 */
#ifndef QUEUE_HPP
#define QUEUE_HPP

#include "Assert.hpp"
#include "Atomic.hpp"
#include "Lock.hpp"
#include "Task.hpp"
#include "ThreadSafeVector.hpp"

#define NO_TASK 0xffffffff

/**
 * @brief Task queue.
 */
class Queue {
private:
  /*! @brief Queue. */
  size_t *_queue;

  /*! @brief Current size of the queue. */
  size_t _current_queue_size;

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
  inline Queue(const size_t size) : _current_queue_size(0), _size(size) {
    _queue = new size_t[size];
  }

  /**
   * @brief Destructor.
   */
  inline ~Queue() { delete[] _queue; }

  /**
   * @brief Add a task to the queue.
   *
   * @param task Task to add.
   */
  inline void add_task(const size_t task) {
    _queue_lock.lock();
    myassert(_current_queue_size < _size, "Too many tasks in queue!");
    _queue[_current_queue_size] = task;
    ++_current_queue_size;
    _queue_lock.unlock();
  }

  /**
   * @brief Get a task from the queue.
   *
   * This version locks the queue.
   *
   * @param tasks Task space.
   * @return Task, or NO_TASK if no task is available.
   */
  inline size_t get_task(ThreadSafeVector< Task > &tasks) {

    // initialize an empty task
    size_t task = NO_TASK;

    _queue_lock.lock();

    // now try to find a task whose dependency can be locked
    size_t index = _current_queue_size;
    while (index > 0 && !tasks[_queue[index - 1]].lock_dependency()) {
      --index;
    }
    if (index > 0) {
      // we found a task and locked it
      --index;
      --_current_queue_size;
      task = _queue[index];

      // shuffle all tasks to close the gap created by removing the task
      for (; index < _current_queue_size; ++index) {
        _queue[index] = _queue[index + 1];
      }
    }

    // we're done: unlock the queue
    _queue_lock.unlock();

    // return the task
    return task;
  }

  /**
   * @brief Try to get a task from the queue.
   *
   * This version tries to lock the queue and bails out if another thread is
   * accessing it.
   *
   * @param tasks Task space.
   * @return Task, or NO_TASK if no task is available.
   */
  inline size_t try_get_task(ThreadSafeVector< Task > &tasks) {

    // initialize an empty task
    size_t task = NO_TASK;

    // lock the queue while we are getting a task
    if (_queue_lock.try_lock()) {

      // now try to find a task whose dependency can be locked
      size_t index = _current_queue_size;
      while (index > 0 && !tasks[_queue[index - 1]].lock_dependency()) {
        --index;
      }
      if (index > 0) {
        // we found a task and locked it
        --index;
        --_current_queue_size;
        task = _queue[index];

        // shuffle all tasks to close the gap created by removing the task
        for (; index < _current_queue_size; ++index) {
          _queue[index] = _queue[index + 1];
        }
      }

      // we're done: unlock the queue
      _queue_lock.unlock();
    }

    // return the task
    return task;
  }

  /**
   * @brief Get the current size of the queue.
   *
   * @return Current size of the queue.
   */
  inline size_t size() const { return _current_queue_size; }
};

#endif // QUEUE_HPP
