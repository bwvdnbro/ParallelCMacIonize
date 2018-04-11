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
 * @file TaskSpace.hpp
 *
 * @brief Memory space that contains all tasks.
 *
 * @author Bert Vandenbroucke (bv7@st-andrews.ac.uk)
 */
#ifndef TASKSPACE_HPP
#define TASKSPACE_HPP

#include "Atomic.hpp"
#include "Lock.hpp"
#include "Task.hpp"

/**
 * @brief Memory space that contains all tasks.
 */
class TaskSpace {
private:
  /*! @brief Current active index in the vector (volatile since we want to make
   *  sure all threads see the same value). */
  volatile size_t _current_index;

  /*! @brief Size of the vector. */
  const size_t _size;

  /*! @brief Current active index in the dependency vector (volatile since we
   *  want all threads to see the same value). */
  volatile size_t _dependency_index;

  /*! @brief Size of the dependency vector. */
  const size_t _dependency_size;

  /*! @brief Vector itself. */
  Task *_tasks;

  /*! @brief Dependency vector. */
  Lock **_dependencies;

  /*! @brief Lock for the space. */
  Lock _lock;

public:
  /**
   * @brief Constructor.
   *
   * @param size Size of the internal array.
   * @param dependency_size Size of the internal dependency array.
   */
  inline TaskSpace(const size_t size, const size_t dependency_size)
      : _current_index(0), _size(size), _dependency_index(0),
        _dependency_size(dependency_size) {

    _tasks = new Task[_size];
    _dependencies = new Lock *[_dependency_size];
  }

  /**
   * @brief Destructor.
   */
  inline ~TaskSpace() {
    delete[] _tasks;
    delete[] _dependencies;
  }

  /**
   * @brief Clear the contents of the vector.
   *
   * This method is not meant to be thread safe.
   */
  inline void clear() {
    _current_index = 0;
    _dependency_index = 0;

    // clear the elements
    delete[] _tasks;
    _tasks = new Task[_size];
    delete[] _dependencies;
    _dependencies = new Lock *[_dependency_size];
  }

  /**
   * @brief Access the task with the given index.
   *
   * @param index Index of a task.
   * @return Read/write reference to the task with that index.
   */
  inline Task &operator[](const size_t index) { return _tasks[index]; }

  /**
   * @brief Read-only access to the task with the given index.
   *
   * @param index Index of a task.
   * @return Read-only reference to the task with that index.
   */
  inline const Task &operator[](const size_t index) const {
    return _tasks[index];
  }

  /**
   * @brief Get the index of a free task in the vector.
   *
   * @param num_dependencies Number of dependencies for the task.
   * @return Index of a free task.
   */
  inline size_t get_free_task(const unsigned int num_dependencies) {
    _lock.lock();
    myassert(_current_index < _size, "No more free tasks!");
    const size_t index = _current_index;
    ++_current_index;
    const size_t dependency_index = _dependency_index;
    _dependency_index += num_dependencies;
    _lock.unlock();

    // we now have thread safe access to a unique task
    _tasks[index]._first_dependency = dependency_index;
    _tasks[index]._number_of_dependencies = num_dependencies;

    return index;
  }

  /**
   * @brief Try to lock the task with the given index.
   *
   * We try to lock all the dependencies of the task. If this fails, we unlock
   * the dependencies that were already locked and return.
   *
   * @param index Index of a task.
   * @return True if all dependencies for the task were successfully locked.
   */
  inline bool lock_task(const size_t index) {
    const size_t last_dependency =
        _tasks[index]._first_dependency + _tasks[index]._number_of_dependencies;
    size_t dependency;
    for (dependency = _tasks[index]._first_dependency;
         dependency < last_dependency; ++dependency) {
      if (!_dependencies[dependency]->try_lock()) {
        break;
      }
    }
    if (dependency < last_dependency) {
      // lock failed, unlock all locked dependencies in reverse order
      for (; dependency > _tasks[index]._first_dependency; --dependency) {
        _dependencies[dependency]->unlock();
      }
      // unlock the last dependency (we cannot use our exit condition for the
      // last dependency, since dependency is unsigned, which would give
      // problems for first_dependency = 0)
      _dependencies[dependency]->unlock();
      return false;
    } else {
      return true;
    }
  }

  /**
   * @brief Unlock all dependencies of the given task
   *
   * @param index Index of a task that was previously locked.
   */
  inline void unlock_task(const size_t index) {
    const size_t last_dependency =
        _tasks[index]._first_dependency + _tasks[index]._number_of_dependencies;
    for (size_t dependency = _tasks[index]._first_dependency;
         dependency < last_dependency; ++dependency) {
      _dependencies[dependency]->unlock();
    }
  }

  /**
   * @brief Add a dependency for the given task.
   *
   * @param index Index of a task.
   * @param dependency_offset Offset of the dependency in the list for that
   * task.
   * @param dependency Pointer to the actual dependency.
   */
  inline void add_dependency(const size_t index, const size_t dependency_offset,
                             Lock *dependency) {
    myassert(dependency_offset < _tasks[index]._number_of_dependencies,
             "Too many dependencies for task!");
    const size_t dependency_index =
        _tasks[index]._first_dependency + dependency_index;
    _dependencies[dependency_index] = dependency;
  }

  /**
   * @brief Get the size of the internal array.
   *
   * @return Number of used tasks.
   */
  inline const size_t size() const { return _current_index; }

  /**
   * @brief Maximum number of tasks in the vector.
   *
   * @return Maximum number of tasks that can be stored in the vector.
   */
  inline const size_t max_size() const { return _size; }
};

#endif // TASKSPACE_HPP
