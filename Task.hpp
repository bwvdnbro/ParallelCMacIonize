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
 * @file Task.hpp
 *
 * @brief Task interface.
 *
 * @author Bert Vandenbroucke (bv7@st-andrews.ac.uk)
 */
#ifndef TASK_HPP
#define TASK_HPP

/*! @brief Activate this to record the start and end time of each task. */
//#define TASK_PLOT

#include "CPUCycle.hpp"
#include "Lock.hpp"

/**
 * @brief Types of tasks.
 */
enum TaskType {
  /*! @brief Draw random photons from the source. */
  TASKTYPE_SOURCE_PHOTON = 0,
  /*! @brief Propagate photons through a subgrid. */
  TASKTYPE_PHOTON_TRAVERSAL,
  /*! @brief Reemit photons. */
  TASKTYPE_PHOTON_REEMIT,
  /*! @brief Send a buffer to another process. */
  TASKTYPE_SEND,
  /*! @brief Receive a buffer from another process. */
  TASKTYPE_RECV,
  /*! @brief Task type counter. */
  TASKTYPE_NUMBER
};

/**
 * @brief Task interface.
 */
class Task {
private:
  /*! @brief Task type. */
  int _type;

  /*! @brief Index of the associated subgrid. */
  size_t _subgrid;

  /*! @brief Index of the associated input photon buffer (if any). */
  size_t _buffer;

  /*! @brief Dependency (if any). */
  Lock *_dependency;

  /*! @brief Index of the first dependency of the task. */
  size_t _first_dependency;

  /*! @brief Number of dependencies of the task. */
  size_t _number_of_dependencies;

#ifdef TASK_PLOT
  /*! @brief Rank of the thread that executed the task. */
  int _thread_id;

  /*! @brief Time stamp for the start of the task. */
  unsigned long _start_time;

  /*! @brief Time stamp for the end of the task. */
  unsigned long _end_time;
#endif

public:
  /**
   * @brief Empty constructor.
   *
   * Used to flag unexecuted tasks and initialize the dependency.
   */
  Task() : _dependency(nullptr) {
#ifdef TASK_PLOT
    _end_time = 0;
#else
    _type = TASKTYPE_NUMBER;
#endif
  }

  /**
   * @brief Record the start time of the task.
   *
   * @param thread_id Thread that executes the task.
   */
  inline void start(const int thread_id) {
#ifdef TASK_PLOT
    _thread_id = thread_id;
    cpucycle_tick(_start_time);
#endif
  }

  /**
   * @brief Record the end time of the task.
   */
  inline void stop() {
#ifdef TASK_PLOT
    cpucycle_tick(_end_time);
#else
    // we need another way to flag the end of the task
    // since we do not care about what task this was (we don't plot it), we can
    // overwrite the type variable
    _type = -1;
#endif
  }

  /**
   * @brief Check if the task was already done.
   *
   * @return True if the task was executed, false otherwise.
   */
  inline const bool done() const {
#ifdef TASK_PLOT
    return _end_time > 0;
#else
    return _type == -1;
#endif
  }

  /**
   * @brief Set the dependency for the task.
   *
   * @param dependency Dependency.
   */
  inline void set_dependency(Lock *dependency) { _dependency = dependency; }

  /**
   * @brief Try to lock the dependency (if there is one) for the task.
   *
   * @return True if locking succeeded, false otherwise.
   */
  inline const bool lock_dependency() {
    if (_dependency != nullptr) {
      return _dependency->try_lock();
    } else {
      return true;
    }
  }

  /**
   * @brief Unlock the dependency (if there is one) for the task.
   */
  inline void unlock_dependency() {
    if (_dependency != nullptr) {
      _dependency->unlock();
    }
  }

  /**
   * @brief Get the type of the task.
   *
   * @return Type of the task.
   */
  inline const int get_type() const { return _type; }

  /**
   * @brief Set the type of the task.
   *
   * @param type Type of the task.
   */
  inline void set_type(const int type) { _type = type; }

  /**
   * @brief Get the buffer associated with this task.
   *
   * @return Index of the associated buffer.
   */
  inline const unsigned int get_buffer() const { return _buffer; }

  /**
   * @brief Set the buffer associated with this task.
   *
   * @param buffer Index of the associated buffer.
   */
  inline void set_buffer(const unsigned int buffer) { _buffer = buffer; }

  /**
   * @brief Set the subgrid associated with this task.
   *
   * @return Index of the associated subgrid.
   */
  inline const unsigned int get_subgrid() const { return _subgrid; }

  /**
   * @brief Set the subgrid associated with this task.
   *
   * @param subgrid Index of the associated subgrid.
   */
  inline void set_subgrid(const unsigned int subgrid) { _subgrid = subgrid; }

#ifdef TASK_PLOT
  /**
   * @brief Get all information necessary to write the task to an output file.
   *
   * @param type Variable to store the task type.
   * @param thread_id Variable to store the id of the thread that executed
   * the task.
   * @param start Variable to store the CPU cycle count at the start of task
   * execution.
   * @param end Variable to store the CPU cycle count at the end of task
   * execution.
   */
  inline void get_timing_information(int &type, int &thread_id,
                                     unsigned long &start,
                                     unsigned long &end) const {
    type = _type;
    thread_id = _thread_id;
    start = _start_time;
    end = _end_time;
  }
#else
#error                                                                         \
    "Task::get_timing_information() is only available when TASK_PLOT is defined!"
#endif
};

#endif // TASK_HPP
