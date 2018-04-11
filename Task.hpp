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

/**
 * @brief Get the CPU cycle time stamp.
 *
 * @param time_variable Variable to store the result in.
 */
#define task_tick(time_variable)                                               \
  {                                                                            \
    unsigned int lo, hi;                                                       \
    __asm__ __volatile__("rdtsc" : "=a"(lo), "=d"(hi));                        \
    time_variable = ((unsigned long)hi << 32) | lo;                            \
  }

/**
 * @brief Types of tasks.
 */
enum TaskType {
  /*! @brief Draw random photons from the source. */
  TASKTYPE_SOURCE_PHOTON,
  /*! @brief Propagate photons through a subgrid. */
  TASKTYPE_PHOTON_TRAVERSAL,
  /*! @brief Reemit photons. */
  TASKTYPE_PHOTON_REEMIT,
  /*! @brief Send a buffer to another process. */
  TASKTYPE_SEND,
  /*! @brief Receive a buffer from another process. */
  TASKTYPE_RECV
};

/**
 * @brief Task interface.
 */
class Task {
public:
  /*! @brief Task type. */
  int _type;

  /*! @brief Index of the associated cell. */
  size_t _cell;

  /*! @brief Index of the associated input photon buffer (if any). */
  size_t _buffer;

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

  /**
   * @brief Record the start time of the task.
   *
   * @param thread_id Thread that executes the task.
   */
  inline void start(const int thread_id) {
#ifdef TASK_PLOT
    _thread_id = thread_id;
    task_tick(_start_time);
#endif
  }

  /**
   * @brief Record the end time of the task.
   */
  inline void stop() {
#ifdef TASK_PLOT
    task_tick(_end_time);
#endif
  }

  /**
   * @brief Empty constructor.
   *
   * Used to flag unexecuted tasks.
   */
  Task() : _end_time(0) {}
};

#endif // TASK_HPP
