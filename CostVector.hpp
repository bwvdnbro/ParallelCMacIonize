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
 * @file CostVector.hpp
 *
 * @brief Object that keeps track of computational costs and tries to balance
 * them.
 *
 * @author Bert Vandenbroucke (bv7@st-andrews.ac.uk)
 */
#ifndef COSTVECTOR_HPP
#define COSTVECTOR_HPP

#include "Assert.hpp"

#include <algorithm>
#include <vector>

/**
 * @brief Sort the given array with given size by argument.
 *
 * @param v Array to sort.
 * @param size Size of the array.
 * @return std::vector<size_t> containing the indices of the array in an order
 * that would sort the array.
 */
template < typename _datatype_ >
inline std::vector< size_t > argsort(const _datatype_ *v, const size_t size) {
  std::vector< size_t > indices(size, 0);
  std::iota(indices.begin(), indices.end(), 0);
  std::sort(indices.begin(), indices.end(),
            [v](size_t i1, size_t i2) { return v[i1] < v[i2]; });
  return indices;
}

/**
 * @brief Object that keeps track of computational costs and tries to balance
 * them.
 */
class CostVector {
private:
  /*! @brief Number of elements. */
  const size_t _size;

  /*! @brief Number of threads. */
  const int _number_of_threads;

  /*! @brief Number of MPI processes. */
  const int _number_of_processes;

  /*! @brief Computational costs for each element. */
  unsigned long *_costs;

  /*! @brief Thread list that links elements to threads in an optimal balancing
   *  scheme. */
  int *_thread_list;

  /*! @brief Process list that links elements to MPI processes in an optimal
   *  balancing scheme. */
  int *_process_list;

public:
  /**
   * @brief Constructor.
   *
   * @param size Number of elements.
   * @param number_of_threads Number of threads.
   * @param number_of_proceses Number of MPI processes.
   */
  inline CostVector(const size_t size, const int number_of_threads,
                    const int number_of_processes)
      : _size(size), _number_of_threads(number_of_threads),
        _number_of_processes(number_of_processes) {
    _costs = new unsigned long[size];
    _thread_list = new int[size];
    _process_list = new int[size];
    for (size_t i = 0; i < size; ++i) {
      _costs[i] = 0;
      // our initial decomposition
      _thread_list[i] = i % number_of_threads;
      _process_list[i] = i % number_of_processes;
    }
  }

  /**
   * @brief Destructor.
   */
  inline ~CostVector() {
    delete[] _costs;
    delete[] _thread_list;
    delete[] _process_list;
  }

  /**
   * @brief Get the thread id for the given element.
   *
   * @param index Index of an element.
   * @return Thread id that owns that element.
   */
  inline const int get_thread(const size_t index) const {
    return _thread_list[index];
  }

  /**
   * @brief Get the process id for the given element.
   *
   * @param index Index of an element.
   * @return Process rank that owns that element.
   */
  inline const int get_process(const size_t index) const {
    return _process_list[index];
  }

  /**
   * @brief Add the given computational cost to the given element.
   *
   * @param index Index of an element.
   * @param cost Computational cost to add.
   */
  inline void add_cost(const size_t index, const unsigned long cost) {
    _costs[index] += cost;
  }

  /**
   * @brief Set the computational cost for the given element.
   *
   * @param index Index of an element.
   * @param cost New computation cost.
   */
  inline void set_cost(const size_t index, const unsigned long cost) {
    _costs[index] = cost;
  }

  /**
   * @brief Get the computational cost for the given element.
   *
   * @param index Index of an element.
   * @return Computational cost for that element.
   */
  inline const unsigned long get_cost(const size_t index) const {
    return _costs[index];
  }

  /**
   * @brief Redistribute the elements such that the computational cost for each
   * thread is as close as possible to the average.
   */
  inline void redistribute() {

    // argsort the elements based on cost
    std::vector< size_t > indices = argsort(_costs, _size);
    // store the cost per thread for later
    std::vector< std::vector< unsigned long > > threadcost(
        _number_of_processes,
        std::vector< unsigned long >(_number_of_threads, 0));
    // loop over the subgrids in descending cost order
    size_t index = 0;
    const size_t max_index = _size - 1;
    // first pass: give every thread and rank an expensive element
    // we do ranks in the inner loop to load balance across processes
    for (int ithread = 0; ithread < _number_of_threads; ++ithread) {
      for (int irank = 0; irank < _number_of_processes; ++irank) {
        const size_t current_index = indices[max_index - index];
        _thread_list[current_index] = ithread;
        _process_list[current_index] = irank;
        threadcost[irank][ithread] += _costs[current_index];
        ++index;
      }
    }
    // second pass: add the remaining indices in an optimal way: we try to
    // find the thread that can fit a cost best
    for (; index < _size; ++index) {
      const size_t current_index = indices[max_index - index];
      // find the thread where this cost has the lowest impact
      int cmatch_rank = -1;
      int cmatch_thread = -1;
      unsigned long cmatch = threadcost[0][0] + _costs[current_index];
      for (int irank = 0; irank < _number_of_processes; ++irank) {
        for (int ithread = 0; ithread < _number_of_threads; ++ithread) {
          const unsigned long cvalue =
              threadcost[irank][ithread] + _costs[current_index];
          if (cvalue <= cmatch) {
            cmatch = cvalue;
            cmatch_rank = irank;
            cmatch_thread = ithread;
          }
        }
      }
      myassert(cmatch_rank >= 0 && cmatch_thread >= 0, "No closest match!");
      _thread_list[current_index] = cmatch_thread;
      _process_list[current_index] = cmatch_rank;
      threadcost[cmatch_rank][cmatch_thread] += _costs[current_index];
    }

    // reset costs
    for (size_t i = 0; i < _size; ++i) {
      _costs[i] = 0;
    }
  }
};

#endif // COSTVECTOR_HPP
