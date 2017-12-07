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

  /*! @brief Computational costs for each element. */
  unsigned long *_costs;

  /*! @brief Thread list that links elements to threads in an optimal balancing
   *  scheme. */
  int *_thread_list;

public:
  /**
   * @brief Constructor.
   *
   * @param size Number of elements.
   * @param number_of_threads Number of threads.
   */
  inline CostVector(const size_t size, const int number_of_threads)
      : _size(size), _number_of_threads(number_of_threads) {
    _costs = new unsigned long[size];
    _thread_list = new int[size];
    for (size_t i = 0; i < size; ++i) {
      _costs[i] = 0;
      // our initial decomposition
      _thread_list[i] = i % number_of_threads;
    }
  }

  /**
   * @brief Destructor.
   */
  inline ~CostVector() {
    delete[] _costs;
    delete[] _thread_list;
  }

  /**
   * @brief Get the thread id for the given element.
   *
   * @param index Index of an element.
   * @return Thread id that owns that element.
   */
  inline const int &operator[](const size_t index) const {
    return _thread_list[index];
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
    // compute the target cost per thread
    unsigned long totcost = 0;
    for (size_t i = 0; i < _size; ++i) {
      totcost += _costs[i];
    }
    unsigned long target_per_thread = totcost / _number_of_threads;
    // remove all subgrids with above average costs from the total, to avoid
    // having threads without any subgrids
    unsigned int num_large = 0;
    for (size_t i = 0; i < _size; ++i) {
      if (_costs[i] > target_per_thread) {
        totcost -= _costs[i];
        ++num_large;
      }
    }
    target_per_thread = totcost / (_number_of_threads - num_large);

    // now for the actual distribution:
    //  we want to assign the _number_of_threads most expensive subgrids to
    //  different threads, and then fill up the threads with the least expensive
    //  subgrids until the target is reached
    // argsort the costs
    std::vector< size_t > indices = argsort(_costs, _size);
    // subdivide
    size_t low_index = 0;
    size_t high_index = _size - 1;
    const size_t max_low_index = _size - _number_of_threads;
    int thread = 0;
    while (thread < _number_of_threads) {
      totcost = 0;
      _thread_list[indices[high_index]] = thread;
      totcost += _costs[indices[high_index]];
      --high_index;
      while (totcost < target_per_thread && low_index < max_low_index) {
        _thread_list[indices[low_index]] = thread;
        totcost += _costs[indices[low_index]];
        ++low_index;
      }
      ++thread;
    }
    // the loop above might have missed the last subgrid; add it here
    --thread;
    while (low_index < max_low_index) {
      _thread_list[indices[low_index]] = thread;
      totcost += _costs[indices[low_index]];
      ++low_index;
    }
    myassert(high_index + 1 == low_index,
             "Missing subgrids (high: " << high_index << ", low: " << low_index
                                        << "!");

    // reset costs
    for (size_t i = 0; i < _size; ++i) {
      _costs[i] = 0;
    }
  }
};

#endif // COSTVECTOR_HPP
