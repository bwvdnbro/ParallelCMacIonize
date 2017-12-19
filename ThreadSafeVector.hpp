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
 * @file ThreadSafeVector.hpp
 *
 * @brief Thread safe fixed size vector.
 *
 * @author Bert Vandenbroucke (bv7@st-andrews.ac.uk)
 */
#ifndef THREADSAFEVECTOR_HPP
#define THREADSAFEVECTOR_HPP

#include "Atomic.hpp"

/**
 * @brief Thread safe fixed size vector.
 */
template <typename _datatype_> class ThreadSafeVector {
private:
  /*! @brief Current active index in the vector. */
  size_t _current_index;

  /*! @brief Size of the vector. */
  const size_t _size;

  /*! @brief Number of elements that have been taken. */
  size_t _number_taken;

  /*! @brief Vector itself. */
  _datatype_ *_vector;

  /*! @brief Atomic locks for the vector elements. */
  bool *_locks;

public:
  /**
   * @brief Constructor.
   *
   * @param size Size of the vector.
   */
  inline ThreadSafeVector(const size_t size)
      : _current_index(0), _size(size), _number_taken(0) {
    _vector = new _datatype_[size];
    _locks = new bool[size];
    for (size_t i = 0; i < size; ++i) {
      _locks[i] = false;
    }
  }

  /**
   * @brief Destructor.
   */
  inline ~ThreadSafeVector() {
    delete[] _vector;
    delete[] _locks;
  }

  /**
   * @brief Clear the contents of the vector.
   *
   * This method is not meant to be thread safe.
   */
  inline void clear() {
    for (size_t i = 0; i < _size; ++i) {
      _locks[i] = false;
    }
    _number_taken = 0;
    _current_index = 0;
  }

  /**
   * @brief Access the element with the given index.
   *
   * @param index Index of an element.
   * @return Read/write reference to the element with that index.
   */
  inline _datatype_ &operator[](const size_t index) {
    myassert(_locks[index], "Element not in use!");
    return _vector[index];
  }

  /**
   * @brief Read-only access to the element with the given index.
   *
   * @param index Index of an element.
   * @return Read-only reference to the element with that index.
   */
  inline const _datatype_ &operator[](const size_t index) const {
    myassert(_locks[index], "Element not in use!");
    return _vector[index];
  }

  /**
   * @brief Get the index of a free element in the vector.
   *
   * This element will be locked and needs to be freed later by calling
   * free_element().
   *
   * @return Index of a free element.
   */
  inline size_t get_free_element() {
    myassert(_number_taken < _size, "No more free elements in vector!");
    size_t index = atomic_post_increment(_current_index) % _size;
    while (!atomic_lock(_locks[index])) {
      index = atomic_post_increment(_current_index) % _size;
    }
    atomic_pre_increment(_number_taken);
    return index;
  }

  /**
   * @brief Free the element with the given index.
   *
   * The element can be overwritten after this method has been called.
   *
   * @param index Index of an element that was in use.
   */
  inline void free_element(const size_t index) {
    myassert(_locks[index], "Element not in use!");
    atomic_unlock(_locks[index]);
    atomic_pre_decrement(_number_taken);
  }

  /**
   * @brief Get the size of the vector.
   *
   * Only makes sense if the vector is continuous, i.e. non of the elements has
   * ever been removed.
   *
   * @return Number of used elements in the vector.
   */
  inline const size_t size() const {
    myassert(_number_taken == _current_index, "Non continuous vector!");
    return _number_taken;
  }
};

#endif // THREADSAFEVECTOR_HPP
