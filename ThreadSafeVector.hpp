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
template < typename _datatype_ > class ThreadSafeVector {
private:
  /*! @brief Current active index in the vector. */
  Atomic< size_t > _current_index;

  /*! @brief Size of the vector. */
  const size_t _size;

  /*! @brief Number of elements that have been taken. */
  Atomic< size_t > _number_taken;

  /*! @brief Vector itself. */
  _datatype_ *_vector;

  /*! @brief Atomic locks for the vector elements. */
  Atomic< bool > *_locks;

public:
  /**
   * @brief Constructor.
   *
   * @param size Size of the vector.
   */
  inline ThreadSafeVector(const size_t size)
      : _current_index(0), _size(size), _number_taken(0) {
    _vector = new _datatype_[size];
    // Atomic values are automatically initialized to 0 (= false) by the
    // constructor
    _locks = new Atomic< bool >[size];
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
      _locks[i].unlock();
    }
    _number_taken.set(0);
    _current_index.set(0);

    // clear the elements
    delete[] _vector;
    _vector = new _datatype_[_size];
  }

  /**
   * @brief Access the element with the given index.
   *
   * @param index Index of an element.
   * @return Read/write reference to the element with that index.
   */
  inline _datatype_ &operator[](const size_t index) {
    myassert(index < _size, "Element out of range (index: "
                                << index << ", size: " << _size << ")!");
    myassert(_locks[index].value(), "Element not in use!");
    return _vector[index];
  }

  /**
   * @brief Read-only access to the element with the given index.
   *
   * @param index Index of an element.
   * @return Read-only reference to the element with that index.
   */
  inline const _datatype_ &operator[](const size_t index) const {
    myassert(index < _size, "Element out of range (index: "
                                << index << ", size: " << _size << ")!");
    myassert(_locks[index].value(), "Element not in use!");
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
    myassert(_number_taken.value() < _size, "No more free elements in vector!");
    size_t index = _current_index.post_increment() % _size;
    while (!_locks[index].lock()) {
      index = _current_index.post_increment() % _size;
    }
    _number_taken.pre_increment();
    return index;
  }

  /**
   * @brief Get the index of a free element in the vector, if available.
   *
   * This element will be locked and needs to be freed later by calling
   * free_element(). If no free element is available, this function returns
   * max_size().
   *
   * @return Index of a free element.
   */
  inline size_t get_free_element_safe() {
    if (_number_taken.value() < _size) {
      size_t index = _current_index.post_increment() % _size;
      while (!_locks[index].lock()) {
        index = _current_index.post_increment() % _size;
      }
      _number_taken.pre_increment();
      return index;
    } else {
      return _size;
    }
  }

  /**
   * @brief Free the element with the given index.
   *
   * The element can be overwritten after this method has been called.
   *
   * @param index Index of an element that was in use.
   */
  inline void free_element(const size_t index) {
    myassert(_locks[index].value(), "Element not in use!");
    _locks[index].unlock();
    _number_taken.pre_decrement();
  }

  /**
   * @brief Get the size of the vector.
   *
   * Only makes sense if the vector is continuous, i.e. non of the elements has
   * ever been removed.
   *
   * @return Number of used elements in the vector.
   */
  inline size_t size() const {
    myassert(_number_taken.value() == _current_index.value(),
             "Non continuous vector!");
    return _number_taken.value();
  }

  /**
   * @brief Maximum number of elements in the vector.
   *
   * @return Maximum number of elements that can be stored in the vector.
   */
  inline size_t max_size() const { return _size; }
};

#endif // THREADSAFEVECTOR_HPP
