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
 * @file Atomic.hpp
 *
 * @brief Atomic operations.
 *
 * @author Bert Vandenbroucke (bv7@st-andrews.ac.uk)
 */
#ifndef ATOMIC_HPP
#define ATOMIC_HPP

/*! @brief Activate this to use standard C++11 atomic operations. */
#define CPP_ATOMIC

/*! @brief Activate this to use GNU GCC specific atomic directives. */
//#define GCC_ATOMIC

#ifdef CPP_ATOMIC
#include <atomic>
#elif GCC_ATOMIC
// nothing to include
#else
#error "No atomic implementation chosen!"
#endif

/**
 * @brief Atomic variable.
 */
template < typename _type_ > class Atomic {
private:
/*! @brief Underlying value. */
#ifdef CPP_ATOMIC
  std::atomic< _type_ > _value;
#else
  volatile _type_ _value;
#endif

public:
  /**
   * @brief Empty constructor.
   */
  inline Atomic() : _value(0) {}

  /**
   * @brief Constructor.
   *
   * @param value Value.
   */
  inline Atomic(_type_ value) : _value(value) {}

  /**
   * @brief Read the value of the atomic variable.
   *
   * @return Current value of the variable.
   */
  inline const _type_ value() const {
#ifdef CPP_ATOMIC
    return _value.load();
#else
    return _value;
#endif
  }

  /**
   * @brief Set the value of the atomic variable.
   *
   * @param value New value for the variable.
   */
  inline void set(const _type_ value) {
#ifdef CPP_ATOMIC
    _value.store(value);
#else
    _value = value;
#endif
  }

  /**
   * @brief Lock the value atomically, making sure only one thread is allowed to
   * set it.
   *
   * @return True if the flag was set, False if it could not be set, meaning the
   * flag has already been set by another thread.
   */
  inline bool lock() {
#ifdef CPP_ATOMIC
    _type_ expected(false);
    return _value.compare_exchange_strong(expected, true);
#else
    return __sync_bool_compare_and_swap(&_value, false, true);
#endif
  }

  /**
   * @brief Unlock the value atomically, making sure only one thread is allowed
   * to set it.
   */
  inline void unlock() {
#ifdef CPP_ATOMIC
    _type_ expected(true);
    _value.compare_exchange_strong(expected, false);
#else
    __sync_bool_compare_and_swap(&_value, true, false);
#endif
  }

  /**
   * @brief Atomically increment the variable by 1 and return the original
   * value.
   *
   * @return Original value of the variable.
   */
  inline _type_ post_increment() {
#ifdef CPP_ATOMIC
    return _value++;
#else
    return __sync_fetch_and_add(&_value, 1);
#endif
  }

  /**
   * @brief Atomically increment the variable by 1 and return the new value.
   *
   * @return New value of the variable.
   */
  inline _type_ pre_increment() {
#ifdef CPP_ATOMIC
    return ++_value;
#else
    return __sync_add_and_fetch(&_value, 1);
#endif
  }

  /**
   * @brief Atomically decrement the variable by 1 and return the new value.
   *
   * @return New value of the variable.
   */
  inline _type_ pre_decrement() {
#ifdef CPP_ATOMIC
    return --_value;
#else
    return __sync_sub_and_fetch(&_value, 1);
#endif
  }

  /**
   * @brief Atomically add the given value to the variable and return the old
   * value of the variable.
   *
   * @param increment Value to add to the variable.
   * @return Original value of the variable.
   */
  inline _type_ post_add(const _type_ increment) {
#ifdef CPP_ATOMIC
    return (_value += increment);
#else
    return __sync_fetch_and_add(&_value, increment);
#endif
  }

  /**
   * @brief Atomically add the given value to the variable and return the new
   * value of the variable.
   *
   * @param increment Value to add to the variable.
   * @return New value of the variable.
   */
  inline _type_ pre_add(const _type_ increment) {
#ifdef CPP_ATOMIC
    return (_value += increment) + increment;
#else
    return __sync_add_and_fetch(&_value, increment);
#endif
  }
};

#endif // ATOMIC_HPP
