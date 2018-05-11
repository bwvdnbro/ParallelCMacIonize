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
 * @file MPIBuffer.hpp
 *
 * @brief MPI communication buffer.
 *
 * @author Bert Vandenbroucke (bv7@st-andrews.ac.uk)
 */
#ifndef MPIBUFFER_HPP
#define MPIBUFFER_HPP

#include <mpi.h>
#include <vector>

/**
 * @brief MPI communication buffer.
 */
class MPIBuffer {
private:
  /*! @brief Actual memory buffer. */
  char *_buffer;

  /*! @brief Memory buffer size. */
  const size_t _buffer_size;

  /*! @brief Element size. */
  const size_t _element_size;

  /*! @brief Requests array. */
  MPI_Request *_requests;

  /*! @brief Requests array size. */
  const size_t _requests_size;

  /*! @brief Index of the last request that was used. */
  size_t _last_index;

  /*! @brief Number of elements that is currently in use. */
  size_t _number_in_use;

#ifdef MPIBUFFER_STATS
  /*! @brief Maximum number of elements taken at any given point. */
  size_t _max_number_in_use;
#endif

public:
  /**
   * @brief Constructor.
   *
   * @param size Number of elements that will be stored in the buffer.
   * @param element_size Size of a single element in the buffer (in bytes).
   */
  inline MPIBuffer(const size_t size, const size_t element_size)
      : _buffer(nullptr), _buffer_size(size * element_size),
        _element_size(element_size), _requests(nullptr), _requests_size(size),
        _last_index(0), _number_in_use(0) {

    if (_buffer_size > 0) {
      _buffer = new char[_buffer_size];
    }

    if (_requests_size > 0) {
      _requests = new MPI_Request[_requests_size];
      for (size_t i = 0; i < _requests_size; ++i) {
        _requests[i] = MPI_REQUEST_NULL;
      }
    }

#ifdef MPIBUFFER_STATS
    _max_number_in_use = 0;
#endif
  }

  /**
   * @brief Destructor.
   *
   * Free the buffer and requests array.
   */
  inline ~MPIBuffer() {
    if (_buffer != nullptr) {
      delete[] _buffer;
    }
    if (_requests != nullptr) {
      delete[] _requests;
    }
  }

  /**
   * @brief Get the size of the buffer.
   *
   * @return Size of the buffer (in bytes).
   */
  inline size_t size() const { return _buffer_size; }

  /**
   * @brief Access the buffer at the given offset.
   *
   * @param index Offset in the buffer (in bytes).
   * @return Memory address of that location in the buffer.
   */
  inline char *operator[](const size_t index) { return &_buffer[index]; }

  /**
   * @brief Wait for all communications involving the buffer to finish.
   */
  inline void wait_for_communications() {
    MPI_Waitall(_requests_size, _requests, MPI_STATUSES_IGNORE);

    // check that all requests finished (should be guaranteed by the MPI_Waitall
    // call above)
    for (size_t i = 0; i < _requests_size; ++i) {
      myassert(_requests[i] == MPI_REQUEST_NULL,
               "Not all communications were finished!");
    }

    _number_in_use = 0;
  }

  /**
   * @brief Get the index of an element in the buffer with a free MPI request.
   *
   * @return Index of a free spot in the MPI buffer.
   */
  inline size_t get_free_element() {

    myassert(_number_in_use < _requests_size, "No more free MPI requests!");

    _last_index = (_last_index + 1) % _requests_size;
    while (_requests[_last_index] != MPI_REQUEST_NULL) {
      _last_index = (_last_index + 1) % _requests_size;
    }
    ++_number_in_use;

#ifdef MPIBUFFER_STATS
    _max_number_in_use = std::max(_max_number_in_use, _number_in_use);
#endif

    return _last_index;
  }

  /**
   * @brief Get a reference to the MPI_Request for the given index.
   *
   * @param index Index of a buffer element.
   * @return Reference to the corresponding MPI_Request.
   */
  inline MPI_Request &get_request(const size_t index) {
    return _requests[index];
  }

  /**
   * @brief Get a pointer to the given buffer element.
   *
   * @param index Index of a buffer element.
   * @return Pointer to the buffer element.
   */
  inline char *get_element(const size_t index) {
    return &_buffer[index * _element_size];
  }

  /**
   * @brief Perform a non-blocking check of the active MPI_Requests for the
   * buffer, and free at most one element for which the communication finished.
   */
  inline void check_finished_communications() {
    int index, flag;
    MPI_Testany(_requests_size, _requests, &index, &flag, MPI_STATUS_IGNORE);
    // we cannot test flag, since flag will also be true if the array only
    // contains MPI_REQUEST_NULL values
    if (index != MPI_UNDEFINED) {
      myassert(_number_in_use > 0, "Negative request counter!");
      --_number_in_use;
    }
  }

  /**
   * @brief Reset the buffer.
   */
  inline void reset() {
    _last_index = 0;
    _number_in_use = 0;

#ifdef MPIBUFFER_STATS
    _max_number_in_use = 0;
#endif
  }

  /**
   * @brief Get the size in memory of the MPI buffer.
   *
   * @return Size in memory of the MPI buffer (in bytes).
   */
  inline size_t get_memory_size() const {
    return sizeof(MPIBuffer) + _buffer_size +
           _requests_size * sizeof(MPI_Request);
  }

/**
 * @brief Get the maximum number of elements that was in use at any given time.
 *
 * @return Maximum number of elements that was in use.
 */
#ifdef MPIBUFFER_STATS
  inline size_t get_max_number_in_use() const { return _max_number_in_use; }
#else
#error "This function should only be used when MPIBUFFER_STATS is defined!"
#endif

/**
 * @brief Reset the counter for the maximum number of elements that was in use.
 */
#ifdef MPIBUFFER_STATS
  inline void reset_max_number_in_use() { _max_number_in_use = 0; }
#else
#error "This function should only be used when MPIBUFFER_STATS is defined!"
#endif
};

#endif // MPIBUFFER_HPP
