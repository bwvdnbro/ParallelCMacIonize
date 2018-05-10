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

public:
  /**
   * @brief Constructor.
   *
   * @param buffer_size Size of the memory buffer (in bytes).
   * @param element_size Size of a single element in the buffer (in bytes).
   */
  inline MPIBuffer(const size_t buffer_size, const size_t element_size)
      : _buffer(nullptr), _buffer_size(buffer_size),
        _element_size(element_size), _requests(nullptr),
        _requests_size(buffer_size / element_size), _last_index(0) {

    if (_buffer_size > 0) {
      _buffer = new char[_buffer_size];
    }

    if (_requests_size > 0) {
      _requests = new MPI_Request[_requests_size];
      for (size_t i = 0; i < _requests_size; ++i) {
        _requests[i] = MPI_REQUEST_NULL;
      }
    }
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
  }

  /**
   * @brief Get the index of an element in the buffer with a free MPI request.
   *
   * @return Index of a free spot in the MPI buffer.
   */
  inline size_t get_free_element() {
    size_t request_index = _last_index;
    while (_requests[request_index] != MPI_REQUEST_NULL) {
      request_index = (request_index + 1) % _requests_size;
      myassert(request_index != _last_index,
               "Unable to obtain a free MPI request!");
    }
    _last_index = request_index;
    return request_index;
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
      // release the request, this will automatically release the
      // corresponding space in the buffer
      _requests[index] = MPI_REQUEST_NULL;
    }
  }

  /**
   * @brief Reset the buffer.
   */
  inline void reset() { _last_index = 0; }
};

#endif // MPIBUFFER_HPP
