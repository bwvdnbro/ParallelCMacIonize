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
 * @file MemoryMap.hpp
 *
 * @brief Memory-mapped output file that can be accessed by multiple threads
 * simultaneously.
 *
 * @author Bert Vandenbroucke (bv7@st-andrews.ac.uk)
 */
#ifndef MEMORYMAP_HPP
#define MEMORYMAP_HPP

#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <string>
#include <sys/mman.h>
#include <unistd.h>

/**
 * @brief Memory-mapped output file that can be accessed by multiple threads
 * simultaneously.
 */
class MemoryMap {
private:
  /*! @brief The memory-mapped buffer. */
  char *_memory_buffer;

  /*! @brief The file. */
  int _file;

  /*! @brief Size of the file (in bytes). */
  const size_t _size;

  /*! @brief Mask used to ensure sizes and offsets that are multiples of the
   *  system page size. */
  const size_t _page_mask;

  /*! @brief The size of the memory-mapped data, in bytes. */
  const size_t _memory_buffer_size;

  /**
   * @brief Get the page size mask that can be used to round sizes and offsets
   * to page size multiples.
   *
   * To round a variable `s` up, use
   * ```
   *  s = (s + ~page_mask) & page_mask;
   * ```
   *
   * To round down, use
   * ```
   *  s &= page_mask;
   * ```
   *
   * Examples (we assume a 16-bit mask for simplicity, page size is \f$PS =
   * 4096\f$, page mask is \f$PM = 1111~0000~0000~0000\f$):
   *  - \f$s = 5 = 101\f$:
   *      \f[
   *        (s + \sim{}PM) \& PM
   *          = (101 + 0000~1111~1111~1111) & 1111~0000~0000~0000
   *          = 0001~0000~0000~0100 & 1111~0000~0000~0000
   *          = 0001~0000~0000~0000
   *      \f]
   *  - \f$s = PM + 2\f$:
   *      \f[
   *        (s + \sim{}PM) \& PM
   *          = (10 + 0001~0000~0000~0000 + 0000~1111~1111~1111) &
   *            1111~0000~0000~0000
   *          = 0010~0000~0000~0100 & 1111~0000~0000~0000
   *          = 0010~0000~0000~0000
   *      \f]
   *
   * @return Page mask that can be used to round sizes and offsets to page size
   * multiples.
   */
  static inline size_t get_page_mask() { return ~(sysconf(_SC_PAGE_SIZE) - 1); }

  /**
   * @brief Round the given size up to a multiple of the page size.
   *
   * @param size Size to round up.
   */
  inline size_t round_page_up(const size_t size) const {
    return (size + ~_page_mask) & _page_mask;
  }

  /**
   * @brief Round the given size down to a multiple of the page size.
   *
   * @param size Size to round down.
   */
  inline size_t round_page_down(const size_t size) const {
    return size & _page_mask;
  }

public:
  /**
   * @brief Constructor.
   *
   * Creates a file with the given file name and size, and memory-maps it
   * for fast write access.
   *
   * @param filename Name of the file.
   * @param size Size of the buffer, in bytes. The same size will be used for
   * the internal file buffer that is stored in memory (rounded up to the
   * nearest multiple of the system page size).
   */
  inline MemoryMap(const std::string filename, const size_t size)
      : _memory_buffer(nullptr), _file(0), _size(size),
        _page_mask(get_page_mask()), _memory_buffer_size(round_page_up(size)) {

    // create the file
    // O_CREAT: create the file if it does not exist
    // O_RDWR: read/write access
    // S_IRUSR/IWUSR/IRGRP/IWGRP: set read/write access for both the user and
    //  group when creating the file (code 0660 in the SWIFT code)
    _file = open(filename.c_str(), O_CREAT | O_RDWR,
                 S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP);
    if (_file < 0) {
      std::cerr << "Error opening file!" << std::endl;
      abort();
    }
    // make sure we have enough space on disk to store the memory mapped file
    // this preallocates the file size without actually writing anything
    // we tell the call we want all file offsets between 0 and size to be
    // available after this call
    if (posix_fallocate(_file, 0, _memory_buffer_size) != 0) {
      std::cerr << "Error reserving file disk space!" << std::endl;
      abort();
    }
    // memory-map the file: make a synced copy of some part of the file in
    // active memory that we can directly access as if it was normal memory
    // parameters:
    //  address: let the system decide where to allocate the memory
    //  size: size of the buffer we want to memory-map (should be a multiple of
    //   page_size!)
    //  protection: we want to write to the buffer
    //  mode: we want to share the memory with all other processes: every write
    //   to this region is synced in the file and directly reflected in all
    //   other processes that memory map the same region
    //  file: this is the file we want to memory-map
    //  offset: we want to start memory mapping from offset 0 in the file
    _memory_buffer = reinterpret_cast< char * >(
        mmap(nullptr, _memory_buffer_size, PROT_WRITE, MAP_SHARED, _file, 0));
    if (_memory_buffer == MAP_FAILED) {
      std::cerr << "Error memory mapping file!" << std::endl;
      abort();
    }
  }

  /**
   * @brief Close the log file.
   *
   * Unmaps the memory-mapped part of the file (forcing it to be written to the
   * file) and closes the log file.
   */
  inline void close_file() {
    // unmap the actively memory-mapped region of the file
    if (munmap(_memory_buffer, _memory_buffer_size) != 0) {
      std::cerr << "Error unmapping file memory!" << std::endl;
      abort();
    }
    _memory_buffer = nullptr;
    // shrink the file to its actual size
    if (ftruncate(_file, _size) != 0) {
      std::cerr << "Error shrinking file!" << std::endl;
      abort();
    }
    // close the file
    if (close(_file) != 0) {
      std::cerr << "Error closing file!" << std::endl;
      abort();
    }
    _file = 0;
  }

  /**
   * @brief Destructor.
   */
  inline ~MemoryMap() {
    if (_file != 0) {
      close_file();
    }
  }

  /**
   * @brief Make sure all contents in the log file are actually written to disk.
   */
  inline void flush() {
    // make sure the memory-mapped region is synced on disk
    // parameters:
    //  address: pointer to the beginning of the region we want synced
    //  size: number of bytes after that pointer we want to be synced
    //  MS_SYNC: force actual writing to disk and wait until it is finished
    msync(_memory_buffer, _size, MS_SYNC);
  }

  /**
   * @brief Write the given value to the file at the given position.
   *
   * @param value Value to write.
   */
  template < typename _type_ >
  inline void write(const size_t position, const _type_ value) {
    const size_t vsize = sizeof(_type_);
    myassert(position < _memory_buffer_size,
             "Offset too large (" << position << ", " << _memory_buffer_size
                                  << ")!");
    memcpy(_memory_buffer + position, &value, vsize);
  }

  /**
   * @brief Get the size in memory of the memory map.
   *
   * @return Size in memory of the memory map (in bytes).
   */
  inline size_t get_memory_size() const { return sizeof(MemoryMap) + _size; }
};

#endif // MEMORYMAP_HPP
