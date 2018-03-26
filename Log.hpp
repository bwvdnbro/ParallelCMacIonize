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
 * @file Log.hpp
 *
 * @brief Logging routines.
 *
 * @author Bert Vandenbroucke (bv7@st-andrews.ac.uk)
 */
#ifndef LOG_HPP
#define LOG_HPP

/**
 * @brief Write a message to the log with the given log level.
 *
 * @param message Message to write.
 * @param loglevel Log level. The message is only written if the LOG_OUTPUT
 * defined is higher than this value.
 */
#ifdef LOG_OUTPUT
#define logmessage(message, loglevel)                                          \
  if (MPI_rank == 0 && loglevel < LOG_OUTPUT) {                                \
    _Pragma("omp single") { std::cerr << message << std::endl; }               \
  }
#else
#define logmessage(message, loglevel)
#endif

/**
 * @brief Write a message to the log with the given log level, irrespective of
 * the rank of the process.
 *
 * @param message Message to write.
 * @param loglevel Log level. The message is only written if the LOG_OUTPUT
 * defined is higher than this value.
 */
#ifdef LOG_OUTPUT
#define logmessage_always(message, loglevel)                                   \
  if (loglevel < LOG_OUTPUT) {                                                 \
    _Pragma("omp single") {                                                    \
      std::cerr << "[rank " << MPI_rank << "]: " << message << std::endl;      \
    }                                                                          \
  }
#else
#define logmessage_always(message, loglevel)
#endif

/**
 * @brief Write a message to the log with the given log level from every
 * process.
 *
 * Note that this macro should only be used in parts of the code that are
 * executed simultaneously by all processes by a single thread.
 *
 * @param message Message to write.
 * @param loglevel Log level. The message is only written if the LOG_OUTPUT
 * defined is higher than this value.
 */
#ifdef LOG_OUTPUT
#define logmessage_all(message, loglevel)                                      \
  if (loglevel < LOG_OUTPUT) {                                                 \
    for (int irank = 0; irank < MPI_size; ++irank) {                           \
      if (irank == MPI_rank) {                                                 \
        std::cerr << "[rank " << irank << "]: " << message << std::endl;       \
      }                                                                        \
      MPI_Barrier(MPI_COMM_WORLD);                                             \
    }                                                                          \
  }
#else
#define logmessage_all(message, loglevel) (void)
#endif

#endif // LOG_HPP
