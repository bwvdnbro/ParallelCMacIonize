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
 * @file MPIMessage.hpp
 *
 * @brief MPI message.
 *
 * @author Bert Vandenbroucke (bv7@st-andrews.ac.uk)
 */
#ifndef MPIMESSAGE_HPP
#define MPIMESSAGE_HPP

#include "Task.hpp"

/**
 * @brief MPI message tags that are supported.
 */
enum MPIMessageTag {
  /*! @brief Photonbuffer communication. */
  MPIMESSAGETAG_PHOTONBUFFER = 0,
  /*! @brief End of local process message. */
  MPIMESSAGETAG_LOCAL_PROCESS_FINISHED,
  /*! @brief Global stop message. */
  MPIMESSAGETAG_STOP
};

/**
 * @brief MPI message events that are logged.
 */
enum MPIMessageType {
  /*! @brief Send event. */
  MPIMESSAGETYPE_SEND = 0,
  /*! @brief Receive event. */
  MPIMESSAGETYPE_RECV
};

/**
 * @brief MPI message log information.
 */
class MPIMessage {
private:
  /*! @brief Type of message. */
  MPIMessageType _type;

  /*! @brief Origin/destination rank of the message. */
  int _rank;

  /*! @brief Thread that did the communication. */
  int _thread;

  /*! @brief Message tag. */
  int _tag;

  /*! @brief Time stamp. */
  unsigned long _timestamp;

public:
  /**
   * @brief Log an MPI event.
   *
   * @param type Event type.
   * @param rank Rank on the other side of the communication.
   * @param thread Thread that does the communication.
   * @param tag Associated tag.
   */
  inline void log_event(const MPIMessageType type, const int rank,
                        const int thread, const int tag) {
    _type = type;
    _rank = rank;
    _thread = thread;
    _tag = tag;
    cpucycle_tick(_timestamp);
  }

  /**
   * @brief Get the information stored in the message for output purposes.
   *
   * @param type Variable to store the message type in.
   * @param rank Variable to store the message rank in.
   * @param thread Variable to store the message thread id in.
   * @param tag Variable to store the message tag in.
   * @param timestamp Variable to store the message timestamp in.
   */
  inline void get_output_info(int &type, int &rank, int &thread, int &tag,
                              unsigned long &timestamp) const {
    type = _type;
    rank = _rank;
    thread = _thread;
    tag = _tag;
    timestamp = _timestamp;
  }
};

#endif // MPIMESSAGE_HPP
