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
  /*! @brief Request for global communication. */
  MPIMESSAGETAG_REDUCE_REQUEST,
  /*! @brief Global stop message. */
  MPIMESSAGETAG_STOP
};

/**
 * @brief Log an MPI send event.
 *
 * @param message Message field to write the log to.
 * @param destination Destination rank of the send.
 * @param tag Associated tag.
 */
#define log_send(message, destination, tag)                                    \
  {                                                                            \
    message._type = MPIMESSAGETYPE_SEND;                                       \
    message._rank = destination;                                               \
    message._tag = tag;                                                        \
    task_tick(message._timestamp);                                             \
  }

/**
 * @brief Log an MPI receive event.
 *
 * @param message Message field to write the log to.
 * @param source Source rank of the communication.
 * @param tag Associated tag.
 */
#define log_recv(message, source, tag)                                         \
  {                                                                            \
    message._type = MPIMESSAGETYPE_RECV;                                       \
    message._rank = source;                                                    \
    message._tag = tag;                                                        \
    task_tick(message._timestamp);                                             \
  }

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
public:
  /*! @brief Type of message. */
  MPIMessageType _type;

  /*! @brief Origin/destination rank of the message. */
  int _rank;

  /*! @brief Message tag. */
  int _tag;

  /*! @brief Time stamp. */
  unsigned long _timestamp;
};

#endif // MPIMESSAGE_HPP
