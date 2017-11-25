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

/**
 * @brief Lock the given flag atomically, making sure only one thread is allowed
 * to set it.
 *
 * @param flag Flag to set.
 * @return True if the flag was set, False if it could not be set, meaning the
 * flag has already been set by another thread.
 */
#define atomic_lock(flag) __sync_bool_compare_and_swap(&flag, false, true)

/**
 * @brief Atomically tries to set the given variable to the given new value.
 *
 * @param variable Variable to set.
 * @param old_value Expected old value for the variable.
 * @param new_value New value for the variable.
 * @return True if the variable was successfully set.
 */
#define atomic_set(variable, old_value, new_value)                             \
  __sync_bool_compare_and_swap(&variable, old_value, new_value)

/**
 * @brief Atomically increment the given variable by 1 and return the original
 * value.
 *
 * @param variable Variable to increment.
 * @return Original value of the variable.
 */
#define atomic_post_increment(variable) __sync_fetch_and_add(&variable, 1)

#endif // ATOMIC_HPP
