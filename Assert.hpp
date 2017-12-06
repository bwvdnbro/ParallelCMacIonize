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
 * @file Assert.hpp
 *
 * @brief Custom assertion macro.
 *
 * @author Bert Vandenbroucke (bv7@st-andrews.ac.uk)
 */
#ifndef ASSERT_HPP
#define ASSERT_HPP

/*! @brief Uncomment this to enable assertions. */
#define DO_ASSERTS

#include <cstdlib>
#include <iostream>

/**
 * @brief Custom assertion macro that supports extra output.
 *
 * @param condition Condition to check.
 * @param message Message to print if the assertion fails.
 */
#ifdef DO_ASSERTS
#define myassert(condition, message)                                           \
  if (!(condition)) {                                                          \
    std::cerr << __FILE__ << ":" << __FUNCTION__ << "():" << __LINE__ << ":"   \
              << std::endl;                                                    \
    std::cerr << "Assertion failed: " #condition << std::endl;                 \
    std::cerr << message << std::endl;                                         \
    abort();                                                                   \
  }
#else
#define myassert(condition, message)
#endif

#endif // ASSERT_HPP
