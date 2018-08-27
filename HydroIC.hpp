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
 * @file HydroIC.hpp
 *
 * @brief Hydro initial conditions related functionality.
 *
 * @author Bert Vandenbroucke (bv7@st-andrews.ac.uk)
 */
#ifndef HYDROIC_HPP
#define HYDROIC_HPP

/**
 * @brief Sod shock tube initial condition.
 */
class SodShockHydroIC {
public:
  /**
   * @brief Set the primitive variables based on the given position.
   *
   * @param midpoint Cell midpoint position (in m).
   * @param W Output primitive variables (density - kg m^-3, velocity - m s^-1,
   * pressure - kg m^-1 s^-2).
   */
  inline static void set_primitive_variables(const double midpoint[3],
                                             float W[5]) {

    if (midpoint[0] < 0.) {
      W[0] = 1.;
      W[4] = 1.;
    } else {
      W[0] = 0.125;
      W[4] = 0.1;
    }
    W[1] = 0.;
    W[2] = 0.;
    W[3] = 0.;
  }
};

#endif // HYDROIC_HPP
