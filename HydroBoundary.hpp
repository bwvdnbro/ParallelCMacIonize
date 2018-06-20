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
 * @file HydroBoundary.hpp
 *
 * @brief Hydro boundary conditions related functionality.
 *
 * @author Bert Vandenbroucke (bv7@st-andrews.ac.uk)
 */
#ifndef HYDROBOUNDARY_HPP
#define HYDROBOUNDARY_HPP

/**
 * @brief Hydro boundary condition related functionality.
 */
class HydroBoundary {
public:
  /**
   * @brief Get the right state primitive variables corresponding to the given
   * left state boundary ghost.
   *
   * @param i Interface direction: x (0), y (1) or z (2).
   * @param WL Left state primitive variables (density - kg m^-3, velocity -
   * m s^-1, pressure - kg m^-1 s^-2).
   * @param WR Output right state primitive variables (density - kg m^-3,
   * velocity - m s^-1, pressure - kg m^-1 s^-2).
   */
  inline void get_right_state_gradient_variables(const int i,
                                                 const double WL[5],
                                                 double WR[5]) const {

    WR[0] = WL[0];
    WR[1] = WL[1];
    WR[2] = WL[2];
    WR[3] = WL[3];
    WR[4] = WL[4];
  }

  /**
   * @brief Get the right state primitive variables and gradients corresponding
   * to the given left state boundary ghost.
   *
   * @param i Interface direction: x (0), y (1) or z (2).
   * @param WL Left state primitive variables (density - kg m^-3, velocity -
   * m s^-1, pressure - kg m^-1 s^-2).
   * @param dWL Left state primitive variable gradients (density - kg m^-4,
   * velocity - s^-1, pressure - kg m^-2 s^-2).
   * @param WR Output right state primitive variables (density - kg m^-3,
   * velocity - m s^-1, pressure - kg m^-1 s^-2).
   * @param dWR Output right state primitive variable gradients (density -
   * kg m^-4, velocity - s^-1, pressure - kg m^-2 s^-2).
   */
  inline void get_right_state_flux_variables(const int i, const double WL[5],
                                             const double dWL[15], double WR[5],
                                             double dWR[15]) const {

    WR[0] = WL[0];
    WR[1] = WL[1];
    WR[2] = WL[2];
    WR[3] = WL[3];
    WR[4] = WL[4];

    dWR[0] = dWL[0];
    dWR[1] = dWL[1];
    dWR[2] = dWL[2];
    dWR[3] = dWL[3];
    dWR[4] = dWL[4];
    dWR[5] = dWL[5];
    dWR[6] = dWL[6];
    dWR[7] = dWL[7];
    dWR[8] = dWL[8];
    dWR[9] = dWL[9];
    dWR[10] = dWL[10];
    dWR[11] = dWL[11];
    dWR[12] = dWL[12];
    dWR[13] = dWL[13];
    dWR[14] = dWL[14];
  }
};

#endif // HYDROBOUNDARY_HPP
