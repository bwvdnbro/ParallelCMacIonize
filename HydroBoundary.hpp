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
 * @brief Inflow hydro boundary.
 */
class InflowHydroBoundary {
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
  inline static void get_right_state_gradient_variables(const int i,
                                                        const float WL[5],
                                                        float WR[5]) {

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
  inline static void get_right_state_flux_variables(const int i,
                                                    const float WL[5],
                                                    const float dWL[15],
                                                    float WR[5],
                                                    float dWR[15]) {

    WR[0] = WL[0];
    WR[1] = WL[1];
    WR[2] = WL[2];
    WR[3] = WL[3];
    WR[4] = WL[4];

#ifdef SECOND_ORDER
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
#endif
  }
};

/**
 * @brief Reflective hydro boundary.
 */
class ReflectiveHydroBoundary {
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
  inline static void get_right_state_gradient_variables(const int i,
                                                        const float WL[5],
                                                        float WR[5]) {

    // indices in a frame where the x-axis is aligned with the interface normal
    const int idx = i;
    const int idy = (i + 1) % 3;
    const int idz = (i + 2) % 3;

    // we need to reverse the sign for the velocity component aligned with the
    // interface normal: idx
    WR[0] = WL[0];
    WR[idx + 1] = -WL[idx + 1];
    WR[idy + 1] = WL[idy + 1];
    WR[idz + 1] = WL[idz + 1];
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
  inline static void get_right_state_flux_variables(const int i,
                                                    const float WL[5],
                                                    const float dWL[15],
                                                    float WR[5],
                                                    float dWR[15]) {

    // indices in a frame where the x-axis is aligned with the interface normal
    const int idx = i;
    const int idy = (i + 1) % 3;
    const int idz = (i + 2) % 3;
#ifdef SECOND_ORDER
    // offsets of the velocity variables in the same reference frame
    const int vdx = 3 * idx + 3;
    const int vdy = 3 * idy + 3;
    const int vdz = 3 * idz + 3;
#endif

    // we need to reverse the sign for the velocity component aligned with the
    // interface normal: idx
    WR[0] = WL[0];
    WR[idx + 1] = -WL[idx + 1];
    WR[idy + 1] = WL[idy + 1];
    WR[idz + 1] = WL[idz + 1];
    WR[4] = WL[4];

#ifdef SECOND_ORDER
    // we need to invert all gradients that are aligned with the interface
    // normal: idx
    // however, we do not invert the gradient of the velocity aligned with the
    // interface normal (idx + vdx), as the velocity itself also changes sign
    dWR[idx] = -dWL[idx];
    dWR[idy] = dWL[idy];
    dWR[idz] = dWL[idz];
    dWR[idx + vdx] = dWL[idx + vdx];
    dWR[idy + vdx] = dWL[idy + vdx];
    dWR[idz + vdx] = dWL[idz + vdx];
    dWR[idx + vdy] = -dWL[idx + vdy];
    dWR[idy + vdy] = dWL[idy + vdy];
    dWR[idz + vdy] = dWL[idz + vdy];
    dWR[idx + vdz] = -dWL[idx + vdz];
    dWR[idy + vdz] = dWL[idy + vdz];
    dWR[idz + vdz] = dWL[idz + vdz];
    dWR[idx + 12] = -dWL[idx + 12];
    dWR[idy + 12] = dWL[idy + 12];
    dWR[idz + 12] = dWL[idz + 12];
#endif
  }
};

#endif // HYDROBOUNDARY_HPP
