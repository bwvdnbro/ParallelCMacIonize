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
 * @file Hydro.hpp
 *
 * @brief Hydro related functionality.
 *
 * @author Bert Vandenbroucke (bv7@st-andrews.ac.uk)
 */
#ifndef HYDRO_HPP
#define HYDRO_HPP

#include "HLLCRiemannSolver.hpp"

/**
 * @brief Hydro related functionality.
 */
class Hydro {
private:
  /*! @brief Polytropic index @f$\gamma{}@f$ of the gas. */
  const double _gamma;

  /*! @brief @f$\gamma{}-1@f$. */
  const double _gamma_minus_one;

  /*! @brief @f$\frac{1}{\gamma{}-1}@f$. */
  const double _one_over_gamma_minus_one;

  /*! @brief Riemann solver used to solve the Riemann problem. */
  const HLLCRiemannSolver _riemann_solver;

public:
  /**
   * @brief Constructor.
   *
   * @param gamma Polytropic index @f$\gamma{}@f$ of the gas.
   */
  inline Hydro(const double gamma = 5. / 3.)
      : _gamma(gamma), _gamma_minus_one(_gamma - 1.),
        _one_over_gamma_minus_one(1. / _gamma_minus_one),
        _riemann_solver(gamma) {}

  /**
   * @brief Get the soundspeed for the given density and pressure.
   *
   * @param density Density (in kg m^-3).
   * @param pressure Pressure (in kg m^-1 s^-2).
   * @return Soundspeed (in m s^-1).
   */
  inline double get_soundspeed(const double density,
                               const double pressure) const {
    return std::sqrt(_gamma * pressure / density);
  }

  /**
   * @brief Transform the given conserved variables into primitive variables
   * using the given volume.
   *
   * @param mass Mass (in kg).
   * @param momentum Momentum (in kg m s^-1).
   * @param total_energy Total energy (in kg m^2 s^-2).
   * @param inverse_volume Inverse of the volume (in m^-3).
   * @param density Output density (in kg m^-3).
   * @param velocity Output velocity (in m s^-1).
   * @param pressure Output pressure (in kg m^-1 s^-2).
   */
  inline void get_primitive_variables(const double mass,
                                      const double momentum[3],
                                      const double total_energy,
                                      const double inverse_volume,
                                      double &density, double velocity[3],
                                      double &pressure) const {

    const double inverse_mass = 1. / mass;

    density = mass * inverse_volume;
    velocity[0] = momentum[0] * inverse_mass;
    velocity[1] = momentum[1] * inverse_mass;
    velocity[2] = momentum[2] * inverse_mass;
    pressure = _gamma_minus_one * inverse_volume *
               (total_energy -
                0.5 * (velocity[0] * momentum[0] + velocity[1] * momentum[1] +
                       velocity[2] * momentum[2]));
  }

  /**
   * @brief Transform the given primitive variables into conserved variables
   * using the given volume.
   *
   * @param density Density (in kg m^-3).
   * @param velocity Velocity (in m s^-1).
   * @param pressure Pressure (in kg m^-1 s^-2).
   * @param volume Volume (in m^-3).
   * @param mass Output mass (in kg).
   * @param momentum Output momentum (in kg m s^-1).
   * @param total_energy Output total energy (in kg m^2 s^-2).
   */
  inline void get_conserved_variables(const double density,
                                      const double velocity[3],
                                      const double pressure,
                                      const double volume, double &mass,
                                      double momentum[3],
                                      double &total_energy) const {

    mass = density * volume;
    momentum[0] = velocity[0] * mass;
    momentum[1] = velocity[1] * mass;
    momentum[2] = velocity[2] * mass;
    total_energy =
        _one_over_gamma_minus_one * pressure * volume +
        0.5 * (velocity[0] * momentum[0] + velocity[1] * momentum[1] +
               velocity[2] * momentum[2]);
  }
};

#endif // HYDRO_HPP
