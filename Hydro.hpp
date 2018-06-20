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
#include "HydroBoundary.hpp"

#include <cfloat>

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

  /**
   * @brief Do the flux calculation for the given interface.
   *
   * @param i Interface direction: x (0), y (1) or z (2).
   * @param WL Left state primitive variables (density - kg m^-3, velocity -
   * m s^-1, pressure - kg m^-1 s^-2).
   * @param dWL Left state primitive variable gradients (density - kg m^-4,
   * velocity - s^-1, pressure - kg m^-2 s^-2).
   * @param WR Right state primitive variables (density - kg m^-3, velocity -
   * m s^-1, pressure - kg m^-1 s^-2).
   * @param dWR Right state primitive variable gradients (density - kg m^-4,
   * velocity - s^-1, pressure - kg m^-2 s^-2).
   * @param dx Distance between left and right state midpoint (in m).
   * @param A Signed surface area of the interface (in m^2).
   * @param dQL Left state conserved variable changes (updated; mass - kg s^-1,
   * momentum - kg m s^-2, total energy - kg m^2 s^-3).
   * @param dQR Right state conserved variable changes (updated; mass - kg s^-1,
   * momentum - kg m s^-2, total energy - kg m^2 s^-3).
   */
  inline void do_flux_calculation(const int i, const double WL[5],
                                  const double dWL[15], const double WR[5],
                                  const double dWR[15], const double dx,
                                  const double A, double dQL[5],
                                  double dQR[5]) const {

    const double WLext[5] = {
        WL[0] + 0.5 * dWL[i] * dx, WL[1] + 0.5 * dWL[3 + i] * dx,
        WL[2] + 0.5 * dWL[6 + i] * dx, WL[3] + 0.5 * dWL[9 + i] * dx,
        WL[4] + 0.5 * dWL[12 + i] * dx};
    const double WRext[5] = {
        WR[0] - 0.5 * dWR[i] * dx, WR[1] - 0.5 * dWR[3 + i] * dx,
        WR[2] - 0.5 * dWR[6 + i] * dx, WR[3] - 0.5 * dWR[9 + i] * dx,
        WR[4] - 0.5 * dWR[12 + i] * dx};

    double mflux = 0.;
    double pflux[3] = {0., 0., 0.};
    double Eflux = 0.;
    double normal[3] = {0, 0, 0};
    normal[i] = 1.;
    _riemann_solver.solve_for_flux(WLext[0], &WLext[1], WLext[4], WRext[0],
                                   &WRext[1], WRext[4], mflux, pflux, Eflux,
                                   normal);

    mflux *= A;
    pflux[0] *= A;
    pflux[1] *= A;
    pflux[2] *= A;
    Eflux *= A;

    dQL[0] -= mflux;
    dQL[1] -= pflux[0];
    dQL[2] -= pflux[1];
    dQL[3] -= pflux[2];
    dQL[4] -= Eflux;

    dQR[0] += mflux;
    dQR[1] += pflux[0];
    dQR[2] += pflux[1];
    dQR[3] += pflux[2];
    dQR[4] += Eflux;
  }

  /**
   * @brief Do the flux calculation across a box boundary.
   *
   * @param i Interface direction: x (0), y (1) or z (2).
   * @param WL Left state primitive variables (density - kg m^-3, velocity -
   * m s^-1, pressure - kg m^-1 s^-2).
   * @param dWL Left state primitive variable gradients (density - kg m^-4,
   * velocity - s^-1, pressure - kg m^-2 s^-2).
   * @param boundary HydroBoundary that sets the right state variables.
   * @param dx Distance between left and right state midpoint (in m).
   * @param A Signed surface area of the interface (in m^2).
   * @param dQL Left state conserved variable changes (updated; mass - kg s^-1,
   * momentum - kg m s^-2, total energy - kg m^2 s^-3).
   */
  template < typename _boundary_ >
  inline void do_ghost_flux_calculation(const int i, const double WL[5],
                                        const double dWL[15],
                                        const _boundary_ &boundary,
                                        const double dx, const double A,
                                        double dQL[5]) const {

    double WR[5], dWR[15];
    boundary.get_right_state_flux_variables(i, WL, dWL, WR, dWR);

    const double WLext[5] = {
        WL[0] + 0.5 * dWL[i] * dx, WL[1] + 0.5 * dWL[3 + i] * dx,
        WL[2] + 0.5 * dWL[6 + i] * dx, WL[3] + 0.5 * dWL[9 + i] * dx,
        WL[4] + 0.5 * dWL[12 + i] * dx};
    const double WRext[5] = {
        WR[0] - 0.5 * dWR[i] * dx, WR[1] - 0.5 * dWR[3 + i] * dx,
        WR[2] - 0.5 * dWR[6 + i] * dx, WR[3] - 0.5 * dWR[9 + i] * dx,
        WR[4] - 0.5 * dWR[12 + i] * dx};

    double mflux = 0.;
    double pflux[3] = {0., 0., 0.};
    double Eflux = 0.;
    double normal[3] = {0, 0, 0};
    normal[i] = 1.;
    _riemann_solver.solve_for_flux(WLext[0], &WLext[1], WLext[4], WRext[0],
                                   &WRext[1], WRext[4], mflux, pflux, Eflux,
                                   normal);

    mflux *= A;
    pflux[0] *= A;
    pflux[1] *= A;
    pflux[2] *= A;
    Eflux *= A;

    dQL[0] -= mflux;
    dQL[1] -= pflux[0];
    dQL[2] -= pflux[1];
    dQL[3] -= pflux[2];
    dQL[4] -= Eflux;
  }

  /**
   * @brief Do the gradient calculation for the given interface.
   *
   * @param i Interface direction: x (0), y (1) or z (2).
   * @param WL Left state primitive variables (density - kg m^-3, velocity -
   * m s^-1, pressure - kg m^-1 s^-2).
   * @param WR Right state primitive variables (density - kg m^-3, velocity -
   * m s^-1, pressure - kg m^-1 s^-2).
   * @param dxinv Inverse distance between left and right state midpoint (in m).
   * @param dWL Left state primitive variable gradients (updated; density -
   * kg m^-4, velocity - s^-1, pressure - kg m^-2 s^-2).
   * @param WLlim Left state primitive variable limiters (updated; density -
   * kg m^-3, velocity - m s^-1, pressure - kg m^-1 s^-2).
   * @param dWR Right state primitive variable gradients (updated; density -
   * kg m^-4, velocity - s^-1, pressure - kg m^-2 s^-2).
   * @param WRlim Right state primitive variable limiters (updated; density -
   * kg m^-3, velocity - m s^-1, pressure - kg m^-1 s^-2).
   */
  inline void do_gradient_calculation(const int i, const double WL[5],
                                      const double WR[5], const double dxinv,
                                      double dWL[15], double WLlim[10],
                                      double dWR[15], double WRlim[10]) const {

    for (int j = 0; j < 5; ++j) {
      const double dwdx = 0.5 * (WL[j] + WR[j]) * dxinv;
      dWL[3 * j + i] += dwdx;
      dWR[3 * j + i] -= dwdx;
      WLlim[2 * j] = std::min(WLlim[2 * j], WR[j]);
      WLlim[2 * j + 1] = std::max(WLlim[2 * j + 1], WR[j]);
      WRlim[2 * j] = std::min(WRlim[2 * j], WL[j]);
      WRlim[2 * j + 1] = std::max(WRlim[2 * j + 1], WL[j]);
    }
  }

  /**
   * @brief Do the gradient calculation across a box boundary.
   *
   * @param i Interface direction: x (0), y (1) or z (2).
   * @param WL Left state primitive variables (density - kg m^-3, velocity -
   * m s^-1, pressure - kg m^-1 s^-2).
   * @param boundary HydroBoundary that sets the right state variables.
   * @param dxinv Inverse distance between left and right state midpoint (in m).
   * @param dWL Left state primitive variable gradients (updated; density -
   * kg m^-4, velocity - s^-1, pressure - kg m^-2 s^-2).
   * @param WLlim Left state primitive variable limiters (updated; density -
   * kg m^-3, velocity - m s^-1, pressure - kg m^-1 s^-2).
   */
  template < typename _boundary_ >
  inline void do_ghost_gradient_calculation(const int i, const double WL[5],
                                            const _boundary_ &boundary,
                                            const double dxinv, double dWL[15],
                                            double WLlim[10]) const {

    double WR[5];
    boundary.get_right_state_gradient_variables(i, WL, WR);
    for (int j = 0; j < 5; ++j) {
      const double dwdx = 0.5 * (WL[j] + WR[j]) * dxinv;
      dWL[3 * j + i] += dwdx;
      WLlim[2 * j] = std::min(WLlim[2 * j], WR[j]);
      WLlim[2 * j + 1] = std::max(WLlim[2 * j + 1], WR[j]);
    }
  }

  /**
   * @brief Apply the slope limiter for the given variables.
   *
   * @param W Primitive variables (density - kg m^-3, velocity - m s^-1,
   * pressure - kg m^-1 s^-2).
   * @param dW Primitive variable gradients (updated; density - kg m^-4,
   * velocity - s^-1, pressure - kg m^-2 s^-2).
   * @param Wlim Primitive variable limiters (density - kg m^-3, velocity -
   * m s^-1, pressure - kg m^-1 s^-2).
   * @param dx Distance between the cell and the neighbouring cells in all
   * directions (in m).
   */
  inline void apply_slope_limiter(const double W[5], double dW[15],
                                  const double Wlim[10],
                                  const double dx[3]) const {

    for (int i = 0; i < 5; ++i) {
      const double dwext[3] = {dW[3 * i] * 0.5 * dx[0],
                               dW[3 * i + 1] * 0.5 * dx[1],
                               dW[3 * i + 2] * 0.5 * dx[2]};
      double dwmax = std::max(W[i] + dwext[0], W[i] - dwext[0]);
      double dwmin = std::min(W[i] + dwext[0], W[i] - dwext[0]);
      for (int j = 1; j < 3; ++j) {
        dwmax = std::max(dwmax, W[i] + dwext[j]);
        dwmin = std::min(dwmin, W[i] + dwext[j]);
        dwmax = std::max(dwmax, W[i] - dwext[j]);
        dwmin = std::min(dwmin, W[i] - dwext[j]);
      }
      dwmax -= W[i];
      dwmin -= W[i];
      double maxfac = DBL_MAX;
      if (dwmax != 0.) {
        const double dwngbmax = Wlim[2 * i + 1] - W[i];
        maxfac = dwngbmax / dwmax;
      }
      double minfac = DBL_MAX;
      if (dwmin != 0.) {
        const double dwngbmin = Wlim[2 * i] - W[i];
        minfac = dwngbmin / dwmin;
      }
      const double alpha = std::min(1., 0.5 * std::min(maxfac, minfac));
      dW[3 * i] *= alpha;
      dW[3 * i + 1] *= alpha;
      dW[3 * i + 2] *= alpha;
    }
  }

  /**
   * @brief Predict the primitive variables forward in time with the given time
   * step.
   *
   * @param W Primitive variables (updated; density - kg m^-3, velocity -
   * m s^-1, pressure - kg m^-1 s^-2).
   * @param dW Primitive variable gradients (updated; density - kg m^-4,
   * velocity - s^-1, pressure - kg m^-2 s^-2).
   * @param dt Time step (in s).
   */
  inline void predict_primitive_variables(double W[5], const double dW[15],
                                          const double dt) const {

    const double rho = W[0];
    const double vx = W[1];
    const double vy = W[2];
    const double vz = W[3];
    const double P = W[4];
    const double rhoinv = 1. / rho;

    const double drhodx = dW[0];
    const double drhody = dW[1];
    const double drhodz = dW[2];

    const double dvxdx = dW[3];
    const double dvydy = dW[7];
    const double dvzdz = dW[11];

    const double dPdx = dW[12];
    const double dPdy = dW[13];
    const double dPdz = dW[14];

    const double divv = dvxdx + dvydy + dvzdz;

    W[0] -= dt * (rho * divv + vx * drhodx + vy * drhody + vz * drhodz);
    W[1] -= dt * (vx * divv + rhoinv * dPdx);
    W[2] -= dt * (vy * divv + rhoinv * dPdy);
    W[3] -= dt * (vz * divv + rhoinv * dPdz);
    W[4] -= dt * (_gamma * P * divv + vx * dPdx + vy * dPdy + vz * dPdz);
  }
};

#endif // HYDRO_HPP
