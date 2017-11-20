/*******************************************************************************
 * This file is part of CMacIonize
 * Copyright (C) 2016 Bert Vandenbroucke (bert.vandenbroucke@gmail.com)
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
 * @file CartesianDensityGrid.cpp
 *
 * @brief Cartesian density grid: implementation.
 *
 * @author Bert Vandenbroucke (bv7@st-andrews.ac.uk)
 */
#ifndef DENSITYSUBGRID_HPP
#define DENSITYSUBGRID_HPP

#include "Photon.hpp"

#include <algorithm>
#include <cassert>
#include <cfloat>
#include <cmath>
#include <ostream>

class DensitySubGrid{
private:

  double _anchor[3];
  double _cell_size[3];
  double _inv_cell_size[3];

  int _number_of_cells[4];// = {num cell x, num cell y, num cell z, num cell y * num cell z}

  double *_number_density;
  double *_neutral_fraction;
  double *_intensity_integral;

  int get_one_index(const int *three_index) const{
    return three_index[0] * _number_of_cells[3] + three_index[1] * _number_of_cells[2] + three_index[2];
  }

  bool is_inside(const int *three_index) const {
    return three_index[0] < _number_of_cells[0] && three_index[0] >= 0 &&
           three_index[1] < _number_of_cells[1] && three_index[1] >= 0 &&
           three_index[2] < _number_of_cells[2] && three_index[2] >= 0;
  }

  int get_start_index(const double *position, const int input_direction, int* three_index){
    three_index[0] = position[0] * _inv_cell_size[0];
    three_index[1] = position[1] * _inv_cell_size[1];
    three_index[2] = position[2] * _inv_cell_size[2];
    assert(is_inside(three_index));
    return get_one_index(three_index);
  }

  int get_output_direction(const int* three_index){
    return -1;
  }

public:
  DensitySubGrid(const double *box, const int *ncell)
    : _anchor{box[0], box[1], box[2]},
      _cell_size{box[3] / ncell[0], box[4] / ncell[1], box[5] / ncell[2]},
      _inv_cell_size{ncell[0] / box[3], ncell[1] / box[4], ncell[2] / box[5]},
      _number_of_cells{ncell[0], ncell[1], ncell[2], ncell[1] * ncell[2]} {

    const int tot_ncell = _number_of_cells[3] * ncell[0];
    _number_density = new double[tot_ncell];
    _neutral_fraction = new double[tot_ncell];
    _intensity_integral = new double[tot_ncell];

    for(int i = 0; i < tot_ncell; ++i){
      _number_density[i] = 1.;
      _neutral_fraction[i] = 1.;
      _intensity_integral[i] = 0.;
    }
  }

  ~DensitySubGrid(){
    delete [] _number_density;
    delete [] _neutral_fraction;
    delete [] _intensity_integral;
  }

  /**
   * @brief Let the given Photon travel through the density grid until the given
   * optical depth is reached.
   *
   * @param photon Photon.
   * @param optical_depth Optical depth the photon should travel in total
   * (dimensionless).
   * @return DensityGrid::iterator pointing to the cell the photon was last in,
   * or DensityGrid::end() if the photon left the box.
   */
   int interact(Photon &photon, const double *inverse_direction) {

    const double direction[3] = {photon._direction[0], photon._direction[1], photon._direction[2]};
    // NOTE: position is relative w.r.t. _anchor!!!
    double position[3] = {photon._position[0] - _anchor[0], photon._position[1] - _anchor[1], photon._position[2] - _anchor[2]};
    double tau_done = photon._current_optical_depth;
    const double tau_target = photon._target_optical_depth;
    const double cross_section = photon._photoionization_cross_section;
    const double photon_weight = photon._weight;
    int three_index[3];
    int active_cell = get_start_index(position, 0, three_index);
    // enter while loop. QUESTION: what is condition?
    // double condition:
    //  - target optical depth not reached (tau_done < tau_target)
    //  - photon still in subgrid: is_inside(three_index)
    while(tau_done < tau_target && is_inside(three_index)){
      // get cell boundaries
      const double cell_low[3] = {three_index[0] * _cell_size[0],
                                  three_index[1] * _cell_size[1],
                                  three_index[2] * _cell_size[2]};
      const double cell_high[3] = {(three_index[0] + 1.) * _cell_size[0],
                                   (three_index[1] + 1.) * _cell_size[1],
                                   (three_index[2] + 1.) * _cell_size[2]};

      // compute cell distances
      double l[3];
      for(unsigned char idim = 0; idim < 3; ++idim){
        if(direction[idim] > 0.){
          l[idim] = (cell_high[idim] - position[idim]) * inverse_direction[idim];
        } else if(direction[idim] < 0.){
          l[idim] = (cell_low[idim] - position[idim]) * inverse_direction[idim];
        } else {
          l[idim] = DBL_MAX;
        }
      }

      // find the minimum
      double lmin = std::min(l[0], std::min(l[1], l[2]));
      double lminsigma = lmin * cross_section;
      const double tau = lminsigma * _number_density[active_cell] * _neutral_fraction[active_cell];
      tau_done += tau;
      if(tau_done >= tau_target){
        const double correction = (tau_done - tau_target) / tau;
        lmin *= (1. - correction);
        lminsigma = lmin * cross_section;
      } else {
        // update three_index
        for(unsigned char idim = 0; idim < 3; ++idim){
          if(l[idim] == lmin){
            three_index[idim] += (direction[idim] > 0.) ? 1 : -1;
          }
        }
      }
      _intensity_integral[active_cell] += lminsigma * photon_weight;
      position[0] += lmin * direction[0];
      position[1] += lmin * direction[1];
      position[2] += lmin * direction[2];
      active_cell = get_one_index(three_index);
    }
    photon._current_optical_depth = tau_done;
    photon._position[0] = position[0] + _anchor[0];
    photon._position[1] = position[1] + _anchor[1];
    photon._position[2] = position[2] + _anchor[2];
    if(tau_done >= tau_target){
      return -2;
    } else {
      return get_output_direction(three_index);
    }
  }

  void print_intensities(std::ostream &stream){
    for(int ix = 0; ix < _number_of_cells[0]; ++ix){
      const double pos_x = _anchor[0] + (ix + 0.5) * _cell_size[0];
      for(int iy = 0; iy < _number_of_cells[1]; ++iy){
        const double pos_y = _anchor[1] + (iy + 0.5) * _cell_size[1];
        for(int iz = 0; iz < _number_of_cells[2]; ++iz){
          const double pos_z = _anchor[2] + (iz + 0.5) * _cell_size[2];
          const int three_index[3] = {ix, iy, iz};
          const int index = get_one_index(three_index);
          stream << pos_x << "\t" << pos_y << "\t" << pos_z << "\t" << _intensity_integral[index] << "\n";
        }
      }
    }
  }

};

#endif  // DENSITYSUBGRID_HPP
