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
 * @file CoarseDensityGrid.hpp
 *
 * @brief Coarse version of the density grid used to figure out node and edge
 * costs for the domain decomposition graph.
 *
 * @author Bert Vandenbroucke (bv7@st-andrews.ac.uk)
 */
#ifndef COARSEDENSITYGRID_HPP
#define COARSEDENSITYGRID_HPP

// local includes
#include "Assert.hpp"
#include "Atomic.hpp"
#include "Error.hpp"
#include "Lock.hpp"
#include "MemoryMap.hpp"
#include "Photon.hpp"
#include "TravelDirections.hpp"

// standard library includes
#include <algorithm>
#include <cfloat>
#include <cmath>
#include <iostream>
#include <omp.h>
#include <ostream>

/**
 * @brief Coarse version of the density grid used to figure out node and edge
 * costs for the domain decomposition graph.
 */
class CoarseDensityGrid {
private:
  /*! @brief Lower front left corner of the box that contains the subgrid (in
   *  m). */
  double _anchor[3];

  /*! @brief Dimensions of a single cell of the subgrid (in m). */
  double _cell_size[3];
  /**
   * @brief Inverse dimensions of a single cell of the subgrid (in m^-1).
   *
   * Used to convert positions into grid indices.
   */
  double _inv_cell_size[3];

  /**
   * @brief Number of cells in each grid dimension (and commonly used
   * combinations).
   *
   * The first 3 elements are just the number of elements in the 3 coordinate
   * directions. The fourth element is the product of the second and the third,
   * so that the single index of a cell with three indices `ix`, `iy` and `iz`
   * is simply given by
   * ```
   *  index = ix * _number_of_cells[3] + iy * _number_of_cells[2] + iz;
   * ```
   */
  int _number_of_cells[4];

  /*! @brief Number density for each cell (in m^-3). */
  double *_number_density;

  /*! @brief Neutral fraction of hydrogen for each cell. */
  double *_neutral_fraction;

  /*! @brief Ionizing intensity estimate for each cell (in m^3). */
  double *_intensity_integral;

  /*! @brief Number of photons crossing from one cell to another. */
  unsigned int *_edge_costs;

  /**
   * @brief Convert the given 3 indices to a single index.
   *
   * The single index can be used to access elements of the internal data
   * arrays.
   *
   * @param three_index 3 indices of a cell.
   * @return Single index of that same cell.
   */
  inline int get_one_index(const int *three_index) const {
    return three_index[0] * _number_of_cells[3] +
           three_index[1] * _number_of_cells[2] + three_index[2];
  }

  /**
   * @brief Check if the given three_index still points to an internal cell.
   *
   * @param three_index 3 indices of a cell.
   * @return True if the 3 indices match a cell in this grid.
   */
  inline bool is_inside(const int *three_index) const {
    return three_index[0] < _number_of_cells[0] && three_index[0] >= 0 &&
           three_index[1] < _number_of_cells[1] && three_index[1] >= 0 &&
           three_index[2] < _number_of_cells[2] && three_index[2] >= 0;
  }

public:
  /**
   * @brief Get the index (and 3 index) of the cell containing the given
   * incoming position, with the given incoming direction.
   *
   * Public for unit testing.
   *
   * @param position Incoming position (in m).
   * @param three_index 3 index (output variable).
   * @return Single index of the cell.
   */
  inline int get_start_index(const double *position, int *three_index) const {

    three_index[0] = position[0] * _inv_cell_size[0];
    three_index[1] = position[1] * _inv_cell_size[1];
    three_index[2] = position[2] * _inv_cell_size[2];

    return get_one_index(three_index);
  }

  /**
   * @brief Constructor.
   *
   * @param box Dimensions of the box that contains the grid (in m; first 3
   * elements are the anchor of the box, 3 last elements are the side lengths
   * of the box).
   * @param ncell Number of cells in each dimension.
   */
  inline CoarseDensityGrid(const double *box, const int *ncell)
      : _anchor{box[0], box[1], box[2]},
        _cell_size{box[3] / ncell[0], box[4] / ncell[1], box[5] / ncell[2]},
        _inv_cell_size{ncell[0] / box[3], ncell[1] / box[4], ncell[2] / box[5]},
        _number_of_cells{ncell[0], ncell[1], ncell[2], ncell[1] * ncell[2]} {

    // allocate memory for data arrays
    const int tot_ncell = _number_of_cells[3] * ncell[0];
    _number_density = new double[tot_ncell];
    _neutral_fraction = new double[tot_ncell];
    _intensity_integral = new double[tot_ncell];
    _edge_costs = new unsigned int[tot_ncell * 27];

    // initialize data arrays
    for (int i = 0; i < tot_ncell; ++i) {
      // initial density (homogeneous density)
      _number_density[i] = 1.e8;
      // initial neutral fraction (low value, to allow radiation to effectively
      // cover the entire volume initially)
      _neutral_fraction[i] = 1.e-6;
      _intensity_integral[i] = 0.;
      _edge_costs[i] = 0;
    }
  }

  /**
   * @brief Destructor.
   */
  inline ~CoarseDensityGrid() {
    // deallocate data arrays
    delete[] _number_density;
    delete[] _neutral_fraction;
    delete[] _intensity_integral;
    delete[] _edge_costs;
  }

  /**
   * @brief Set the number density for the cell with the given index.
   *
   * @param index Index of a cell.
   * @param number_density New value for the number density (in kg m^-3).
   */
  inline void set_number_density(const unsigned int index,
                                 const double number_density) {
    _number_density[index] = number_density;
  }

  /**
   * @brief Set the neutral fraction for the cell with the given index.
   *
   * @param index Index of a cell.
   * @param number_density New value for the neutral fraction.
   */
  inline void set_neutral_fraction(const unsigned int index,
                                   const double neutral_fraction) {
    _neutral_fraction[index] = neutral_fraction;
  }

  /**
   * @brief Set the intensity integral for the cell with the given index.
   *
   * @param index Index of a cell.
   * @param number_density New value for the intensity integral.
   */
  inline void set_intensity_integral(const unsigned int index,
                                     const double intensity_integral) {
    _intensity_integral[index] = intensity_integral;
  }

  /**
   * @brief Check if the given position is in the box that contains this
   * subgrid.
   *
   * @param position Position (in m).
   * @return True if the given position is in the box of the subgrid.
   */
  inline bool is_in_box(const double *position) const {
    return position[0] >= _anchor[0] &&
           position[0] <= _anchor[0] + _cell_size[0] * _number_of_cells[0] &&
           position[1] >= _anchor[1] &&
           position[1] <= _anchor[1] + _cell_size[1] * _number_of_cells[1] &&
           position[2] >= _anchor[2] &&
           position[2] <= _anchor[2] + _cell_size[2] * _number_of_cells[2];
  }

  /**
   * @brief Let the given Photon travel through the density grid.
   *
   * @param photon Photon.
   */
  inline void interact(Photon &photon) {

    // get some photon variables
    const double *direction = photon.get_direction();

    const double inverse_direction[3] = {1. / direction[0], 1. / direction[1],
                                         1. / direction[2]};
    // NOTE: position is relative w.r.t. _anchor!!!
    double position[3] = {photon.get_position()[0] - _anchor[0],
                          photon.get_position()[1] - _anchor[1],
                          photon.get_position()[2] - _anchor[2]};
    double tau_done = 0.;
    const double tau_target = photon.get_target_optical_depth();

    myassert(tau_done < tau_target, "tau_done: " << tau_done
                                                 << ", target: " << tau_target);

    const double cross_section = photon.get_photoionization_cross_section();
    const double photon_weight = photon.get_weight();
    // get the indices of the first cell on the photon's path
    int three_index[3];
    int active_cell = get_start_index(position, three_index);

    myassert(active_cell >= 0 &&
                 active_cell < _number_of_cells[0] * _number_of_cells[3],
             "active_cell: " << active_cell << ", size: "
                             << _number_of_cells[0] * _number_of_cells[3]);

    // enter photon traversal loop
    // double condition:
    //  - target optical depth not reached (tau_done < tau_target)
    //  - photon still in subgrid: is_inside(three_index)
    while (tau_done < tau_target && is_inside(three_index)) {
      // get cell boundaries
      const double cell_low[3] = {three_index[0] * _cell_size[0],
                                  three_index[1] * _cell_size[1],
                                  three_index[2] * _cell_size[2]};
      const double cell_high[3] = {(three_index[0] + 1.) * _cell_size[0],
                                   (three_index[1] + 1.) * _cell_size[1],
                                   (three_index[2] + 1.) * _cell_size[2]};

      myassert(cell_low[0] <= position[0] && cell_high[0] >= position[0] &&
                   cell_low[1] <= position[1] && cell_high[1] >= position[1] &&
                   cell_low[2] <= position[2] && cell_high[2] >= position[2],
               "position: "
                   << position[0] << " " << position[1] << " " << position[2]
                   << "\ncell_low: " << cell_low[0] << " " << cell_low[1] << " "
                   << cell_low[2] << "\ncell_high: " << cell_high[0] << " "
                   << cell_high[1] << " " << cell_high[2]
                   << "\ndirection: " << direction[0] << " " << direction[1]
                   << " " << direction[2] << "\nthree_index: " << three_index[0]
                   << " " << three_index[1] << " " << three_index[2]);

      // compute cell distances
      double l[3];
      for (unsigned char idim = 0; idim < 3; ++idim) {
        if (direction[idim] > 0.) {
          l[idim] =
              (cell_high[idim] - position[idim]) * inverse_direction[idim];
        } else if (direction[idim] < 0.) {
          l[idim] = (cell_low[idim] - position[idim]) * inverse_direction[idim];
        } else {
          l[idim] = DBL_MAX;
        }
      }

      // find the minimum
      double lmin = std::min(l[0], std::min(l[1], l[2]));

      myassert(lmin >= 0.,
               "lmin: " << lmin << "\nl: " << l[0] << " " << l[1] << " " << l[2]
                        << "\ncell: " << cell_low[0] << " " << cell_low[1]
                        << " " << cell_low[2] << ", " << cell_high[0] << " "
                        << cell_high[1] << " " << cell_high[2]
                        << "\nposition: " << position[0] << " " << position[1]
                        << " " << position[2] << "\ndirection: " << direction[0]
                        << " " << direction[1] << " " << direction[2]);

      double lminsigma = lmin * cross_section;
      // compute the corresponding optical depth
      const double tau = lminsigma * _number_density[active_cell] *
                         _neutral_fraction[active_cell];
      tau_done += tau;
      // check if the target optical depth was reached
      if (tau_done >= tau_target) {
        // if so: subtract the surplus from the path
        const double correction = (tau_done - tau_target) / tau;
        lmin *= (1. - correction);
        lminsigma = lmin * cross_section;
      } else {
        // if not: photon leaves cell
        // update three_index
        for (unsigned char idim = 0; idim < 3; ++idim) {
          if (l[idim] == lmin) {
            three_index[idim] += (direction[idim] > 0.) ? 1 : -1;
          }
        }
      }
      // add the pathlength to the intensity counter
      _intensity_integral[active_cell] += lminsigma * photon_weight;

      // update the photon position
      // we use the complicated syntax below to make sure the positions we
      // know are 100% accurate (only important for our assertions)
      position[0] = (l[0] == lmin)
                        ? ((direction[0] > 0.) ? cell_high[0] : cell_low[0])
                        : position[0] + lmin * direction[0];
      position[1] = (l[1] == lmin)
                        ? ((direction[1] > 0.) ? cell_high[1] : cell_low[1])
                        : position[1] + lmin * direction[1];
      position[2] = (l[2] == lmin)
                        ? ((direction[2] > 0.) ? cell_high[2] : cell_low[2])
                        : position[2] + lmin * direction[2];

      // update the cell index
      const int new_active_cell = get_one_index(three_index);

      if (is_inside(three_index)) {
        // log the edge cost of this photon crossing from the old to the new
        // cell
        unsigned char bit[3];
        for (unsigned char idim = 0; idim < 3; ++idim) {
          bit[idim] = 1;
          if (l[idim] == lmin) {
            bit[idim] = (direction[idim] > 0.) ? 2 : 0;
          }
        }
        const unsigned char out_index = bit[0] * 9 + bit[1] * 3 + bit[2];
        const unsigned char in_index =
            (2 - bit[0]) * 9 + (2 - bit[1]) * 3 + 2 - bit[2];
        ++_edge_costs[active_cell * 27 + out_index];
        ++_edge_costs[new_active_cell * 27 + in_index];
      }

      active_cell = new_active_cell;
    }
  }

  /**
   * @brief Get the edge cost for the given direction.
   *
   * @param icell Cell index.
   * @param travel_direction TravelDirection.
   * @return Edge cost for that direction.
   */
  inline unsigned int get_edge_cost(const unsigned int icell,
                                    const int travel_direction) const {

    unsigned int index;
    switch (travel_direction) {
    case TRAVELDIRECTION_INSIDE:
      index = 13;
      break;
    case TRAVELDIRECTION_CORNER_PPP:
      index = 26;
      break;
    case TRAVELDIRECTION_CORNER_PPN:
      index = 24;
      break;
    case TRAVELDIRECTION_CORNER_PNP:
      index = 20;
      break;
    case TRAVELDIRECTION_CORNER_PNN:
      index = 18;
      break;
    case TRAVELDIRECTION_CORNER_NPP:
      index = 8;
      break;
    case TRAVELDIRECTION_CORNER_NPN:
      index = 6;
      break;
    case TRAVELDIRECTION_CORNER_NNP:
      index = 2;
      break;
    case TRAVELDIRECTION_CORNER_NNN:
      index = 0;
      break;
    case TRAVELDIRECTION_EDGE_X_PP:
      index = 17;
      break;
    case TRAVELDIRECTION_EDGE_X_PN:
      index = 15;
      break;
    case TRAVELDIRECTION_EDGE_X_NP:
      index = 11;
      break;
    case TRAVELDIRECTION_EDGE_X_NN:
      index = 9;
      break;
    case TRAVELDIRECTION_EDGE_Y_PP:
      index = 23;
      break;
    case TRAVELDIRECTION_EDGE_Y_PN:
      index = 21;
      break;
    case TRAVELDIRECTION_EDGE_Y_NP:
      index = 5;
      break;
    case TRAVELDIRECTION_EDGE_Y_NN:
      index = 3;
      break;
    case TRAVELDIRECTION_EDGE_Z_PP:
      index = 25;
      break;
    case TRAVELDIRECTION_EDGE_Z_PN:
      index = 19;
      break;
    case TRAVELDIRECTION_EDGE_Z_NP:
      index = 7;
      break;
    case TRAVELDIRECTION_EDGE_Z_NN:
      index = 1;
      break;
    case TRAVELDIRECTION_FACE_X_P:
      index = 22;
      break;
    case TRAVELDIRECTION_FACE_X_N:
      index = 4;
      break;
    case TRAVELDIRECTION_FACE_Y_P:
      index = 16;
      break;
    case TRAVELDIRECTION_FACE_Y_N:
      index = 10;
      break;
    case TRAVELDIRECTION_FACE_Z_P:
      index = 14;
      break;
    case TRAVELDIRECTION_FACE_Z_N:
      index = 12;
      break;
    default:
      // something went wrong
      cmac_error("Unknown travel direction: %i", travel_direction);
    }

    return _edge_costs[icell * 27 + index];
  }

  /**
   * @brief Get the intensity integral for the cell with the given indices.
   *
   * @param ix X index of the cell.
   * @param iy Y index of the cell.
   * @param iz Z index of the cell.
   * @return Luminosity integral for the cell with the given indices.
   */
  inline double get_intensity_integral(const int ix, const int iy,
                                       const int iz) const {
    const int three_index[3] = {ix, iy, iz};
    const int index = get_one_index(three_index);
    return _intensity_integral[index];
  }

  /**
   * @brief Update the ionization state for all the cells in this subgrid.
   *
   * @param num_photon Total number of photon packets that was used, used to
   * normalize the intensity estimates.
   */
  inline void compute_neutral_fraction(const unsigned int num_photon) {
    // compute the normalization factor:
    // constant source luminosity / number of packets / cell volume
    // if we multiply this with the intensity estimates (unit: m^3), we get a
    // quantity in s^-1
    const double jfac = 4.26e49 / num_photon * _inv_cell_size[0] *
                        _inv_cell_size[1] * _inv_cell_size[2];
    // constant recombination rate
    const double alphaH = 4.e-19;
    // total number of cells in the subgrid
    const int ncell_tot = _number_of_cells[0] * _number_of_cells[3];
    // compute the balance for each cell
    for (int i = 0; i < ncell_tot; ++i) {
      // normalize the intensity estimate
      const double jH = jfac * _intensity_integral[i];
      // if the intensity was non-zero: solve the balance equation
      if (jH > 0.) {
        const double ntot = _number_density[i];
        const double aa = 0.5 * jH / (ntot * alphaH);
        const double bb = 2. / aa;
        const double cc = std::sqrt(bb + 1.);
        _neutral_fraction[i] = 1. + aa * (1. - cc);
      } else {
        // if there was no ionizing radiation, the cell is trivially neutral
        _neutral_fraction[i] = 1.;
      }
      // reset the intensity for the next loop
      _intensity_integral[i] = 0.;
    }
  }
};

#endif // COARSEDENSITYGRID_HPP
