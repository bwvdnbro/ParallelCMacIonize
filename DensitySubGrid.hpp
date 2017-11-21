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
 * @file DensitySubGrid.hpp
 *
 * @brief Small portion of the density grid that acts as an individual density
 * grid.
 *
 * @author Bert Vandenbroucke (bv7@st-andrews.ac.uk)
 */
#ifndef DENSITYSUBGRID_HPP
#define DENSITYSUBGRID_HPP

// local includes
#include "Photon.hpp"
#include "PhotonBuffer.hpp"

// standard library includes
#include <algorithm>
#include <cfloat>
#include <cmath>
#include <iostream>
#include <ostream>

/**
 * @brief Custom assertion macro that supports extra output.
 *
 * @param condition Condition to check.
 * @param message Message to print if the assertion fails.
 */
#define myassert(condition, message)                                           \
  if (!(condition)) {                                                          \
    std::cerr << "Assertion failed: " #condition << std::endl;                 \
    std::cerr << message << std::endl;                                         \
    abort();                                                                   \
  }

/**
 * @brief Custom abort macro that prints a message before aborting.
 *
 * @param message Message to print.
 */
#define myabort(message)                                                       \
  std::cerr << message << std::endl;                                           \
  abort();

/**
 * @brief Direction of travel of a photon when it enters or leaves the subgrid.
 *
 * - INSIDE: Photon is already inside the subgrid (it was (re-)emitted in the
 *   volume covered by the subgrid).
 * - CORNER: Photon enters/leaves through a corner. The 8 corners of the cubic
 *   volume are labelled by the 3 coordinates: P means the upper limit of that
 *   coordinate, N the lower limit.
 * - EDGE: Photon enters/leaves through an edge. The 12 edges are labelled by
 *   the plane in which they make up a square, and the 2 remaining identifying
 *   coordinates in that square, as above.
 * - FACE: Photon enters/leaves through a face. The 6 faces are labelled by the
 *   plane to which they are parallel and the direction of travel perpendicular
 *   to that plane.
 */
enum TravelDirection {
  TRAVELDIRECTION_INSIDE = 0,
  TRAVELDIRECTION_CORNER_PPP,
  TRAVELDIRECTION_CORNER_PPN,
  TRAVELDIRECTION_CORNER_PNP,
  TRAVELDIRECTION_CORNER_PNN,
  TRAVELDIRECTION_CORNER_NPP,
  TRAVELDIRECTION_CORNER_NPN,
  TRAVELDIRECTION_CORNER_NNP,
  TRAVELDIRECTION_CORNER_NNN,
  TRAVELDIRECTION_EDGE_X_PP,
  TRAVELDIRECTION_EDGE_X_PN,
  TRAVELDIRECTION_EDGE_X_NP,
  TRAVELDIRECTION_EDGE_X_NN,
  TRAVELDIRECTION_EDGE_Y_PP,
  TRAVELDIRECTION_EDGE_Y_PN,
  TRAVELDIRECTION_EDGE_Y_NP,
  TRAVELDIRECTION_EDGE_Y_NN,
  TRAVELDIRECTION_EDGE_Z_PP,
  TRAVELDIRECTION_EDGE_Z_PN,
  TRAVELDIRECTION_EDGE_Z_NP,
  TRAVELDIRECTION_EDGE_Z_NN,
  TRAVELDIRECTION_FACE_X_P,
  TRAVELDIRECTION_FACE_X_N,
  TRAVELDIRECTION_FACE_Y_P,
  TRAVELDIRECTION_FACE_Y_N,
  TRAVELDIRECTION_FACE_Z_P,
  TRAVELDIRECTION_FACE_Z_N
};

/**
 * @brief Convert an outgoing direction into an ingoing direction, using the
 * fact that what goes out through one corner of a cube has to come in through
 * the opposite corner of the neighbouring cube.
 *
 * @param output_direction Outward TravelDirection.
 * @return Inward TravelDirection.
 */
inline static int output_to_input_direction(const int output_direction) {
  switch (output_direction) {
  case TRAVELDIRECTION_INSIDE:
    return TRAVELDIRECTION_INSIDE;
  case TRAVELDIRECTION_CORNER_PPP:
    return TRAVELDIRECTION_CORNER_NNN;
  case TRAVELDIRECTION_CORNER_PPN:
    return TRAVELDIRECTION_CORNER_NNP;
  case TRAVELDIRECTION_CORNER_PNP:
    return TRAVELDIRECTION_CORNER_NPN;
  case TRAVELDIRECTION_CORNER_PNN:
    return TRAVELDIRECTION_CORNER_NPP;
  case TRAVELDIRECTION_CORNER_NPP:
    return TRAVELDIRECTION_CORNER_PNN;
  case TRAVELDIRECTION_CORNER_NPN:
    return TRAVELDIRECTION_CORNER_PNP;
  case TRAVELDIRECTION_CORNER_NNP:
    return TRAVELDIRECTION_CORNER_PPN;
  case TRAVELDIRECTION_CORNER_NNN:
    return TRAVELDIRECTION_CORNER_PPP;
  case TRAVELDIRECTION_EDGE_X_PP:
    return TRAVELDIRECTION_EDGE_X_NN;
  case TRAVELDIRECTION_EDGE_X_PN:
    return TRAVELDIRECTION_EDGE_X_NP;
  case TRAVELDIRECTION_EDGE_X_NP:
    return TRAVELDIRECTION_EDGE_X_PN;
  case TRAVELDIRECTION_EDGE_X_NN:
    return TRAVELDIRECTION_EDGE_X_PP;
  case TRAVELDIRECTION_EDGE_Y_PP:
    return TRAVELDIRECTION_EDGE_Y_NN;
  case TRAVELDIRECTION_EDGE_Y_PN:
    return TRAVELDIRECTION_EDGE_Y_NP;
  case TRAVELDIRECTION_EDGE_Y_NP:
    return TRAVELDIRECTION_EDGE_Y_PN;
  case TRAVELDIRECTION_EDGE_Y_NN:
    return TRAVELDIRECTION_EDGE_Y_PP;
  case TRAVELDIRECTION_EDGE_Z_PP:
    return TRAVELDIRECTION_EDGE_Z_NN;
  case TRAVELDIRECTION_EDGE_Z_PN:
    return TRAVELDIRECTION_EDGE_Z_NP;
  case TRAVELDIRECTION_EDGE_Z_NP:
    return TRAVELDIRECTION_EDGE_Z_PN;
  case TRAVELDIRECTION_EDGE_Z_NN:
    return TRAVELDIRECTION_EDGE_Z_PP;
  case TRAVELDIRECTION_FACE_X_P:
    return TRAVELDIRECTION_FACE_X_N;
  case TRAVELDIRECTION_FACE_X_N:
    return TRAVELDIRECTION_FACE_X_P;
  case TRAVELDIRECTION_FACE_Y_P:
    return TRAVELDIRECTION_FACE_Y_N;
  case TRAVELDIRECTION_FACE_Y_N:
    return TRAVELDIRECTION_FACE_Y_P;
  case TRAVELDIRECTION_FACE_Z_P:
    return TRAVELDIRECTION_FACE_Z_N;
  case TRAVELDIRECTION_FACE_Z_N:
    return TRAVELDIRECTION_FACE_Z_P;
  default:
    // something went wrong
    myabort("Unknown output direction: " << output_direction);
    return -1;
  }
}

/**
 * @brief Small fraction of a density grid that acts as an individual density
 * grid.
 */
class DensitySubGrid {
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

  /*! @brief Indices of the neighbouring subgrids. */
  unsigned int _ngbs[27];

  /*! @brief Input buffers. Contain the photons that should be propagated
   *  through this subgrid. */
  PhotonBuffer _input_buffers[27];

  /*! @brief Output buffers. Contain the photons that have been propagated
   *  through this subgrid. */
  PhotonBuffer _output_buffers[27];

  /*! @brief Number density for each cell (in m^-3). */
  double *_number_density;

  /*! @brief Neutral fraction of hydrogen for each cell. */
  double *_neutral_fraction;

  /*! @brief Ionizing intensity estimate for each cell (in m^3). */
  double *_intensity_integral;

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

  /**
   * @brief Get the x 3 index corresponding to the given coordinate and incoming
   * direction.
   *
   * @param x X coordinate (in m).
   * @param direction Incoming direction.
   * @return x component of the 3 index of the cell that contains the given
   * position.
   */
  inline int get_x_index(const double x, const int direction) const {
    if (direction == TRAVELDIRECTION_INSIDE ||
        direction == TRAVELDIRECTION_EDGE_X_PP ||
        direction == TRAVELDIRECTION_EDGE_X_PN ||
        direction == TRAVELDIRECTION_EDGE_X_NP ||
        direction == TRAVELDIRECTION_EDGE_X_NN ||
        direction == TRAVELDIRECTION_FACE_Y_P ||
        direction == TRAVELDIRECTION_FACE_Y_N ||
        direction == TRAVELDIRECTION_FACE_Z_P ||
        direction == TRAVELDIRECTION_FACE_Z_N) {
      // need to compute the index
      return x * _inv_cell_size[0];
    } else if (direction == TRAVELDIRECTION_CORNER_NPP ||
               direction == TRAVELDIRECTION_CORNER_NPN ||
               direction == TRAVELDIRECTION_CORNER_NNP ||
               direction == TRAVELDIRECTION_CORNER_NNN ||
               direction == TRAVELDIRECTION_EDGE_Y_NP ||
               direction == TRAVELDIRECTION_EDGE_Y_NN ||
               direction == TRAVELDIRECTION_EDGE_Z_NP ||
               direction == TRAVELDIRECTION_EDGE_Z_NN ||
               direction == TRAVELDIRECTION_FACE_X_N) {
      // index is lower limit of box
      return 0;
    } else if (direction == TRAVELDIRECTION_CORNER_PPP ||
               direction == TRAVELDIRECTION_CORNER_PPN ||
               direction == TRAVELDIRECTION_CORNER_PNP ||
               direction == TRAVELDIRECTION_CORNER_PNN ||
               direction == TRAVELDIRECTION_EDGE_Y_PP ||
               direction == TRAVELDIRECTION_EDGE_Y_PN ||
               direction == TRAVELDIRECTION_EDGE_Z_PP ||
               direction == TRAVELDIRECTION_EDGE_Z_PN ||
               direction == TRAVELDIRECTION_FACE_X_P) {
      // index is upper limit of box
      return _number_of_cells[0] - 1;
    } else {
      // something went wrong
      myabort("Unknown incoming x direction: " << direction);
      return -1;
    }
  }

  /**
   * @brief Get the y 3 index corresponding to the given coordinate and incoming
   * direction.
   *
   * @param y Y coordinate (in m).
   * @param direction Incoming direction.
   * @return y component of the 3 index of the cell that contains the given
   * position.
   */
  inline int get_y_index(const double y, const int direction) const {
    if (direction == TRAVELDIRECTION_INSIDE ||
        direction == TRAVELDIRECTION_EDGE_Y_PP ||
        direction == TRAVELDIRECTION_EDGE_Y_PN ||
        direction == TRAVELDIRECTION_EDGE_Y_NP ||
        direction == TRAVELDIRECTION_EDGE_Y_NN ||
        direction == TRAVELDIRECTION_FACE_X_P ||
        direction == TRAVELDIRECTION_FACE_X_N ||
        direction == TRAVELDIRECTION_FACE_Z_P ||
        direction == TRAVELDIRECTION_FACE_Z_N) {
      // need to compute the index
      return y * _inv_cell_size[1];
    } else if (direction == TRAVELDIRECTION_CORNER_PNP ||
               direction == TRAVELDIRECTION_CORNER_PNN ||
               direction == TRAVELDIRECTION_CORNER_NNP ||
               direction == TRAVELDIRECTION_CORNER_NNN ||
               direction == TRAVELDIRECTION_EDGE_X_NP ||
               direction == TRAVELDIRECTION_EDGE_X_NN ||
               direction == TRAVELDIRECTION_EDGE_Z_PN ||
               direction == TRAVELDIRECTION_EDGE_Z_NN ||
               direction == TRAVELDIRECTION_FACE_Y_N) {
      // index is lower limit of box
      return 0;
    } else if (direction == TRAVELDIRECTION_CORNER_PNP ||
               direction == TRAVELDIRECTION_CORNER_PNN ||
               direction == TRAVELDIRECTION_CORNER_NNP ||
               direction == TRAVELDIRECTION_CORNER_NNN ||
               direction == TRAVELDIRECTION_EDGE_X_PP ||
               direction == TRAVELDIRECTION_EDGE_X_PN ||
               direction == TRAVELDIRECTION_EDGE_Z_PP ||
               direction == TRAVELDIRECTION_EDGE_Z_NP ||
               direction == TRAVELDIRECTION_FACE_Y_P) {
      // index is upper limit of box
      return _number_of_cells[1] - 1;
    } else {
      // something went wrong
      myabort("Unknown incoming y direction: " << direction);
      return -1;
    }
  }

  /**
   * @brief Get the z 3 index corresponding to the given coordinate and incoming
   * direction.
   *
   * @param z Z coordinate (in m).
   * @param direction Incoming direction.
   * @return z component of the 3 index of the cell that contains the given
   * position.
   */
  inline int get_z_index(const double z, const int direction) const {
    if (direction == TRAVELDIRECTION_INSIDE ||
        direction == TRAVELDIRECTION_EDGE_Z_PP ||
        direction == TRAVELDIRECTION_EDGE_Z_PN ||
        direction == TRAVELDIRECTION_EDGE_Z_NP ||
        direction == TRAVELDIRECTION_EDGE_Z_NN ||
        direction == TRAVELDIRECTION_FACE_X_P ||
        direction == TRAVELDIRECTION_FACE_X_N ||
        direction == TRAVELDIRECTION_FACE_Y_P ||
        direction == TRAVELDIRECTION_FACE_Y_N) {
      // need to compute the index
      return z * _inv_cell_size[2];
    } else if (direction == TRAVELDIRECTION_CORNER_PPN ||
               direction == TRAVELDIRECTION_CORNER_PNN ||
               direction == TRAVELDIRECTION_CORNER_NPN ||
               direction == TRAVELDIRECTION_CORNER_NNN ||
               direction == TRAVELDIRECTION_EDGE_X_PN ||
               direction == TRAVELDIRECTION_EDGE_X_NN ||
               direction == TRAVELDIRECTION_EDGE_Y_PN ||
               direction == TRAVELDIRECTION_EDGE_Y_NN ||
               direction == TRAVELDIRECTION_FACE_Z_N) {
      // index is lower limit of box
      return 0;
    } else if (direction == TRAVELDIRECTION_CORNER_PPP ||
               direction == TRAVELDIRECTION_CORNER_PNP ||
               direction == TRAVELDIRECTION_CORNER_NPP ||
               direction == TRAVELDIRECTION_CORNER_NNP ||
               direction == TRAVELDIRECTION_EDGE_X_PP ||
               direction == TRAVELDIRECTION_EDGE_X_NP ||
               direction == TRAVELDIRECTION_EDGE_Y_PP ||
               direction == TRAVELDIRECTION_EDGE_Y_NP ||
               direction == TRAVELDIRECTION_FACE_Z_P) {
      // index is upper limit of box
      return _number_of_cells[2] - 1;
    } else {
      // something went wrong
      myabort("Unknown incoming z direction: " << direction);
      return -1;
    }
  }

  /**
   * @brief Get the index (and 3 index) of the cell containing the given
   * incoming position, with the given incoming direction.
   *
   * @param position Incoming position (in m).
   * @param input_direction Incoming direction.
   * @param three_index 3 index (output variable).
   * @return Single index of the cell.
   */
  inline int get_start_index(const double *position, const int input_direction,
                             int *three_index) const {
    three_index[0] = get_x_index(position[0], input_direction);
    three_index[1] = get_y_index(position[1], input_direction);
    three_index[2] = get_z_index(position[2], input_direction);
    myassert(is_inside(three_index),
             "position: " << position[0] << " " << position[1] << " "
                          << position[2] << "\nbox:\t" << _anchor[0] << " "
                          << _anchor[1] << " " << _anchor[2] << "\n\t"
                          << _cell_size[0] * _number_of_cells[0] << " "
                          << _cell_size[1] * _number_of_cells[1] << " "
                          << _cell_size[2] * _number_of_cells[2]);
    return get_one_index(three_index);
  }

public:
  /**
   * @brief Get the outgoing direction corresponding to the given 3 index.
   *
   * Public because the subgrid setup routine uses this routine.
   *
   * @param three_index 3 index of a cell, possibly no longer inside this grid.
   * @return Outgoing direction corresponding to that 3 index.
   */
  inline int get_output_direction(const int *three_index) const {
    // this is hopefully compiled into a bitwise operation rather than an
    // actual condition
    const bool x_low = three_index[0] < 0;
    // this is a non-conditional check to see if
    // three_index[0] == _number_of_cells[0], and should therefore be more
    // efficient (no idea if it actually is)
    const bool x_high = three_index[0] / _number_of_cells[0];
    const bool y_low = three_index[1] < 0;
    const bool y_high = three_index[1] / _number_of_cells[1];
    const bool z_low = three_index[2] < 0;
    const bool z_high = three_index[2] / _number_of_cells[2];
    const int mask = (x_high << 5) | (x_low << 4) | (y_high << 3) |
                     (y_low << 2) | (z_high << 1) | z_low;
    // we now have a mask that combines the info on the 6 checks we have to do
    switch (mask) {
    case 0:
      return TRAVELDIRECTION_INSIDE;
    case 1:
      return TRAVELDIRECTION_FACE_Z_N;
    case 2:
      return TRAVELDIRECTION_FACE_Z_P;
    case 4:
      return TRAVELDIRECTION_FACE_Y_N;
    case 8:
      return TRAVELDIRECTION_FACE_Y_P;
    case 16:
      return TRAVELDIRECTION_FACE_X_N;
    case 32:
      return TRAVELDIRECTION_FACE_X_P;
    case 5:
      return TRAVELDIRECTION_EDGE_X_NN;
    case 6:
      return TRAVELDIRECTION_EDGE_X_NP;
    case 9:
      return TRAVELDIRECTION_EDGE_X_PN;
    case 10:
      return TRAVELDIRECTION_EDGE_X_PP;
    case 17:
      return TRAVELDIRECTION_EDGE_Y_NN;
    case 18:
      return TRAVELDIRECTION_EDGE_Y_NP;
    case 33:
      return TRAVELDIRECTION_EDGE_Y_PN;
    case 34:
      return TRAVELDIRECTION_EDGE_Y_PP;
    case 20:
      return TRAVELDIRECTION_EDGE_Z_NN;
    case 24:
      return TRAVELDIRECTION_EDGE_Z_NP;
    case 36:
      return TRAVELDIRECTION_EDGE_Z_PN;
    case 40:
      return TRAVELDIRECTION_EDGE_Z_PP;
    case 21:
      return TRAVELDIRECTION_CORNER_NNN;
    case 22:
      return TRAVELDIRECTION_CORNER_NNP;
    case 25:
      return TRAVELDIRECTION_CORNER_NPN;
    case 26:
      return TRAVELDIRECTION_CORNER_NPP;
    case 37:
      return TRAVELDIRECTION_CORNER_PNN;
    case 38:
      return TRAVELDIRECTION_CORNER_PNP;
    case 41:
      return TRAVELDIRECTION_CORNER_PPN;
    case 42:
      return TRAVELDIRECTION_CORNER_PPP;
    default:
      // something went wrong
      myabort("Unknown outgoing check mask: " << mask);
      return -1;
    }
  }

  /**
   * @brief Constructor.
   *
   * @param box Dimensions of the box that contains the grid (in m; first 3
   * elements are the anchor of the box, 3 last elements are the side lengths
   * of the box).
   * @param ncell Number of cells in each dimension.
   */
  inline DensitySubGrid(const double *box, const int *ncell)
      : _anchor{box[0], box[1], box[2]},
        _cell_size{box[3] / ncell[0], box[4] / ncell[1], box[5] / ncell[2]},
        _inv_cell_size{ncell[0] / box[3], ncell[1] / box[4], ncell[2] / box[5]},
        _number_of_cells{ncell[0], ncell[1], ncell[2], ncell[1] * ncell[2]} {

    // allocate memory for data arrays
    const int tot_ncell = _number_of_cells[3] * ncell[0];
    _number_density = new double[tot_ncell];
    _neutral_fraction = new double[tot_ncell];
    _intensity_integral = new double[tot_ncell];

    // initialize data arrays
    for (int i = 0; i < tot_ncell; ++i) {
      // initial density (homogeneous density)
      _number_density[i] = 1.e8;
      // initial neutral fraction (low value, to allow radiation to effectively
      // cover the entire volume initially)
      _neutral_fraction[i] = 1.e-6;
      _intensity_integral[i] = 0.;
    }
  }

  /**
   * @brief Destructor.
   */
  inline ~DensitySubGrid() {
    // deallocate data arrays
    delete[] _number_density;
    delete[] _neutral_fraction;
    delete[] _intensity_integral;
  }

  /**
   * @brief Get a reference to the input buffer for the given direction.
   *
   * @param input_direction TravelDirection.
   * @return Reference to the corresponding input buffer.
   */
  inline PhotonBuffer &get_input_buffer(const int input_direction) {
    return _input_buffers[input_direction];
  }

  /**
   * @brief Get a reference to the output buffer corresponding to the given
   * direction.
   *
   * @param output_direction TravelDirection.
   * @return Reference to the corresponding output buffer.
   */
  inline PhotonBuffer &get_output_buffer(const int output_direction) {
    return _output_buffers[output_direction];
  }

  /**
   * @brief Get the neighbour for the given direction.
   *
   * @param output_direction TravelDirection.
   * @return Index of the neighbouring subgrid for that direction.
   */
  inline unsigned int get_neighbour(const int output_direction) const {
    return _ngbs[output_direction];
  }

  /**
   * @brief Set the neighbour for the given direction.
   *
   * @param output_direction TravelDirection.
   * @param ngb Neighbour index.
   */
  inline void set_neighbour(const int output_direction,
                            const unsigned int ngb) {
    _ngbs[output_direction] = ngb;
  }

  /**
   * @brief Let the given Photon travel through the density grid.
   *
   * @param photon Photon.
   * @param inverse_direction Inverse of the direction vector, pre-computed for
   * better efficiency (maybe not necessary?).
   * @param input_direction Direction from which the photon enters the grid.
   * @return TravelDirection of the photon after it has traversed this grid.
   */
  inline int interact(Photon &photon, const double *inverse_direction,
                      const int input_direction) {

    // get some photon variables
    const double direction[3] = {photon._direction[0], photon._direction[1],
                                 photon._direction[2]};
    // NOTE: position is relative w.r.t. _anchor!!!
    double position[3] = {photon._position[0] - _anchor[0],
                          photon._position[1] - _anchor[1],
                          photon._position[2] - _anchor[2]};
    double tau_done = photon._current_optical_depth;
    const double tau_target = photon._target_optical_depth;
    const double cross_section = photon._photoionization_cross_section;
    const double photon_weight = photon._weight;
    // get the indices of the first cell on the photons path
    int three_index[3];
    int active_cell = get_start_index(position, input_direction, three_index);
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
        // update the cell index
        active_cell = get_one_index(three_index);
      }
      // add the pathlength to the intensity counter
      _intensity_integral[active_cell] += lminsigma * photon_weight;
      // update the photon position
      position[0] += lmin * direction[0];
      position[1] += lmin * direction[1];
      position[2] += lmin * direction[2];
    }
    // update photon quantities
    photon._current_optical_depth = tau_done;
    photon._position[0] = position[0] + _anchor[0];
    photon._position[1] = position[1] + _anchor[1];
    photon._position[2] = position[2] + _anchor[2];
    // get the outgoing direction
    if (tau_done >= tau_target) {
      return TRAVELDIRECTION_INSIDE;
    } else {
      return get_output_direction(three_index);
    }
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

  /**
   * @brief Print the neutral fractions to the given ASCII stream.
   *
   * @param stream std::ostream to write to.
   */
  inline void print_intensities(std::ostream &stream) {
    for (int ix = 0; ix < _number_of_cells[0]; ++ix) {
      const double pos_x = _anchor[0] + (ix + 0.5) * _cell_size[0];
      for (int iy = 0; iy < _number_of_cells[1]; ++iy) {
        const double pos_y = _anchor[1] + (iy + 0.5) * _cell_size[1];
        for (int iz = 0; iz < _number_of_cells[2]; ++iz) {
          const double pos_z = _anchor[2] + (iz + 0.5) * _cell_size[2];
          const int three_index[3] = {ix, iy, iz};
          const int index = get_one_index(three_index);
          stream << pos_x << "\t" << pos_y << "\t" << pos_z << "\t"
                 << _neutral_fraction[index] << "\n";
        }
      }
    }
  }

  /**
   * @brief Print the neutral fractions to the given binary stream.
   *
   * @param stream std::ostream to write to.
   */
  inline void output_intensities(std::ostream &stream) {
    for (int ix = 0; ix < _number_of_cells[0]; ++ix) {
      // not const since we cast it to a char* below
      double pos_x = _anchor[0] + (ix + 0.5) * _cell_size[0];
      for (int iy = 0; iy < _number_of_cells[1]; ++iy) {
        double pos_y = _anchor[1] + (iy + 0.5) * _cell_size[1];
        for (int iz = 0; iz < _number_of_cells[2]; ++iz) {
          double pos_z = _anchor[2] + (iz + 0.5) * _cell_size[2];
          const int three_index[3] = {ix, iy, iz};
          const int index = get_one_index(three_index);
          stream.write(reinterpret_cast<char *>(&pos_x), sizeof(double));
          stream.write(reinterpret_cast<char *>(&pos_y), sizeof(double));
          stream.write(reinterpret_cast<char *>(&pos_z), sizeof(double));
          stream.write(reinterpret_cast<char *>(&_neutral_fraction[index]),
                       sizeof(double));
        }
      }
    }
  }
};

#endif // DENSITYSUBGRID_HPP
