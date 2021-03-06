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
#include "Assert.hpp"
#include "Atomic.hpp"
#include "Error.hpp"
#include "Hydro.hpp"
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

#ifdef WITH_MPI
#include <mpi.h>
#endif

/*! @brief Special neighbour index marking a neighbour that does not exist. */
#define NEIGHBOUR_OUTSIDE 0xffffffff

/*! @brief Enable this to disable hydro variables (decreases memory footprint
 *  of code). */
//#define NO_HYDRO

/*! @brief Enable this to activate cell locking. */
//#define SUBGRID_CELL_LOCK

#ifdef SUBGRID_CELL_LOCK
#define subgrid_cell_lock_variables() Lock *_locks
#define subgrid_cell_lock_init(ncell) _locks = new Lock[ncell]
#define subgrid_cell_lock_destroy() delete[] _locks
#define subgrid_cell_lock_lock(cell) _locks[cell].lock()
#define subgrid_cell_lock_unlock(cell) _locks[cell].unlock()
#else
#define subgrid_cell_lock_variables()
#define subgrid_cell_lock_init(ncell)
#define subgrid_cell_lock_destroy()
#define subgrid_cell_lock_lock(cell)
#define subgrid_cell_lock_unlock(cell)
#endif

/*! @brief Size of the DensitySubGrid variables that need to be communicated
 *  over MPI, and whose size is known at compile time. */
#define DENSITYSUBGRID_FIXED_MPI_SIZE                                          \
  (sizeof(unsigned long) + (9 + 5) * sizeof(double) + 4 * sizeof(int) +        \
   TRAVELDIRECTION_NUMBER * sizeof(unsigned int))

/*! @brief Size of all DensitySubGrid variables whose size is known at compile
 *  time. */
#define DENSITYSUBGRID_FIXED_SIZE sizeof(DensitySubGrid)

/*! @brief Number of variables stored in each cell of the DensitySubGrid
 *  (excluding potential lock variables). */
#define DENSITYSUBGRID_ELEMENT_SIZE (3 + 40) * sizeof(double)

/**
 * @brief Small fraction of a density grid that acts as an individual density
 * grid.
 */
class DensitySubGrid {
private:
  /*! @brief Indices of the neighbouring subgrids. */
  unsigned int _ngbs[TRAVELDIRECTION_NUMBER];

  /*! @brief Indices of the active buffers. */
  unsigned int _active_buffers[TRAVELDIRECTION_NUMBER];

#ifdef DENSITYGRID_EDGECOST
  /*! @brief Communication cost per edge. */
  unsigned int _communication_cost[TRAVELDIRECTION_NUMBER];
#endif

  /*! @brief Computational cost of this subgrid. */
  unsigned long _computational_cost;

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

  /*! @brief Dependency lock. */
  Lock _dependency;

  /*! @brief ID of the last thread that used this subgrid. */
  int _owning_thread;

  /*! @brief Index of the largest active buffer. */
  unsigned char _largest_buffer_index;

  /*! @brief Size of the largest active buffer. */
  unsigned int _largest_buffer_size;

  /// PHOTOIONIZATION VARIABLES

  /*! @brief Number density for each cell (in m^-3). */
  double *_number_density;

  /*! @brief Neutral fraction of hydrogen for each cell. */
  double *_neutral_fraction;

  /*! @brief Ionizing intensity estimate for each cell (in m^3). */
  double *_intensity_integral;

  /// HYDRO VARIABLES

  /*! @brief Volume of a single cell (in m^3). */
  double _cell_volume;

  /*! @brief Inverse volume of a single cell (in m^-3). */
  double _inverse_cell_volume;

  /*! @brief Surface areas of a single cell (in m^2). */
  double _cell_areas[3];

  /*! @brief Conserved variables (mass - kg, momentum - kg m s^-1, energy -
   *  kg m^2 s^-2). */
  float *_conserved_variables;

  /*! @brief Change in conserved variables (mass - kg s^-1, momentum - kg m
   *  s^-2, energy - kg m^2 s^-3). */
  float *_delta_conserved_variables;

  /*! @brief Primitive variables (density - kg m^-3, velocity - m s^-1,
   *  pressure - kg m^-1 s^-2). */
  float *_primitive_variables;

  /*! @brief Primitive variable gradients (density - kg m^-4, velocity - s^-1,
   *  pressure - kg m^-2 s^-2). */
  float *_primitive_variable_gradients;

  /*! @brief Minimum and maximum neighbouring values used by the slope limiter
   *  (density - kg m^-3, velocity - m s^-1, pressure - kg m^-1 s^-2). */
  float *_primitive_variable_limiters;

  /*! @brief Indices of the hydro tasks associated with this subgrid. */
  size_t _hydro_tasks[18];

  /*! @brief Cell locks (if active). */
  subgrid_cell_lock_variables();

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
   * @brief Convert the given single index into 3 indices.
   *
   * @param one_index Single index of the cell.
   * @param three_index 3 indices of that same cell.
   */
  inline void get_three_index(const int one_index, int *three_index) const {
    three_index[0] = one_index / _number_of_cells[3];
    three_index[1] = (one_index - three_index[0] * _number_of_cells[3]) /
                     _number_of_cells[2];
    three_index[2] = one_index - three_index[0] * _number_of_cells[3] -
                     three_index[1] * _number_of_cells[2];
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
    // we need to distinguish between cases where
    //  - we need to compute the x index (position is inside or on an edge/face
    //    that does not have a fixed x value: EDGE_X and FACE_Y/Z)
    //  - the x index is the lower value (position is on a corner with N x
    //    coordinate, or in a not X edge with N x coordinate, or on the N
    //    FACE_X)
    //  - the x index is the upper value (position is on a corner with P x
    //    coordinate, or in a not X edge with P x coordinate, or on the P
    //    FACE_X)
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
      cmac_error("Unknown incoming x direction: %i", direction);
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
    // we need to distinguish between cases where
    //  - we need to compute the y index (position is inside or on an edge/face
    //    that does not have a fixed y value: EDGE_Y and FACE_X/Z)
    //  - the y index is the lower value (position is on a corner with N y
    //    coordinate, or in a not Y edge with N y coordinate - note that this is
    //    the first coordinate for EDGE_X and the second for EDGE_Y! -, or on
    //    the N FACE_Y)
    //  - the y index is the upper value (position is on a corner with P y
    //    coordinate, or in a not Y edge with P y coordinate, or on the P
    //    FACE_Y)
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
    } else if (direction == TRAVELDIRECTION_CORNER_PPP ||
               direction == TRAVELDIRECTION_CORNER_PPN ||
               direction == TRAVELDIRECTION_CORNER_NPP ||
               direction == TRAVELDIRECTION_CORNER_NPN ||
               direction == TRAVELDIRECTION_EDGE_X_PP ||
               direction == TRAVELDIRECTION_EDGE_X_PN ||
               direction == TRAVELDIRECTION_EDGE_Z_PP ||
               direction == TRAVELDIRECTION_EDGE_Z_NP ||
               direction == TRAVELDIRECTION_FACE_Y_P) {
      // index is upper limit of box
      return _number_of_cells[1] - 1;
    } else {
      // something went wrong
      cmac_error("Unknown incoming y direction: %i", direction);
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
    // we need to distinguish between cases where
    //  - we need to compute the z index (position is inside or on an edge/face
    //    that does not have a fixed z value: EDGE_Z and FACE_X/Y)
    //  - the z index is the lower value (position is on a corner with N z
    //    coordinate, or in a not Z edge with N z coordinate, or on the N
    //    FACE_Z)
    //  - the z index is the upper value (position is on a corner with P z
    //    coordinate, or in a not Z edge with P z coordinate, or on the P
    //    FACE_Z)
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
      cmac_error("Unknown incoming z direction: %i", direction);
      return -1;
    }
  }

public:
  /**
   * @brief Get the index (and 3 index) of the cell containing the given
   * incoming position, with the given incoming direction.
   *
   * Public for unit testing.
   *
   * @param position Incoming position (in m).
   * @param input_direction Incoming direction.
   * @param three_index 3 index (output variable).
   * @return Single index of the cell.
   */
  inline int get_start_index(const double *position, const int input_direction,
                             int *three_index) const {

    three_index[0] = get_x_index(position[0], input_direction);

    myassert(std::abs(three_index[0] - (int)(position[0] * _inv_cell_size[0])) <
                 2,
             "input_direction: "
                 << input_direction << "\nposition: " << position[0] << " "
                 << position[1] << " " << position[2]
                 << "\nthree_index[0]: " << three_index[0]
                 << "\nreal: " << (int)(position[0] * _inv_cell_size[0]));

    three_index[1] = get_y_index(position[1], input_direction);

    myassert(
        std::abs(three_index[1] - (int)(position[1] * _inv_cell_size[1])) < 2,
        "input_direction: " << input_direction << "\nposition: " << position[0]
                            << " " << position[1] << " " << position[2]
                            << "\nthree_index[1]: " << three_index[1]);

    three_index[2] = get_z_index(position[2], input_direction);

    myassert(
        std::abs(three_index[2] - (int)(position[2] * _inv_cell_size[2])) < 2,
        "input_direction: " << input_direction << "\nposition: " << position[0]
                            << " " << position[1] << " " << position[2]
                            << "\nthree_index[2]: " << three_index[2]);

    myassert(is_inside(three_index),
             "position: " << position[0] + _anchor[0] << " "
                          << position[1] + _anchor[1] << " "
                          << position[2] + _anchor[2] << "\nbox:\t"
                          << _anchor[0] << " " << _anchor[1] << " "
                          << _anchor[2] << "\n\t"
                          << _cell_size[0] * _number_of_cells[0] << " "
                          << _cell_size[1] * _number_of_cells[1] << " "
                          << _cell_size[2] * _number_of_cells[2]
                          << "\ndirection: " << input_direction);

    return get_one_index(three_index);
  }

  /**
   * @brief Get the outgoing direction corresponding to the given 3 index.
   *
   * Public because the subgrid setup routine uses this routine.
   *
   * Instead of doing a very complicated nested conditional structure, we
   * convert the 6 conditions into a single condition mask and use a switch
   * statement.
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
    const bool x_high = (three_index[0] / _number_of_cells[0]) > 0;
    const bool y_low = three_index[1] < 0;
    const bool y_high = (three_index[1] / _number_of_cells[1]) > 0;
    const bool z_low = three_index[2] < 0;
    const bool z_high = (three_index[2] / _number_of_cells[2]) > 0;
    const int mask = (x_high << 5) | (x_low << 4) | (y_high << 3) |
                     (y_low << 2) | (z_high << 1) | z_low;
    // we now have a mask that combines the info on the 6 checks we have to do:
    // the highest two bits give us the x checks, and so on
    //  e.g. mask = 40 = 101000 means both the x and y index are above the range
    const int output_direction = TravelDirections::get_output_direction(mask);
    if (output_direction < 0) {
      cmac_error("Unknown outgoing check mask: %i (three_index: %i %i %i)",
                 mask, three_index[0], three_index[1], three_index[2]);
    }
    return output_direction;
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
      : _computational_cost(0), _anchor{box[0], box[1], box[2]},
        _cell_size{box[3] / ncell[0], box[4] / ncell[1], box[5] / ncell[2]},
        _inv_cell_size{ncell[0] / box[3], ncell[1] / box[4], ncell[2] / box[5]},
        _number_of_cells{ncell[0], ncell[1], ncell[2], ncell[1] * ncell[2]},
        _owning_thread(0), _largest_buffer_index(TRAVELDIRECTION_NUMBER),
        _largest_buffer_size(0),
        _cell_volume(_cell_size[0] * _cell_size[1] * _cell_size[2]),
        _inverse_cell_volume(1. / _cell_volume),
        _cell_areas{_cell_size[1] * _cell_size[2],
                    _cell_size[0] * _cell_size[2],
                    _cell_size[0] * _cell_size[1]} {

#ifdef DENSITYGRID_EDGECOST
    // initialize edge communication costs
    for (int i = 0; i < TRAVELDIRECTION_NUMBER; ++i) {
      _communication_cost[i] = 0;
    }
#endif

    // allocate memory for data arrays
    const int tot_ncell = _number_of_cells[3] * ncell[0];
    _number_density = new double[tot_ncell];
    _neutral_fraction = new double[tot_ncell];
    _intensity_integral = new double[tot_ncell];
    subgrid_cell_lock_init(tot_ncell);

    // initialize data arrays
    for (int i = 0; i < tot_ncell; ++i) {
      // initial density (homogeneous density)
      _number_density[i] = 1.e8;
      // initial neutral fraction (low value, to allow radiation to effectively
      // cover the entire volume initially)
      _neutral_fraction[i] = 1.e-6;
      _intensity_integral[i] = 0.;
    }

#ifndef NO_HYDRO
    _conserved_variables = new float[tot_ncell * 5];
    _delta_conserved_variables = new float[tot_ncell * 5];
    _primitive_variables = new float[tot_ncell * 5];
#ifdef SECOND_ORDER
    _primitive_variable_gradients = new float[tot_ncell * 15];
    _primitive_variable_limiters = new float[tot_ncell * 10];
#endif

    for (int i = 0; i < tot_ncell; ++i) {
      _conserved_variables[5 * i] = 0.;
      _conserved_variables[5 * i + 1] = 0.;
      _conserved_variables[5 * i + 2] = 0.;
      _conserved_variables[5 * i + 3] = 0.;
      _conserved_variables[5 * i + 4] = 0.;

      _delta_conserved_variables[5 * i] = 0.;
      _delta_conserved_variables[5 * i + 1] = 0.;
      _delta_conserved_variables[5 * i + 2] = 0.;
      _delta_conserved_variables[5 * i + 3] = 0.;
      _delta_conserved_variables[5 * i + 4] = 0.;

      _primitive_variables[5 * i] = 0.;
      _primitive_variables[5 * i + 1] = 0.;
      _primitive_variables[5 * i + 2] = 0.;
      _primitive_variables[5 * i + 3] = 0.;
      _primitive_variables[5 * i + 4] = 0.;

#ifdef SECOND_ORDER
      _primitive_variable_gradients[15 * i] = 0.;
      _primitive_variable_gradients[15 * i + 1] = 0.;
      _primitive_variable_gradients[15 * i + 2] = 0.;
      _primitive_variable_gradients[15 * i + 3] = 0.;
      _primitive_variable_gradients[15 * i + 4] = 0.;
      _primitive_variable_gradients[15 * i + 5] = 0.;
      _primitive_variable_gradients[15 * i + 6] = 0.;
      _primitive_variable_gradients[15 * i + 7] = 0.;
      _primitive_variable_gradients[15 * i + 8] = 0.;
      _primitive_variable_gradients[15 * i + 9] = 0.;
      _primitive_variable_gradients[15 * i + 10] = 0.;
      _primitive_variable_gradients[15 * i + 11] = 0.;
      _primitive_variable_gradients[15 * i + 12] = 0.;
      _primitive_variable_gradients[15 * i + 13] = 0.;
      _primitive_variable_gradients[15 * i + 14] = 0.;

      _primitive_variable_limiters[10 * i] = DBL_MAX;
      _primitive_variable_limiters[10 * i + 1] = -DBL_MAX;
      _primitive_variable_limiters[10 * i + 2] = DBL_MAX;
      _primitive_variable_limiters[10 * i + 3] = -DBL_MAX;
      _primitive_variable_limiters[10 * i + 4] = DBL_MAX;
      _primitive_variable_limiters[10 * i + 5] = -DBL_MAX;
      _primitive_variable_limiters[10 * i + 6] = DBL_MAX;
      _primitive_variable_limiters[10 * i + 7] = -DBL_MAX;
      _primitive_variable_limiters[10 * i + 8] = DBL_MAX;
      _primitive_variable_limiters[10 * i + 9] = -DBL_MAX;
#endif // SECOND_ORDER
    }
#endif // NO_HYDRO
  }

  /**
   * @brief Copy constructor.
   *
   * @param original DensitySubGrid to copy.
   */
  inline DensitySubGrid(const DensitySubGrid &original)
      : _computational_cost(0), _anchor{original._anchor[0],
                                        original._anchor[1],
                                        original._anchor[2]},
        _cell_size{original._cell_size[0], original._cell_size[1],
                   original._cell_size[2]},
        _inv_cell_size{original._inv_cell_size[0], original._inv_cell_size[1],
                       original._inv_cell_size[2]},
        _number_of_cells{
            original._number_of_cells[0], original._number_of_cells[1],
            original._number_of_cells[2], original._number_of_cells[3]},
        _owning_thread(original._owning_thread),
        _largest_buffer_index(TRAVELDIRECTION_NUMBER), _largest_buffer_size(0),
        _cell_volume(original._cell_volume),
        _inverse_cell_volume(original._inverse_cell_volume),
        _cell_areas{original._cell_areas[0], original._cell_areas[1],
                    original._cell_areas[2]} {

#ifdef DENSITYGRID_EDGECOST
    // initialize edge communication costs
    for (int i = 0; i < TRAVELDIRECTION_NUMBER; ++i) {
      _communication_cost[i] = 0;
    }
#endif

    const int tot_ncell = _number_of_cells[3] * _number_of_cells[0];
    _number_density = new double[tot_ncell];
    _neutral_fraction = new double[tot_ncell];
    _intensity_integral = new double[tot_ncell];
    subgrid_cell_lock_init(tot_ncell);

    // copy data arrays
    for (int i = 0; i < tot_ncell; ++i) {
      // initial density (homogeneous density)
      _number_density[i] = original._number_density[i];
      _neutral_fraction[i] = original._neutral_fraction[i];
      _intensity_integral[i] = 0.;
    }

#ifndef NO_HYDRO
    _conserved_variables = new float[tot_ncell * 5];
    _delta_conserved_variables = new float[tot_ncell * 5];
    _primitive_variables = new float[tot_ncell * 5];
#ifdef SECOND_ORDER
    _primitive_variable_gradients = new float[tot_ncell * 15];
    _primitive_variable_limiters = new float[10 * tot_ncell];
#endif

    for (int i = 0; i < tot_ncell; ++i) {
      _conserved_variables[5 * i] = original._conserved_variables[5 * i];
      _conserved_variables[5 * i + 1] =
          original._conserved_variables[5 * i + 1];
      _conserved_variables[5 * i + 2] =
          original._conserved_variables[5 * i + 2];
      _conserved_variables[5 * i + 3] =
          original._conserved_variables[5 * i + 3];
      _conserved_variables[5 * i + 4] =
          original._conserved_variables[5 * i + 4];

      _delta_conserved_variables[5 * i] =
          original._delta_conserved_variables[5 * i];
      _delta_conserved_variables[5 * i + 1] =
          original._delta_conserved_variables[5 * i + 1];
      _delta_conserved_variables[5 * i + 2] =
          original._delta_conserved_variables[5 * i + 2];
      _delta_conserved_variables[5 * i + 3] =
          original._delta_conserved_variables[5 * i + 3];
      _delta_conserved_variables[5 * i + 4] =
          original._delta_conserved_variables[5 * i + 4];

      _primitive_variables[5 * i] = original._primitive_variables[5 * i];
      _primitive_variables[5 * i + 1] =
          original._primitive_variables[5 * i + 1];
      _primitive_variables[5 * i + 2] =
          original._primitive_variables[5 * i + 2];
      _primitive_variables[5 * i + 3] =
          original._primitive_variables[5 * i + 3];
      _primitive_variables[5 * i + 4] =
          original._primitive_variables[5 * i + 4];

#ifdef SECOND_ORDER
      _primitive_variable_gradients[15 * i] =
          original._primitive_variable_gradients[15 * i];
      _primitive_variable_gradients[15 * i + 1] =
          original._primitive_variable_gradients[15 * i + 1];
      _primitive_variable_gradients[15 * i + 2] =
          original._primitive_variable_gradients[15 * i + 2];
      _primitive_variable_gradients[15 * i + 3] =
          original._primitive_variable_gradients[15 * i + 3];
      _primitive_variable_gradients[15 * i + 4] =
          original._primitive_variable_gradients[15 * i + 4];
      _primitive_variable_gradients[15 * i + 5] =
          original._primitive_variable_gradients[15 * i + 5];
      _primitive_variable_gradients[15 * i + 6] =
          original._primitive_variable_gradients[15 * i + 6];
      _primitive_variable_gradients[15 * i + 7] =
          original._primitive_variable_gradients[15 * i + 7];
      _primitive_variable_gradients[15 * i + 8] =
          original._primitive_variable_gradients[15 * i + 8];
      _primitive_variable_gradients[15 * i + 9] =
          original._primitive_variable_gradients[15 * i + 9];
      _primitive_variable_gradients[15 * i + 10] =
          original._primitive_variable_gradients[15 * i + 10];
      _primitive_variable_gradients[15 * i + 11] =
          original._primitive_variable_gradients[15 * i + 12];
      _primitive_variable_gradients[15 * i + 12] =
          original._primitive_variable_gradients[15 * i + 13];
      _primitive_variable_gradients[15 * i + 13] =
          original._primitive_variable_gradients[15 * i + 14];
      _primitive_variable_gradients[15 * i + 14] =
          original._primitive_variable_gradients[15 * i + 14];

      _primitive_variable_limiters[10 * i] =
          original._primitive_variable_limiters[10 * i];
      _primitive_variable_limiters[10 * i + 1] =
          original._primitive_variable_limiters[10 * i + 1];
      _primitive_variable_limiters[10 * i + 2] =
          original._primitive_variable_limiters[10 * i + 2];
      _primitive_variable_limiters[10 * i + 3] =
          original._primitive_variable_limiters[10 * i + 3];
      _primitive_variable_limiters[10 * i + 4] =
          original._primitive_variable_limiters[10 * i + 4];
      _primitive_variable_limiters[10 * i + 5] =
          original._primitive_variable_limiters[10 * i + 5];
      _primitive_variable_limiters[10 * i + 6] =
          original._primitive_variable_limiters[10 * i + 6];
      _primitive_variable_limiters[10 * i + 7] =
          original._primitive_variable_limiters[10 * i + 7];
      _primitive_variable_limiters[10 * i + 8] =
          original._primitive_variable_limiters[10 * i + 8];
      _primitive_variable_limiters[10 * i + 9] =
          original._primitive_variable_limiters[10 * i + 9];
#endif // SECOND_ORDER
    }
#endif // NO_HYDRO
  }

  /**
   * @brief Destructor.
   */
  inline ~DensitySubGrid() {
    // deallocate data arrays
    delete[] _number_density;
    delete[] _neutral_fraction;
    delete[] _intensity_integral;
    subgrid_cell_lock_destroy();
#ifndef NO_HYDRO
    delete[] _conserved_variables;
    delete[] _delta_conserved_variables;
    delete[] _primitive_variables;
#ifdef SECOND_ORDER
    delete[] _primitive_variable_gradients;
    delete[] _primitive_variable_limiters;
#endif
#endif // NO_HYDRO
  }

  /**
   * @brief Get the number of cells in a single subgrid.
   *
   * @return Number of cells in the subgrid.
   */
  inline size_t get_number_of_cells() const {
    return _number_of_cells[0] * _number_of_cells[3];
  }

  /**
   * @brief Get a read-only access pointer to the neutral fractions stored in
   * this subgrid.
   *
   * @return Read-only pointer to the neutral fractions.
   */
  inline const double *get_neutral_fraction() const {
    return _neutral_fraction;
  }

  /**
   * @brief Get a read-only access pointer to the primitive variables stored in
   * this subgrid.
   *
   * @return Read-only pointer to the primitive fractions.
   */
  inline const float *get_primitives() const { return _primitive_variables; }

  /**
   * @brief Get the midpoint of the subgrid box for domain decomposition
   * plotting.
   *
   * @param midpoint Variable to set.
   */
  inline void get_midpoint(double midpoint[3]) const {
    midpoint[0] = _anchor[0] + 0.5 * _number_of_cells[0] * _cell_size[0];
    midpoint[1] = _anchor[1] + 0.5 * _number_of_cells[1] * _cell_size[1];
    midpoint[2] = _anchor[2] + 0.5 * _number_of_cells[2] * _cell_size[2];
  }

  /**
   * @brief Get the neighbour for the given direction.
   *
   * @param output_direction TravelDirection.
   * @return Index of the neighbouring subgrid for that direction.
   */
  inline unsigned int get_neighbour(const int output_direction) const {
    myassert(output_direction >= 0 && output_direction < TRAVELDIRECTION_NUMBER,
             "output_direction: " << output_direction);
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

    myassert(output_direction >= 0 && output_direction < TRAVELDIRECTION_NUMBER,
             "output_direction: " << output_direction);
    _ngbs[output_direction] = ngb;
  }

#ifdef DENSITYGRID_EDGECOST
  /**
   * @brief Add the given communication cost to the given edge.
   *
   * @param direction Neighbour direction.
   * @param cost Cost to add.
   */
  inline void add_communication_cost(const int direction,
                                     const unsigned int cost) {
    _communication_cost[direction] += cost;
  }

  /**
   * @brief Get the communication cost for the given edge.
   *
   * @param direction Neighbour direction.
   * @return Communication cost.
   */
  inline unsigned int get_communication_cost(const int direction) const {
    return _communication_cost[direction];
  }

  /**
   * @brief Reset the communication costs for all edges.
   */
  inline void reset_communication_costs() {
    for (int i = 0; i < TRAVELDIRECTION_NUMBER; ++i) {
      _communication_cost[i] = 0;
    }
  }
#endif

  /**
   * @brief Get the active buffer for the given direction.
   *
   * @param direction TravelDirection.
   * @return Index of the corresponding active buffer.
   */
  inline unsigned int get_active_buffer(const int direction) const {
    return _active_buffers[direction];
  }

  /**
   * @brief Set the active buffer for the given direction.
   *
   * This method will also argsort the active buffers.
   *
   * @param direction TravelDirection.
   * @param index Index of the corresponding active buffer.
   */
  inline void set_active_buffer(const int direction, const unsigned int index) {
    _active_buffers[direction] = index;
  }

  /**
   * @brief Set the size and index of the largest active buffer.
   *
   * @param index Index of the largest active buffer.
   * @param size Size of the largest active buffer.
   */
  inline void set_largest_buffer(const unsigned char index,
                                 const unsigned int size) {
    _largest_buffer_index = index;
    _largest_buffer_size = size;
  }

  /**
   * @brief Get the index of the largest active buffer.
   *
   * @return Index of the largest active buffer.
   */
  inline unsigned char get_largest_buffer_index() const {
    return _largest_buffer_index;
  }

  /**
   * @brief Get the size of the largest active buffer.
   *
   * @return Size of the largest active buffer.
   */
  inline unsigned int get_largest_buffer_size() const {
    return _largest_buffer_size;
  }

  /**
   * @brief Get the size of a DensitySubGrid when it is communicated over MPI.
   *
   * @return Size of a DensitySubGrid that is communicated over MPI.
   */
  inline int get_MPI_size() const {
    return DENSITYSUBGRID_FIXED_MPI_SIZE + DENSITYSUBGRID_ELEMENT_SIZE *
                                               _number_of_cells[0] *
                                               _number_of_cells[3];
  }

  /**
   * @brief Get the size of a DensitySubGrid when it is stored in memory.
   *
   * @return Size of a DensitySubGrid that is stored in memory (in bytes).
   */
  inline size_t get_memory_size() const {
    return DENSITYSUBGRID_FIXED_SIZE + DENSITYSUBGRID_ELEMENT_SIZE *
                                           _number_of_cells[0] *
                                           _number_of_cells[3];
  }

#ifdef WITH_MPI
  /**
   * @brief Pack the DensitySubGrid into the given MPI buffer for communication.
   *
   * @param buffer MPI buffer (should at least have size get_MPI_size()).
   * @param buffer_size Actual size of the buffer.
   */
  inline void pack(char *buffer, const int buffer_size) {

    myassert(buffer_size >= get_MPI_size(), "Buffer too small!");

    int buffer_position = 0;
    MPI_Pack(&_computational_cost, 1, MPI_UNSIGNED_LONG, buffer, buffer_size,
             &buffer_position, MPI_COMM_WORLD);
    MPI_Pack(_anchor, 3, MPI_DOUBLE, buffer, buffer_size, &buffer_position,
             MPI_COMM_WORLD);
    MPI_Pack(_cell_size, 3, MPI_DOUBLE, buffer, buffer_size, &buffer_position,
             MPI_COMM_WORLD);
    MPI_Pack(_inv_cell_size, 3, MPI_DOUBLE, buffer, buffer_size,
             &buffer_position, MPI_COMM_WORLD);
    MPI_Pack(_number_of_cells, 4, MPI_INT, buffer, buffer_size,
             &buffer_position, MPI_COMM_WORLD);
    MPI_Pack(_ngbs, TRAVELDIRECTION_NUMBER, MPI_UNSIGNED, buffer, buffer_size,
             &buffer_position, MPI_COMM_WORLD);

    const int tot_num_cells = _number_of_cells[0] * _number_of_cells[3];
    MPI_Pack(_number_density, tot_num_cells, MPI_DOUBLE, buffer, buffer_size,
             &buffer_position, MPI_COMM_WORLD);
    MPI_Pack(_neutral_fraction, tot_num_cells, MPI_DOUBLE, buffer, buffer_size,
             &buffer_position, MPI_COMM_WORLD);
    MPI_Pack(_intensity_integral, tot_num_cells, MPI_DOUBLE, buffer,
             buffer_size, &buffer_position, MPI_COMM_WORLD);

    MPI_Pack(&_cell_volume, 1, MPI_DOUBLE, buffer, buffer_size,
             &buffer_position, MPI_COMM_WORLD);
    MPI_Pack(&_inverse_cell_volume, 1, MPI_DOUBLE, buffer, buffer_size,
             &buffer_position, MPI_COMM_WORLD);
    MPI_Pack(_cell_areas, 3, MPI_DOUBLE, buffer, buffer_size, &buffer_position,
             MPI_COMM_WORLD);
    MPI_Pack(_conserved_variables, 5 * tot_num_cells, MPI_DOUBLE, buffer,
             buffer_size, &buffer_position, MPI_COMM_WORLD);
    MPI_Pack(_delta_conserved_variables, 5 * tot_num_cells, MPI_DOUBLE, buffer,
             buffer_size, &buffer_position, MPI_COMM_WORLD);
    MPI_Pack(_primitive_variables, 5 * tot_num_cells, MPI_DOUBLE, buffer,
             buffer_size, &buffer_position, MPI_COMM_WORLD);
    MPI_Pack(_primitive_variable_gradients, 15 * tot_num_cells, MPI_DOUBLE,
             buffer, buffer_size, &buffer_position, MPI_COMM_WORLD);
    MPI_Pack(_primitive_variable_limiters, 10 * tot_num_cells, MPI_DOUBLE,
             buffer, buffer_size, &buffer_position, MPI_COMM_WORLD);
  }

  /**
   * @brief Unpack the given MPI communication buffer into this DensitySubGrid.
   *
   * Note that we cannot check if the buffer is big enough to contain the
   * correct number of variables.
   *
   * @param buffer MPI buffer.
   * @param buffer_size Actual size of the buffer.
   */
  inline void unpack(char *buffer, const int buffer_size) {
    int buffer_position = 0;
    MPI_Unpack(buffer, buffer_size, &buffer_position, &_computational_cost, 1,
               MPI_UNSIGNED_LONG, MPI_COMM_WORLD);
    MPI_Unpack(buffer, buffer_size, &buffer_position, _anchor, 3, MPI_DOUBLE,
               MPI_COMM_WORLD);
    MPI_Unpack(buffer, buffer_size, &buffer_position, _cell_size, 3, MPI_DOUBLE,
               MPI_COMM_WORLD);
    MPI_Unpack(buffer, buffer_size, &buffer_position, _inv_cell_size, 3,
               MPI_DOUBLE, MPI_COMM_WORLD);
    unsigned int new_number_of_cells[4];
    MPI_Unpack(buffer, buffer_size, &buffer_position, new_number_of_cells, 4,
               MPI_INT, MPI_COMM_WORLD);
    MPI_Unpack(buffer, buffer_size, &buffer_position, _ngbs,
               TRAVELDIRECTION_NUMBER, MPI_UNSIGNED, MPI_COMM_WORLD);

    const int tot_num_cells = new_number_of_cells[0] * new_number_of_cells[3];
    // DensitySubgrids don't necessarily have the same size, so we make sure
    // we adjust the size of the cell arrays if they are different
    const int old_num_cells = _number_of_cells[0] * _number_of_cells[3];
    if (tot_num_cells != old_num_cells) {
      _number_of_cells[0] = new_number_of_cells[0];
      _number_of_cells[1] = new_number_of_cells[1];
      _number_of_cells[2] = new_number_of_cells[2];
      _number_of_cells[3] = new_number_of_cells[3];
      delete[] _number_density;
      delete[] _neutral_fraction;
      delete[] _intensity_integral;
      _number_density = new double[tot_num_cells];
      _neutral_fraction = new double[tot_num_cells];
      _intensity_integral = new double[tot_num_cells];

      delete[] _conserved_variables;
      delete[] _delta_conserved_variables;
      delete[] _primitive_variables;
      delete[] _primitive_variable_gradients;
      delete[] _primitive_variable_limiters;
      _conserved_variables = new double[5 * tot_num_cells];
      _delta_conserved_variables = new double[5 * tot_num_cells];
      _primitive_variables = new double[5 * tot_num_cells];
      _primitive_variable_gradients = new double[15 * tot_num_cells];
      _primitive_variable_limiters = new double[10 * tot_num_cells];
    }
    MPI_Unpack(buffer, buffer_size, &buffer_position, _number_density,
               tot_num_cells, MPI_DOUBLE, MPI_COMM_WORLD);
    MPI_Unpack(buffer, buffer_size, &buffer_position, _neutral_fraction,
               tot_num_cells, MPI_DOUBLE, MPI_COMM_WORLD);
    MPI_Unpack(buffer, buffer_size, &buffer_position, _intensity_integral,
               tot_num_cells, MPI_DOUBLE, MPI_COMM_WORLD);

    MPI_Unpack(buffer, buffer_size, &buffer_position, &_cell_volume, 1,
               MPI_DOUBLE, MPI_COMM_WORLD);
    MPI_Unpack(buffer, buffer_size, &buffer_position, &_inverse_cell_volume, 1,
               MPI_DOUBLE, MPI_COMM_WORLD);
    MPI_Unpack(buffer, buffer_size, &buffer_position, _cell_areas, 3,
               MPI_DOUBLE, MPI_COMM_WORLD);
    MPI_Unpack(buffer, buffer_size, &buffer_position, _conserved_variables,
               5 * tot_num_cells, MPI_DOUBLE, MPI_COMM_WORLD);
    MPI_Unpack(buffer, buffer_size, &buffer_position,
               _delta_conserved_variables, 5 * tot_num_cells, MPI_DOUBLE,
               MPI_COMM_WORLD);
    MPI_Unpack(buffer, buffer_size, &buffer_position, _primitive_variables,
               5 * tot_num_cells, MPI_DOUBLE, MPI_COMM_WORLD);
    MPI_Unpack(buffer, buffer_size, &buffer_position,
               _primitive_variable_gradients, 15 * tot_num_cells, MPI_DOUBLE,
               MPI_COMM_WORLD);
    MPI_Unpack(buffer, buffer_size, &buffer_position,
               _primitive_variable_limiters, 10 * tot_num_cells, MPI_DOUBLE,
               MPI_COMM_WORLD);
  }
#endif

  /**
   * @brief Sync the neutral fractions with the given subgrid.
   *
   * @param original Original subgrid from which to copy.
   */
  inline void update_neutral_fractions(const DensitySubGrid &original) {
    const int tot_ncell = _number_of_cells[3] * _number_of_cells[0];
    for (int i = 0; i < tot_ncell; ++i) {
      _neutral_fraction[i] = original._neutral_fraction[i];
      _intensity_integral[i] = 0.;
    }
  }

  /**
   * @brief Add the contributions of subgrid copies to the intensity integrals.
   *
   * @param copy Subgrid copy from which to read.
   */
  inline void update_intensities(const DensitySubGrid &copy) {

#ifdef DENSITYGRID_EDGECOST
    for (int i = 0; i < TRAVELDIRECTION_NUMBER; ++i) {
      _communication_cost[i] += copy._communication_cost[i];
    }
#endif

    const int tot_ncell = _number_of_cells[3] * _number_of_cells[0];
    for (int i = 0; i < tot_ncell; ++i) {
      _intensity_integral[i] += copy._intensity_integral[i];
    }
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
   * @brief Get the box containing the sub grid.
   *
   * @param box Output array (in m).
   */
  inline void get_grid_box(double *box) const {
    box[0] = _anchor[0];
    box[1] = _anchor[1];
    box[2] = _anchor[2];
    box[3] = _cell_size[0] * _number_of_cells[0];
    box[4] = _cell_size[1] * _number_of_cells[1];
    box[5] = _cell_size[2] * _number_of_cells[2];
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
   * @param input_direction Direction from which the photon enters the grid.
   * @return TravelDirection of the photon after it has traversed this grid.
   */
  inline int interact(Photon &photon, const int input_direction) {

    myassert(input_direction >= 0 && input_direction < TRAVELDIRECTION_NUMBER,
             "input_direction: " << input_direction);

    // get some photon variables
    const double *direction = photon.get_direction();

    myassert(TravelDirections::is_compatible_input_direction(direction,
                                                             input_direction),
             "direction: " << direction[0] << " " << direction[1] << " "
                           << direction[2]
                           << ", input_direction: " << input_direction);

    const double inverse_direction[3] = {1. / direction[0], 1. / direction[1],
                                         1. / direction[2]};
    // NOTE: position is relative w.r.t. _anchor!!!
    double position[3] = {photon.get_position()[0] - _anchor[0],
                          photon.get_position()[1] - _anchor[1],
                          photon.get_position()[2] - _anchor[2]};
    double tau_done = 0.;
    const double tau_target = photon.get_target_optical_depth();

    myassert(tau_done < tau_target,
             "tau_done: " << tau_done << ", target: " << tau_target);

    const double cross_section = photon.get_photoionization_cross_section();
    const double photon_weight = photon.get_weight();
    // get the indices of the first cell on the photon's path
    int three_index[3];
    int active_cell = get_start_index(position, input_direction, three_index);

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
      subgrid_cell_lock_lock(active_cell);
      _intensity_integral[active_cell] += lminsigma * photon_weight;
      subgrid_cell_lock_unlock(active_cell);
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
      active_cell = get_one_index(three_index);
    }
    // update photon quantities
    photon.set_target_optical_depth(tau_target - tau_done);
    photon.set_position(position[0] + _anchor[0], position[1] + _anchor[1],
                        position[2] + _anchor[2]);
    // get the outgoing direction
    int output_direction;
    if (tau_done >= tau_target) {
      output_direction = TRAVELDIRECTION_INSIDE;
    } else {
      output_direction = get_output_direction(three_index);
    }

    myassert(TravelDirections::is_compatible_output_direction(direction,
                                                              output_direction),
             "wrong output direction!");

    return output_direction;
  }

  /**
   * @brief Let the given Photon travel through the density grid without
   * interacting with the grid.
   *
   * @param photon Photon.
   * @param input_direction Direction from which the photon enters the grid.
   * @return TravelDirection of the photon after it has traversed this grid.
   */
  inline int propagate(Photon &photon, const int input_direction) {

    myassert(input_direction >= 0 && input_direction < TRAVELDIRECTION_NUMBER,
             "input_direction: " << input_direction);

    // get some photon variables
    const double *direction = photon.get_direction();

    myassert(TravelDirections::is_compatible_input_direction(direction,
                                                             input_direction),
             "direction: " << direction[0] << " " << direction[1] << " "
                           << direction[2]
                           << ", input_direction: " << input_direction);

    const double inverse_direction[3] = {1. / direction[0], 1. / direction[1],
                                         1. / direction[2]};
    // NOTE: position is relative w.r.t. _anchor!!!
    double position[3] = {photon.get_position()[0] - _anchor[0],
                          photon.get_position()[1] - _anchor[1],
                          photon.get_position()[2] - _anchor[2]};
    double tau_done = 0.;
    const double tau_target = photon.get_target_optical_depth();

    myassert(tau_done < tau_target,
             "tau_done: " << tau_done << ", target: " << tau_target);

    const double cross_section = photon.get_photoionization_cross_section();
    // get the indices of the first cell on the photon's path
    int three_index[3];
    int active_cell = get_start_index(position, input_direction, three_index);

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
      active_cell = get_one_index(three_index);
    }
    // update photon quantities
    photon.set_target_optical_depth(tau_target - tau_done);
    photon.set_position(position[0] + _anchor[0], position[1] + _anchor[1],
                        position[2] + _anchor[2]);
    // get the outgoing direction
    int output_direction;
    if (tau_done >= tau_target) {
      output_direction = TRAVELDIRECTION_INSIDE;
    } else {
      output_direction = get_output_direction(three_index);
    }

    myassert(TravelDirections::is_compatible_output_direction(direction,
                                                              output_direction),
             "wrong output direction!");

    return output_direction;
  }

  /**
   * @brief Add the optical depth contribution for traversing this subgrid to
   * the total optical depth of the given photon.
   *
   * @param photon Photon.
   * @param input_direction Direction from which the photon enters the grid.
   * @return TravelDirection of the photon after it has traversed this grid.
   */
  inline int compute_optical_depth(Photon &photon, const int input_direction) {

    myassert(input_direction >= 0 && input_direction < TRAVELDIRECTION_NUMBER,
             "input_direction: " << input_direction);

    // get some photon variables
    const double *direction = photon.get_direction();

    myassert(TravelDirections::is_compatible_input_direction(direction,
                                                             input_direction),
             "direction: " << direction[0] << " " << direction[1] << " "
                           << direction[2]
                           << ", input_direction: " << input_direction);

    const double inverse_direction[3] = {1. / direction[0], 1. / direction[1],
                                         1. / direction[2]};
    // NOTE: position is relative w.r.t. _anchor!!!
    double position[3] = {photon.get_position()[0] - _anchor[0],
                          photon.get_position()[1] - _anchor[1],
                          photon.get_position()[2] - _anchor[2]};
    double tau_done = 0.;

    const double cross_section = photon.get_photoionization_cross_section();
    // get the indices of the first cell on the photon's path
    int three_index[3];
    int active_cell = get_start_index(position, input_direction, three_index);

    myassert(active_cell >= 0 &&
                 active_cell < _number_of_cells[0] * _number_of_cells[3],
             "active_cell: " << active_cell << ", size: "
                             << _number_of_cells[0] * _number_of_cells[3]);

    // enter photon traversal loop
    // double condition:
    //  - photon still in subgrid: is_inside(three_index)
    while (is_inside(three_index)) {
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
      // photon leaves cell
      // update three_index
      for (unsigned char idim = 0; idim < 3; ++idim) {
        if (l[idim] == lmin) {
          three_index[idim] += (direction[idim] > 0.) ? 1 : -1;
        }
      }

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
      active_cell = get_one_index(three_index);
    }
    // update photon quantities
    photon.set_target_optical_depth(photon.get_target_optical_depth() +
                                    tau_done);
    photon.set_position(position[0] + _anchor[0], position[1] + _anchor[1],
                        position[2] + _anchor[2]);
    // get the outgoing direction
    int output_direction = get_output_direction(three_index);

    myassert(TravelDirections::is_compatible_output_direction(direction,
                                                              output_direction),
             "wrong output direction!");

    return output_direction;
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
   * @param luminosity Total ionising luminosity of all sources (in s^-1).
   * @param num_photon Total number of photon packets that was used, used to
   * normalize the intensity estimates.
   */
  inline void compute_neutral_fraction(const double luminosity,
                                       const unsigned int num_photon) {
    // compute the normalization factor:
    // constant source luminosity / number of packets / cell volume
    // if we multiply this with the intensity estimates (unit: m^3), we get a
    // quantity in s^-1
    const double jfac = luminosity / num_photon * _inv_cell_size[0] *
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
                 << _neutral_fraction[index] << "\t" << _number_density[index]
#ifdef NO_HYDRO
                 << "\n";
#else
                 << "\t" << _primitive_variables[5 * index] << "\t"
                 << _primitive_variables[5 * index + 1] << "\t"
                 << _primitive_variables[5 * index + 2] << "\t"
                 << _primitive_variables[5 * index + 3] << "\t"
                 << _primitive_variables[5 * index + 4] << "\n";
#endif
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
          stream.write(reinterpret_cast< char * >(&pos_x), sizeof(double));
          stream.write(reinterpret_cast< char * >(&pos_y), sizeof(double));
          stream.write(reinterpret_cast< char * >(&pos_z), sizeof(double));
          stream.write(reinterpret_cast< char * >(&_neutral_fraction[index]),
                       sizeof(double));
          stream.write(reinterpret_cast< char * >(&_number_density[index]),
                       sizeof(double));
          stream.write(
              reinterpret_cast< char * >(&_primitive_variables[5 * index]),
              sizeof(float));
          stream.write(
              reinterpret_cast< char * >(&_primitive_variables[5 * index + 1]),
              sizeof(float));
          stream.write(
              reinterpret_cast< char * >(&_primitive_variables[5 * index + 2]),
              sizeof(float));
          stream.write(
              reinterpret_cast< char * >(&_primitive_variables[5 * index + 3]),
              sizeof(float));
          stream.write(
              reinterpret_cast< char * >(&_primitive_variables[5 * index + 4]),
              sizeof(float));
        }
      }
    }
  }

  /**
   * @brief Write the neutral fractions to the given memory-mapped file.
   *
   * @param offset Offset within the file (in bytes).
   * @param file Memory-mapped file to write to.
   */
  inline void output_intensities(const size_t offset, MemoryMap &file) {
    size_t cell_offset = 0;
    for (int ix = 0; ix < _number_of_cells[0]; ++ix) {
      // not const since we cast it to a char* below
      double pos_x = _anchor[0] + (ix + 0.5) * _cell_size[0];
      for (int iy = 0; iy < _number_of_cells[1]; ++iy) {
        double pos_y = _anchor[1] + (iy + 0.5) * _cell_size[1];
        for (int iz = 0; iz < _number_of_cells[2]; ++iz) {
          double pos_z = _anchor[2] + (iz + 0.5) * _cell_size[2];
          const int three_index[3] = {ix, iy, iz};
          const int index = get_one_index(three_index);

          file.write(offset + cell_offset, pos_x);
          cell_offset += sizeof(double);
          file.write(offset + cell_offset, pos_y);
          cell_offset += sizeof(double);
          file.write(offset + cell_offset, pos_z);
          cell_offset += sizeof(double);
          file.write(offset + cell_offset, _neutral_fraction[index]);
          cell_offset += sizeof(double);
          file.write(offset + cell_offset, _number_density[index]);
          cell_offset += sizeof(double);
#ifdef NO_HYDRO
          file.write(offset + cell_offset, 0.);
          cell_offset += sizeof(float);
          file.write(offset + cell_offset, 0.);
          cell_offset += sizeof(float);
          file.write(offset + cell_offset, 0.);
          cell_offset += sizeof(float);
          file.write(offset + cell_offset, 0.);
          cell_offset += sizeof(float);
          file.write(offset + cell_offset, 0.);
          cell_offset += sizeof(float);
#else  // NO_HYDRO
          file.write(offset + cell_offset, _primitive_variables[5 * index]);
          cell_offset += sizeof(float);
          file.write(offset + cell_offset, _primitive_variables[5 * index + 1]);
          cell_offset += sizeof(float);
          file.write(offset + cell_offset, _primitive_variables[5 * index + 2]);
          cell_offset += sizeof(float);
          file.write(offset + cell_offset, _primitive_variables[5 * index + 3]);
          cell_offset += sizeof(float);
          file.write(offset + cell_offset, _primitive_variables[5 * index + 4]);
          cell_offset += sizeof(float);
#endif // NO_HYDRO
        }
      }
    }
  }

  /**
   * @brief Get the size (in bytes) of the output array for this subgrid.
   *
   * @return Size (in bytes) that will be output by output_intensities().
   */
  inline size_t get_output_size() const {
    return _number_of_cells[0] * _number_of_cells[1] * _number_of_cells[2] *
           (5 * sizeof(double) + 5 * sizeof(float));
  }

  /**
   * @brief Add the given ammount to the computational cost.
   *
   * @param computational_cost Amount to add.
   */
  inline void add_computational_cost(const unsigned long computational_cost) {
    _computational_cost += computational_cost;
  }

  /**
   * @brief Reset the computational cost.
   */
  inline void reset_computational_cost() { _computational_cost = 0; }

  /**
   * @brief Get the computational cost for this subgrid.
   *
   * @return Computational cost.
   */
  inline unsigned long get_computational_cost() const {
    return _computational_cost;
  }

  /**
   * @brief Get the dependency lock for this subgrid.
   *
   * @return Pointer to the dependency lock for this subgrid.
   */
  inline Lock *get_dependency() { return &_dependency; }

  /**
   * @brief Get the id of the thread that owns this subgrid.
   *
   * @return Id of the thread that owns this subgrid.
   */
  inline int get_owning_thread() const { return _owning_thread; }

  /**
   * @brief Set the id of the thread that owns this subgrid.
   *
   * @param owning_thread Id of the thread that owns this subgrid.
   */
  inline void set_owning_thread(const int owning_thread) {
    _owning_thread = owning_thread;
  }

  /**
   * @brief Check if the given DensitySubGrid is equal to this one.
   *
   * We only compare variables that are communicated over MPI.
   *
   * @param other Other DensitySubGrid.
   */
  inline void check_equal(const DensitySubGrid &other) {

    myassert(_computational_cost == other._computational_cost,
             "Costs not the same!");
    myassert(_anchor[0] == other._anchor[0], "Anchor not the same!");
    myassert(_anchor[1] == other._anchor[1], "Anchor not the same!");
    myassert(_anchor[2] == other._anchor[2], "Anchor not the same!");
    myassert(_cell_size[0] == other._cell_size[0], "Cell size not the same!");
    myassert(_cell_size[1] == other._cell_size[1], "Cell size not the same!");
    myassert(_cell_size[2] == other._cell_size[2], "Cell size not the same!");
    myassert(_inv_cell_size[0] == other._inv_cell_size[0],
             "Inverse cell size not the same!");
    myassert(_inv_cell_size[1] == other._inv_cell_size[1],
             "Inverse cell size not the same!");
    myassert(_inv_cell_size[2] == other._inv_cell_size[2],
             "Inverse cell size not the same!");
    myassert(_number_of_cells[0] == other._number_of_cells[0],
             "Number of cells not the same!");
    myassert(_number_of_cells[1] == other._number_of_cells[1],
             "Number of cells not the same!");
    myassert(_number_of_cells[2] == other._number_of_cells[2],
             "Number of cells not the same!");
    myassert(_number_of_cells[3] == other._number_of_cells[3],
             "Number of cells not the same!");
    for (unsigned int i = 0; i < TRAVELDIRECTION_NUMBER; ++i) {
      myassert(_ngbs[i] == other._ngbs[i], "Neighbours not the same!");
    }
    const int tot_num_cells = _number_of_cells[0] * _number_of_cells[3];
    for (int i = 0; i < tot_num_cells; ++i) {
      myassert(_number_density[i] == other._number_density[i],
               "Number density not the same!");
      myassert(_neutral_fraction[i] == other._neutral_fraction[i],
               "Neutral fraction not the same!");
      myassert(_intensity_integral[i] == other._intensity_integral[i],
               "Intensity integral not the same!");

      for (int j = 0; j < 5; ++j) {
        myassert(_conserved_variables[5 * i + j] ==
                     other._conserved_variables[5 * i + j],
                 "Conserved variable not the same!");
        myassert(_delta_conserved_variables[5 * i + j] ==
                     other._delta_conserved_variables[5 * i + j],
                 "Conserved variable change not the same!");
        myassert(_primitive_variables[5 * i + j] ==
                     other._primitive_variables[5 * i + j],
                 "Primitive variable not the same!");
        for (int k = 0; k < 3; ++k) {
          myassert(_primitive_variable_gradients[15 * i + 3 * j + k] ==
                       other._primitive_variable_gradients[15 * i + 3 * j + k],
                   "Primitive variable gradient not the same!");
        }
        for (int k = 0; k < 2; ++k) {
          myassert(_primitive_variable_limiters[10 * i + 2 * j + k] ==
                       other._primitive_variable_limiters[10 * i + 2 * j + k],
                   "Primitive variable limiters not the same!");
        }
      }
    }
    myassert(_cell_volume == other._cell_volume, "Cell volumes not the same!");
    myassert(_inverse_cell_volume == other._inverse_cell_volume,
             "Inverse cell volumes not the same!");
    myassert(_cell_areas[0] == other._cell_areas[0],
             "Cell areas not the same!");
    myassert(_cell_areas[1] == other._cell_areas[1],
             "Cell areas not the same!");
    myassert(_cell_areas[2] == other._cell_areas[2],
             "Cell areas not the same!");
  }

  /**
   * @brief Initialize the conserved variables for the grid.
   *
   * @param hydro Hydro instance to use.
   */
  inline void initialize_conserved_variables(const Hydro &hydro) {

    const int tot_num_cells = _number_of_cells[0] * _number_of_cells[3];
    for (int i = 0; i < tot_num_cells; ++i) {
      hydro.get_conserved_variables(
          _primitive_variables[5 * i], &_primitive_variables[5 * i + 1],
          _primitive_variables[5 * i + 4], _cell_volume,
          _conserved_variables[5 * i], &_conserved_variables[5 * i + 1],
          _conserved_variables[5 * i + 4]);
    }
  }

  /**
   * @brief Update the conserved variables for all cells in the grid.
   *
   * @param time_step Integration time step size (in s).
   */
  inline void update_conserved_variables(const double time_step) {
    const int tot_num_cells = _number_of_cells[0] * _number_of_cells[3];
    for (int i = 0; i < tot_num_cells; ++i) {
      for (int j = 0; j < 5; ++j) {
        _conserved_variables[5 * i + j] +=
            _delta_conserved_variables[5 * i + j] * time_step;
      }
    }

    // reset all hydro variables (except the conserved and primitive ones)
    for (int i = 0; i < tot_num_cells; ++i) {
      _delta_conserved_variables[5 * i] = 0.;
      _delta_conserved_variables[5 * i + 1] = 0.;
      _delta_conserved_variables[5 * i + 2] = 0.;
      _delta_conserved_variables[5 * i + 3] = 0.;
      _delta_conserved_variables[5 * i + 4] = 0.;

#ifdef SECOND_ORDER
      _primitive_variable_gradients[15 * i] = 0.;
      _primitive_variable_gradients[15 * i + 1] = 0.;
      _primitive_variable_gradients[15 * i + 2] = 0.;
      _primitive_variable_gradients[15 * i + 3] = 0.;
      _primitive_variable_gradients[15 * i + 4] = 0.;
      _primitive_variable_gradients[15 * i + 5] = 0.;
      _primitive_variable_gradients[15 * i + 6] = 0.;
      _primitive_variable_gradients[15 * i + 7] = 0.;
      _primitive_variable_gradients[15 * i + 8] = 0.;
      _primitive_variable_gradients[15 * i + 9] = 0.;
      _primitive_variable_gradients[15 * i + 10] = 0.;
      _primitive_variable_gradients[15 * i + 11] = 0.;
      _primitive_variable_gradients[15 * i + 12] = 0.;
      _primitive_variable_gradients[15 * i + 13] = 0.;
      _primitive_variable_gradients[15 * i + 14] = 0.;

      _primitive_variable_limiters[10 * i] = DBL_MAX;
      _primitive_variable_limiters[10 * i + 1] = -DBL_MAX;
      _primitive_variable_limiters[10 * i + 2] = DBL_MAX;
      _primitive_variable_limiters[10 * i + 3] = -DBL_MAX;
      _primitive_variable_limiters[10 * i + 4] = DBL_MAX;
      _primitive_variable_limiters[10 * i + 5] = -DBL_MAX;
      _primitive_variable_limiters[10 * i + 6] = DBL_MAX;
      _primitive_variable_limiters[10 * i + 7] = -DBL_MAX;
      _primitive_variable_limiters[10 * i + 8] = DBL_MAX;
      _primitive_variable_limiters[10 * i + 9] = -DBL_MAX;
#endif // SECOND_ORDER
    }
  }

  /**
   * @brief Update the primitive variables for the grid.
   *
   * @param hydro Hydro instance to use.
   */
  inline void update_primitive_variables(const Hydro &hydro) {
    const int tot_num_cells = _number_of_cells[0] * _number_of_cells[3];
    for (int i = 0; i < tot_num_cells; ++i) {
      hydro.get_primitive_variables(
          _conserved_variables[5 * i], &_conserved_variables[5 * i + 1],
          _conserved_variables[5 * i + 4], _inverse_cell_volume,
          _primitive_variables[5 * i], &_primitive_variables[5 * i + 1],
          _primitive_variables[5 * i + 4]);
    }
  }

  /**
   * @brief Half time step prediction for the primitive variables.
   *
   * @param hydro Hydro instance to use.
   * @param timestep Half system time step (in s).
   */
  inline void predict_primitive_variables(const Hydro &hydro,
                                          const double timestep) {
#ifdef SECOND_ORDER
    const int tot_num_cells = _number_of_cells[0] * _number_of_cells[3];
    for (int i = 0; i < tot_num_cells; ++i) {
      hydro.predict_primitive_variables(&_primitive_variables[5 * i],
                                        &_primitive_variable_gradients[15 * i],
                                        timestep);
    }
#endif // SECOND_ORDER
  }

  /**
   * @brief Apply the slope limiter to all primitive variable gradients.
   *
   * @param hydro Hydro instance to use.
   */
  inline void apply_slope_limiter(const Hydro &hydro) {
#ifdef SECOND_ORDER
    const int tot_num_cells = _number_of_cells[0] * _number_of_cells[3];
    for (int i = 0; i < tot_num_cells; ++i) {
      hydro.apply_slope_limiter(
          &_primitive_variables[5 * i], &_primitive_variable_gradients[15 * i],
          &_primitive_variable_limiters[10 * i], _cell_size);
    }
#endif // SECOND_ORDER
  }

  /**
   * @brief Get the midpoint of the cell with the given index.
   *
   * @param index Index of a cell.
   * @param midpoint Coordinates of the midpoint of that cell (in m).
   */
  inline void get_cell_midpoint(const unsigned int index, double midpoint[3]) {

    int three_index[3];
    get_three_index(index, three_index);
    midpoint[0] = _anchor[0] + (three_index[0] + 0.5) * _cell_size[0];
    midpoint[1] = _anchor[1] + (three_index[1] + 0.5) * _cell_size[1];
    midpoint[2] = _anchor[2] + (three_index[2] + 0.5) * _cell_size[2];
  }

  /**
   * @brief Set the primitive variables for the cell with the given index.
   *
   * @param index Index of a cell.
   * @param density Density (in kg m^-3).
   * @param velocity Velocity (in m s^-1).
   * @param pressure Pressure (in kg m^-1 s^-2).
   */
  inline void set_primitive_variables(const unsigned int index,
                                      const double density,
                                      const double velocity[3],
                                      const double pressure) {

    _primitive_variables[5 * index] = density;
    _primitive_variables[5 * index + 1] = velocity[0];
    _primitive_variables[5 * index + 2] = velocity[1];
    _primitive_variables[5 * index + 3] = velocity[2];
    _primitive_variables[5 * index + 4] = pressure;
  }

  /**
   * @brief Set the primitive variables for all cells based on the given initial
   * condition.
   *
   * @param ic HydroIC to use.
   */
  template < typename _ic_ >
  inline void set_primitive_variables(const _ic_ &ic) {
    const int tot_num_cells = _number_of_cells[0] * _number_of_cells[3];
    for (int i = 0; i < tot_num_cells; ++i) {
      double midpoint[3];
      get_cell_midpoint(i, midpoint);
      ic.set_primitive_variables(midpoint, &_primitive_variables[5 * i]);
    }
  }

  /**
   * @brief Compute the hydrodynamical fluxes for all interfaces inside the
   * subgrid.
   *
   * @param hydro Hydro instance to use.
   */
  inline void inner_flux_sweep(const Hydro &hydro) {

    // we do three separate sweeps: one for every coordinate direction
    for (int ix = 0; ix < _number_of_cells[0] - 1; ++ix) {
      for (int iy = 0; iy < _number_of_cells[1]; ++iy) {
        for (int iz = 0; iz < _number_of_cells[2]; ++iz) {
          const unsigned int index000 =
              ix * _number_of_cells[3] + iy * _number_of_cells[2] + iz;
          const unsigned int index100 =
              (ix + 1) * _number_of_cells[3] + iy * _number_of_cells[2] + iz;
#ifdef SECOND_ORDER
          const float *gradL = &_primitive_variable_gradients[15 * index000];
          const float *gradR = &_primitive_variable_gradients[15 * index100];
#else
          const float *gradL = nullptr;
          const float *gradR = nullptr;
#endif
          // x direction
          hydro.do_flux_calculation(0, &_primitive_variables[5 * index000],
                                    gradL, &_primitive_variables[5 * index100],
                                    gradR, _cell_size[0], _cell_areas[0],
                                    &_delta_conserved_variables[5 * index000],
                                    &_delta_conserved_variables[5 * index100]);
        }
      }
    }
    for (int ix = 0; ix < _number_of_cells[0]; ++ix) {
      for (int iy = 0; iy < _number_of_cells[1] - 1; ++iy) {
        for (int iz = 0; iz < _number_of_cells[2]; ++iz) {
          const unsigned int index000 =
              ix * _number_of_cells[3] + iy * _number_of_cells[2] + iz;
          const unsigned int index010 =
              ix * _number_of_cells[3] + (iy + 1) * _number_of_cells[2] + iz;
#ifdef SECOND_ORDER
          const float *gradL = &_primitive_variable_gradients[15 * index000];
          const float *gradR = &_primitive_variable_gradients[15 * index010];
#else
          const float *gradL = nullptr;
          const float *gradR = nullptr;
#endif
          // y direction
          hydro.do_flux_calculation(1, &_primitive_variables[5 * index000],
                                    gradL, &_primitive_variables[5 * index010],
                                    gradR, _cell_size[1], _cell_areas[1],
                                    &_delta_conserved_variables[5 * index000],
                                    &_delta_conserved_variables[5 * index010]);
        }
      }
    }
    for (int ix = 0; ix < _number_of_cells[0]; ++ix) {
      for (int iy = 0; iy < _number_of_cells[1]; ++iy) {
        for (int iz = 0; iz < _number_of_cells[2] - 1; ++iz) {
          const unsigned int index000 =
              ix * _number_of_cells[3] + iy * _number_of_cells[2] + iz;
          const unsigned int index001 =
              ix * _number_of_cells[3] + iy * _number_of_cells[2] + iz + 1;
#ifdef SECOND_ORDER
          const float *gradL = &_primitive_variable_gradients[15 * index000];
          const float *gradR = &_primitive_variable_gradients[15 * index001];
#else
          const float *gradL = nullptr;
          const float *gradR = nullptr;
#endif
          // z direction
          hydro.do_flux_calculation(2, &_primitive_variables[5 * index000],
                                    gradL, &_primitive_variables[5 * index001],
                                    gradR, _cell_size[2], _cell_areas[2],
                                    &_delta_conserved_variables[5 * index000],
                                    &_delta_conserved_variables[5 * index001]);
        }
      }
    }
  }

  /**
   * @brief Compute the hydrodynamical fluxes for all interfaces at the boundary
   * between this subgrid and the given neighbouring subgrid.
   *
   * @param direction TravelDirection of the neighbour.
   * @param hydro Hydro instance to use.
   * @param neighbour Neighbouring DensitySubGrid.
   */
  inline void outer_flux_sweep(const int direction, const Hydro &hydro,
                               DensitySubGrid &neighbour) {
    int i;
    unsigned int start_index_left, start_index_right, row_increment, row_length,
        column_increment, column_length;
    double dx, A;
    DensitySubGrid *left_grid, *right_grid;
    switch (direction) {
    case TRAVELDIRECTION_FACE_X_P:
      i = 0;
      left_grid = this;
      right_grid = &neighbour;
      start_index_left = (_number_of_cells[0] - 1) * _number_of_cells[3];
      start_index_right = 0;
      row_increment = 1;
      row_length = _number_of_cells[2];
      column_increment = _number_of_cells[2];
      column_length = _number_of_cells[1];
      dx = _cell_size[0];
      A = _cell_areas[0];
      break;
    case TRAVELDIRECTION_FACE_X_N:
      i = 0;
      left_grid = &neighbour;
      right_grid = this;
      start_index_left = (_number_of_cells[0] - 1) * _number_of_cells[3];
      start_index_right = 0;
      row_increment = 1;
      row_length = _number_of_cells[2];
      column_increment = _number_of_cells[2];
      column_length = _number_of_cells[1];
      dx = _cell_size[0];
      A = _cell_areas[0];
      break;
    case TRAVELDIRECTION_FACE_Y_P:
      i = 1;
      left_grid = this;
      right_grid = &neighbour;
      start_index_left = (_number_of_cells[1] - 1) * _number_of_cells[2];
      start_index_right = 0;
      row_increment = 1;
      row_length = _number_of_cells[2];
      column_increment = _number_of_cells[3];
      column_length = _number_of_cells[0];
      dx = _cell_size[1];
      A = _cell_areas[1];
      break;
    case TRAVELDIRECTION_FACE_Y_N:
      i = 1;
      left_grid = &neighbour;
      right_grid = this;
      start_index_left = (_number_of_cells[1] - 1) * _number_of_cells[2];
      start_index_right = 0;
      row_increment = 1;
      row_length = _number_of_cells[2];
      column_increment = _number_of_cells[3];
      column_length = _number_of_cells[0];
      dx = _cell_size[1];
      A = _cell_areas[1];
      break;
    case TRAVELDIRECTION_FACE_Z_P:
      i = 2;
      left_grid = this;
      right_grid = &neighbour;
      start_index_left = _number_of_cells[2] - 1;
      start_index_right = 0;
      row_increment = _number_of_cells[2];
      row_length = _number_of_cells[1];
      column_increment = _number_of_cells[3];
      column_length = _number_of_cells[0];
      dx = _cell_size[2];
      A = _cell_areas[2];
      break;
    case TRAVELDIRECTION_FACE_Z_N:
      i = 2;
      left_grid = &neighbour;
      right_grid = this;
      start_index_left = _number_of_cells[2] - 1;
      start_index_right = 0;
      row_increment = _number_of_cells[2];
      row_length = _number_of_cells[1];
      column_increment = _number_of_cells[3];
      column_length = _number_of_cells[0];
      dx = _cell_size[2];
      A = _cell_areas[2];
      break;
    default:
      cmac_error("Unknown hydro neighbour: %i", direction);
      break;
    }

    // using the index computation below is (much) faster than setting the
    // increment correctly and summing the indices manually
    for (unsigned int ic = 0; ic < column_length; ++ic) {
      for (unsigned int ir = 0; ir < row_length; ++ir) {
        const unsigned int index_left =
            start_index_left + ic * column_increment + ir * row_increment;
        const unsigned int index_right =
            start_index_right + ic * column_increment + ir * row_increment;
#ifdef SECOND_ORDER
        const float *gradL =
            &left_grid->_primitive_variable_gradients[15 * index_left];
        const float *gradR =
            &right_grid_primitive_variable_gradients[15 * index_right];
#else
        const float *gradL = nullptr;
        const float *gradR = nullptr;
#endif
        hydro.do_flux_calculation(
            i, &left_grid->_primitive_variables[5 * index_left], gradL,
            &right_grid->_primitive_variables[5 * index_right], gradR, dx, A,
            &left_grid->_delta_conserved_variables[5 * index_left],
            &right_grid->_delta_conserved_variables[5 * index_right]);
      }
    }
  }

  /**
   * @brief Compute the hydrodynamical fluxes for all interfaces at the boundary
   * between this subgrid and the given box boundary.
   *
   * @param direction TravelDirection of the neighbour.
   * @param hydro Hydro instance to use.
   * @param boundary HydroBoundary that sets the right state primitive
   * variables.
   */
  template < typename _boundary_ >
  inline void outer_ghost_flux_sweep(const int direction, const Hydro &hydro,
                                     const _boundary_ &boundary) {
    int i;
    unsigned int start_index_left, row_increment, row_length, column_increment,
        column_length;
    double dx, A;
    switch (direction) {
    case TRAVELDIRECTION_FACE_X_P:
      i = 0;
      start_index_left = (_number_of_cells[0] - 1) * _number_of_cells[3];
      row_increment = 1;
      row_length = _number_of_cells[2];
      column_increment = _number_of_cells[2];
      column_length = _number_of_cells[1];
      dx = _cell_size[0];
      A = _cell_areas[0];
      break;
    case TRAVELDIRECTION_FACE_X_N:
      i = 0;
      start_index_left = 0;
      row_increment = 1;
      row_length = _number_of_cells[2];
      column_increment = _number_of_cells[2];
      column_length = _number_of_cells[1];
      dx = -_cell_size[0];
      A = -_cell_areas[0];
      break;
    case TRAVELDIRECTION_FACE_Y_P:
      i = 1;
      start_index_left = (_number_of_cells[1] - 1) * _number_of_cells[2];
      row_increment = 1;
      row_length = _number_of_cells[2];
      column_increment = _number_of_cells[3];
      column_length = _number_of_cells[0];
      dx = _cell_size[1];
      A = _cell_areas[1];
      break;
    case TRAVELDIRECTION_FACE_Y_N:
      i = 1;
      start_index_left = 0;
      row_increment = 1;
      row_length = _number_of_cells[2];
      column_increment = _number_of_cells[3];
      column_length = _number_of_cells[0];
      dx = -_cell_size[1];
      A = -_cell_areas[1];
      break;
    case TRAVELDIRECTION_FACE_Z_P:
      i = 2;
      start_index_left = _number_of_cells[2] - 1;
      row_increment = _number_of_cells[2];
      row_length = _number_of_cells[1];
      column_increment = _number_of_cells[3];
      column_length = _number_of_cells[0];
      dx = _cell_size[2];
      A = _cell_areas[2];
      break;
    case TRAVELDIRECTION_FACE_Z_N:
      i = 2;
      start_index_left = 0;
      row_increment = _number_of_cells[2];
      row_length = _number_of_cells[1];
      column_increment = _number_of_cells[3];
      column_length = _number_of_cells[0];
      dx = -_cell_size[2];
      A = -_cell_areas[2];
      break;
    default:
      cmac_error("Unknown hydro neighbour: %i", direction);
      break;
    }

    // using the index computation below is (much) faster than setting the
    // increment correctly and summing the indices manually
    for (unsigned int ic = 0; ic < column_length; ++ic) {
      for (unsigned int ir = 0; ir < row_length; ++ir) {
        const unsigned int index_left =
            start_index_left + ic * column_increment + ir * row_increment;
#ifdef SECOND_ORDER
        const float *gradL = &_primitive_variable_gradients[15 * index_left];
#else
        const float *gradL = nullptr;
#endif
        hydro.do_ghost_flux_calculation(
            i, &_primitive_variables[5 * index_left], gradL, boundary, dx, A,
            &_delta_conserved_variables[5 * index_left]);
      }
    }
  }

  /**
   * @brief Compute the hydrodynamical gradients for all interfaces inside the
   * subgrid.
   *
   * @param hydro Hydro instance to use.
   */
  inline void inner_gradient_sweep(const Hydro &hydro) {

#ifdef SECOND_ORDER
    // we do three separate sweeps: one for every coordinate direction
    for (int ix = 0; ix < _number_of_cells[0] - 1; ++ix) {
      for (int iy = 0; iy < _number_of_cells[1]; ++iy) {
        for (int iz = 0; iz < _number_of_cells[2]; ++iz) {
          const unsigned int index000 =
              ix * _number_of_cells[3] + iy * _number_of_cells[2] + iz;
          const unsigned int index100 =
              (ix + 1) * _number_of_cells[3] + iy * _number_of_cells[2] + iz;
          // x direction
          hydro.do_gradient_calculation(
              0, &_primitive_variables[5 * index000],
              &_primitive_variables[5 * index100], _inv_cell_size[0],
              &_primitive_variable_gradients[15 * index000],
              &_primitive_variable_limiters[10 * index000],
              &_primitive_variable_gradients[15 * index100],
              &_primitive_variable_limiters[10 * index100]);
        }
      }
    }
    for (int ix = 0; ix < _number_of_cells[0]; ++ix) {
      for (int iy = 0; iy < _number_of_cells[1] - 1; ++iy) {
        for (int iz = 0; iz < _number_of_cells[2]; ++iz) {
          const unsigned int index000 =
              ix * _number_of_cells[3] + iy * _number_of_cells[2] + iz;
          const unsigned int index010 =
              ix * _number_of_cells[3] + (iy + 1) * _number_of_cells[2] + iz;
          // y direction
          hydro.do_gradient_calculation(
              1, &_primitive_variables[5 * index000],
              &_primitive_variables[5 * index010], _inv_cell_size[1],
              &_primitive_variable_gradients[15 * index000],
              &_primitive_variable_limiters[10 * index000],
              &_primitive_variable_gradients[15 * index010],
              &_primitive_variable_limiters[10 * index010]);
        }
      }
    }
    for (int ix = 0; ix < _number_of_cells[0]; ++ix) {
      for (int iy = 0; iy < _number_of_cells[1]; ++iy) {
        for (int iz = 0; iz < _number_of_cells[2] - 1; ++iz) {
          const unsigned int index000 =
              ix * _number_of_cells[3] + iy * _number_of_cells[2] + iz;
          const unsigned int index001 =
              ix * _number_of_cells[3] + iy * _number_of_cells[2] + iz + 1;
          // z direction
          hydro.do_gradient_calculation(
              2, &_primitive_variables[5 * index000],
              &_primitive_variables[5 * index001], _inv_cell_size[2],
              &_primitive_variable_gradients[15 * index000],
              &_primitive_variable_limiters[10 * index000],
              &_primitive_variable_gradients[15 * index001],
              &_primitive_variable_limiters[10 * index001]);
        }
      }
    }
#endif // SECOND_ORDER
  }

  /**
   * @brief Compute the hydrodynamical gradients for all interfaces at the
   * boundary between this subgrid and the given neighbouring subgrid.
   *
   * @param direction TravelDirection of the neighbour.
   * @param hydro Hydro instance to use.
   * @param neighbour Neighbouring DensitySubGrid.
   */
  inline void outer_gradient_sweep(const int direction, const Hydro &hydro,
                                   DensitySubGrid &neighbour) {

#ifdef SECOND_ORDER
    int i;
    unsigned int start_index_left, start_index_right, row_increment, row_length,
        column_increment, column_length;
    double dxinv;
    DensitySubGrid *left_grid, *right_grid;
    switch (direction) {
    case TRAVELDIRECTION_FACE_X_P:
      i = 0;
      left_grid = this;
      right_grid = &neighbour;
      start_index_left = (_number_of_cells[0] - 1) * _number_of_cells[3];
      start_index_right = 0;
      row_increment = 1;
      row_length = _number_of_cells[2];
      column_increment = _number_of_cells[2];
      column_length = _number_of_cells[1];
      dxinv = _inv_cell_size[0];
      break;
    case TRAVELDIRECTION_FACE_X_N:
      i = 0;
      left_grid = &neighbour;
      right_grid = this;
      start_index_left = (_number_of_cells[0] - 1) * _number_of_cells[3];
      start_index_right = 0;
      row_increment = 1;
      row_length = _number_of_cells[2];
      column_increment = _number_of_cells[2];
      column_length = _number_of_cells[1];
      dxinv = _inv_cell_size[0];
      break;
    case TRAVELDIRECTION_FACE_Y_P:
      i = 1;
      left_grid = this;
      right_grid = &neighbour;
      start_index_left = (_number_of_cells[1] - 1) * _number_of_cells[2];
      start_index_right = 0;
      row_increment = 1;
      row_length = _number_of_cells[2];
      column_increment = _number_of_cells[3];
      column_length = _number_of_cells[0];
      dxinv = _inv_cell_size[1];
      break;
    case TRAVELDIRECTION_FACE_Y_N:
      i = 1;
      left_grid = &neighbour;
      right_grid = this;
      start_index_left = (_number_of_cells[1] - 1) * _number_of_cells[2];
      start_index_right = 0;
      row_increment = 1;
      row_length = _number_of_cells[2];
      column_increment = _number_of_cells[3];
      column_length = _number_of_cells[0];
      dxinv = _inv_cell_size[1];
      break;
    case TRAVELDIRECTION_FACE_Z_P:
      i = 2;
      left_grid = this;
      right_grid = &neighbour;
      start_index_left = _number_of_cells[2] - 1;
      start_index_right = 0;
      row_increment = _number_of_cells[2];
      row_length = _number_of_cells[1];
      column_increment = _number_of_cells[3];
      column_length = _number_of_cells[0];
      dxinv = _inv_cell_size[2];
      break;
    case TRAVELDIRECTION_FACE_Z_N:
      i = 2;
      left_grid = &neighbour;
      right_grid = this;
      start_index_left = _number_of_cells[2] - 1;
      start_index_right = 0;
      row_increment = _number_of_cells[2];
      row_length = _number_of_cells[1];
      column_increment = _number_of_cells[3];
      column_length = _number_of_cells[0];
      dxinv = _inv_cell_size[2];
      break;
    default:
      cmac_error("Unknown hydro neighbour: %i", direction);
      break;
    }

    // using the index computation below is (much) faster than setting the
    // increment correctly and summing the indices manually
    for (unsigned int ic = 0; ic < column_length; ++ic) {
      for (unsigned int ir = 0; ir < row_length; ++ir) {
        const unsigned int index_left =
            start_index_left + ic * column_increment + ir * row_increment;
        const unsigned int index_right =
            start_index_right + ic * column_increment + ir * row_increment;
        hydro.do_gradient_calculation(
            i, &left_grid->_primitive_variables[5 * index_left],
            &right_grid->_primitive_variables[5 * index_right], dxinv,
            &left_grid->_primitive_variable_gradients[15 * index_left],
            &left_grid->_primitive_variable_limiters[10 * index_left],
            &right_grid->_primitive_variable_gradients[15 * index_right],
            &right_grid->_primitive_variable_limiters[10 * index_right]);
      }
    }
#endif // SECOND_ORDER
  }

  /**
   * @brief Compute the hydrodynamical gradients for all interfaces at the
   * boundary between this subgrid and the given box boundary with boundary
   * condition.
   *
   * @param direction TravelDirection of the neighbour.
   * @param hydro Hydro instance to use.
   * @param boundary HydroBoundary that sets the right state primitive
   * variables.
   */
  template < typename _boundary_ >
  inline void outer_ghost_gradient_sweep(const int direction,
                                         const Hydro &hydro,
                                         const _boundary_ &boundary) {

#ifdef SECOND_ORDER
    int i;
    unsigned int start_index_left, row_increment, row_length, column_increment,
        column_length;
    double dxinv;
    DensitySubGrid *left_grid;
    switch (direction) {
    case TRAVELDIRECTION_FACE_X_P:
      i = 0;
      left_grid = this;
      start_index_left = (_number_of_cells[0] - 1) * _number_of_cells[3];
      row_increment = 1;
      row_length = _number_of_cells[2];
      column_increment = _number_of_cells[2];
      column_length = _number_of_cells[1];
      dxinv = _inv_cell_size[0];
      break;
    case TRAVELDIRECTION_FACE_X_N:
      i = 0;
      left_grid = this;
      start_index_left = 0;
      row_increment = 1;
      row_length = _number_of_cells[2];
      column_increment = _number_of_cells[2];
      column_length = _number_of_cells[1];
      dxinv = -_inv_cell_size[0];
      break;
    case TRAVELDIRECTION_FACE_Y_P:
      i = 1;
      left_grid = this;
      start_index_left = (_number_of_cells[1] - 1) * _number_of_cells[2];
      row_increment = 1;
      row_length = _number_of_cells[2];
      column_increment = _number_of_cells[3];
      column_length = _number_of_cells[0];
      dxinv = _inv_cell_size[1];
      break;
    case TRAVELDIRECTION_FACE_Y_N:
      i = 1;
      left_grid = this;
      start_index_left = 0;
      row_increment = 1;
      row_length = _number_of_cells[2];
      column_increment = _number_of_cells[3];
      column_length = _number_of_cells[0];
      dxinv = -_inv_cell_size[1];
      break;
    case TRAVELDIRECTION_FACE_Z_P:
      i = 2;
      left_grid = this;
      start_index_left = _number_of_cells[2] - 1;
      row_increment = _number_of_cells[2];
      row_length = _number_of_cells[1];
      column_increment = _number_of_cells[3];
      column_length = _number_of_cells[0];
      dxinv = _inv_cell_size[2];
      break;
    case TRAVELDIRECTION_FACE_Z_N:
      i = 2;
      left_grid = this;
      start_index_left = 0;
      row_increment = _number_of_cells[2];
      row_length = _number_of_cells[1];
      column_increment = _number_of_cells[3];
      column_length = _number_of_cells[0];
      dxinv = -_inv_cell_size[2];
      break;
    default:
      cmac_error("Unknown hydro neighbour: %i", direction);
      break;
    }

    // using the index computation below is (much) faster than setting the
    // increment correctly and summing the indices manually
    for (unsigned int ic = 0; ic < column_length; ++ic) {
      for (unsigned int ir = 0; ir < row_length; ++ir) {
        const unsigned int index_left =
            start_index_left + ic * column_increment + ir * row_increment;
        hydro.do_ghost_gradient_calculation(
            i, &left_grid->_primitive_variables[5 * index_left], boundary,
            dxinv, &left_grid->_primitive_variable_gradients[15 * index_left],
            &left_grid->_primitive_variable_limiters[10 * index_left]);
      }
    }
#endif // SECOND_ORDER
  }

  /**
   * @brief Set the hydro task with the given index.
   *
   * @param i Index.
   * @param task Task.
   */
  inline void set_hydro_task(const int i, const size_t task) {
    _hydro_tasks[i] = task;
  }

  /**
   * @brief Get the hydro task with the given index.
   *
   * @param i Index.
   * @return Task.
   */
  inline size_t get_hydro_task(const int i) const { return _hydro_tasks[i]; }
};

#endif // DENSITYSUBGRID_HPP
