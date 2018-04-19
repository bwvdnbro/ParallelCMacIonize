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
#include "Lock.hpp"
#include "Photon.hpp"
#include "TravelDirections.hpp"

// standard library includes
#include <algorithm>
#include <cfloat>
#include <cmath>
#include <iostream>
#include <mpi.h>
#include <omp.h>
#include <ostream>

/*! @brief Special neighbour index marking a neighbour that does not exist. */
#define NEIGHBOUR_OUTSIDE 0xffffffff

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

/*! @brief Size of the DensitySubGrid variables whose size is known at compile
 *  time. */
#define DENSITYSUBGRID_FIXED_MPI_SIZE                                          \
  (sizeof(unsigned long) + 9 * sizeof(double) + 4 * sizeof(int) +              \
   (TRAVELDIRECTION_NUMBER + 1) * sizeof(unsigned int))

/*! @brief Number of variables stored in each cell of the DensitySubGrid
 *  (excluding potential lock variables). */
#define DENSITYSUBGRID_NUMBER_OF_CELL_VARIABLES 3

/**
 * @brief Check if the given DensitySubGrids are the same.
 *
 * @param a First DensitySubGrid.
 * @param b Second DensitySubGrid.
 */
#define densitysubgrid_check_equal(a, b)                                       \
  myassert(a._computational_cost == b._computational_cost,                     \
           "Costs not the same!");                                             \
  myassert(a._anchor[0] == b._anchor[0], "Anchor not the same!");              \
  myassert(a._anchor[1] == b._anchor[1], "Anchor not the same!");              \
  myassert(a._anchor[2] == b._anchor[2], "Anchor not the same!");              \
  myassert(a._cell_size[0] == b._cell_size[0], "Cell size not the same!");     \
  myassert(a._cell_size[1] == b._cell_size[1], "Cell size not the same!");     \
  myassert(a._cell_size[2] == b._cell_size[2], "Cell size not the same!");     \
  myassert(a._inv_cell_size[0] == b._inv_cell_size[0],                         \
           "Inverse cell size not the same!");                                 \
  myassert(a._inv_cell_size[1] == b._inv_cell_size[1],                         \
           "Inverse cell size not the same!");                                 \
  myassert(a._inv_cell_size[2] == b._inv_cell_size[2],                         \
           "Inverse cell size not the same!");                                 \
  myassert(a._number_of_cells[0] == b._number_of_cells[0],                     \
           "Number of cells not the same!");                                   \
  myassert(a._number_of_cells[1] == b._number_of_cells[1],                     \
           "Number of cells not the same!");                                   \
  myassert(a._number_of_cells[2] == b._number_of_cells[2],                     \
           "Number of cells not the same!");                                   \
  myassert(a._number_of_cells[3] == b._number_of_cells[3],                     \
           "Number of cells not the same!");                                   \
  for (unsigned int i = 0; i < TRAVELDIRECTION_NUMBER; ++i) {                  \
    myassert(a._ngbs[i] == b._ngbs[i], "Neighbours not the same!");            \
  }                                                                            \
  myassert(a._subgrid_index == b._subgrid_index,                               \
           "Subgrid indices not the same!");                                   \
  const int tot_num_cells = a._number_of_cells[0] * a._number_of_cells[3];     \
  for (int i = 0; i < tot_num_cells; ++i) {                                    \
    myassert(a._number_density[i] == b._number_density[i],                     \
             "Number density not the same!");                                  \
    myassert(a._neutral_fraction[i] == b._neutral_fraction[i],                 \
             "Neutral fraction not the same!");                                \
    myassert(a._intensity_integral[i] == b._intensity_integral[i],             \
             "Intensity integral not the same!");                              \
  }

/**
 * @brief Small fraction of a density grid that acts as an individual density
 * grid.
 */
class DensitySubGrid {
public:
  /*! @brief Indices of the neighbouring subgrids. */
  unsigned int _ngbs[TRAVELDIRECTION_NUMBER];

  /*! @brief Indices of the active buffers. */
  unsigned int _active_buffers[TRAVELDIRECTION_NUMBER];

  /*! @brief Index of the SubGrid in the subgrid list. */
  unsigned int _subgrid_index;

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

  /*! @brief Number density for each cell (in m^-3). */
  double *_number_density;

  /*! @brief Neutral fraction of hydrogen for each cell. */
  double *_neutral_fraction;

  /*! @brief Ionizing intensity estimate for each cell (in m^3). */
  double *_intensity_integral;

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
        _owning_thread(-1), _largest_buffer_index(TRAVELDIRECTION_NUMBER),
        _largest_buffer_size(0) {

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
  }

  /**
   * @brief Copy constructor.
   *
   * @param original DensitySubGrid to copy.
   */
  inline DensitySubGrid(const DensitySubGrid &original)
      : _computational_cost(0),
        _anchor{original._anchor[0], original._anchor[1], original._anchor[2]},
        _cell_size{original._cell_size[0], original._cell_size[1],
                   original._cell_size[2]},
        _inv_cell_size{original._inv_cell_size[0], original._inv_cell_size[1],
                       original._inv_cell_size[2]},
        _number_of_cells{
            original._number_of_cells[0], original._number_of_cells[1],
            original._number_of_cells[2], original._number_of_cells[3]},
        _owning_thread(-1), _largest_buffer_index(TRAVELDIRECTION_NUMBER),
        _largest_buffer_size(0) {

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
  }

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
    return DENSITYSUBGRID_FIXED_MPI_SIZE +
           DENSITYSUBGRID_NUMBER_OF_CELL_VARIABLES * _number_of_cells[0] *
               _number_of_cells[3] * sizeof(double);
  }

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
    MPI_Pack(&_subgrid_index, 1, MPI_UNSIGNED, buffer, buffer_size,
             &buffer_position, MPI_COMM_WORLD);

    const int tot_num_cells = _number_of_cells[0] * _number_of_cells[3];
    MPI_Pack(_number_density, tot_num_cells, MPI_DOUBLE, buffer, buffer_size,
             &buffer_position, MPI_COMM_WORLD);
    MPI_Pack(_neutral_fraction, tot_num_cells, MPI_DOUBLE, buffer, buffer_size,
             &buffer_position, MPI_COMM_WORLD);
    MPI_Pack(_intensity_integral, tot_num_cells, MPI_DOUBLE, buffer,
             buffer_size, &buffer_position, MPI_COMM_WORLD);
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
    MPI_Unpack(buffer, buffer_size, &buffer_position, &_subgrid_index, 1,
               MPI_UNSIGNED, MPI_COMM_WORLD);

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
    }
    MPI_Unpack(buffer, buffer_size, &buffer_position, _number_density,
               tot_num_cells, MPI_DOUBLE, MPI_COMM_WORLD);
    MPI_Unpack(buffer, buffer_size, &buffer_position, _neutral_fraction,
               tot_num_cells, MPI_DOUBLE, MPI_COMM_WORLD);
    MPI_Unpack(buffer, buffer_size, &buffer_position, _intensity_integral,
               tot_num_cells, MPI_DOUBLE, MPI_COMM_WORLD);
  }

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
    const int tot_ncell = _number_of_cells[3] * _number_of_cells[0];
    for (int i = 0; i < tot_ncell; ++i) {
      _intensity_integral[i] += copy._intensity_integral[i];
    }
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
    const double direction[3] = {photon._direction[0], photon._direction[1],
                                 photon._direction[2]};

    myassert(TravelDirections::is_compatible_input_direction(direction,
                                                             input_direction),
             "direction: " << direction[0] << " " << direction[1] << " "
                           << direction[2]
                           << ", input_direction: " << input_direction);

    const double inverse_direction[3] = {photon._inverse_direction[0],
                                         photon._inverse_direction[1],
                                         photon._inverse_direction[2]};
    // NOTE: position is relative w.r.t. _anchor!!!
    double position[3] = {photon._position[0] - _anchor[0],
                          photon._position[1] - _anchor[1],
                          photon._position[2] - _anchor[2]};
    double tau_done = photon._current_optical_depth;
    const double tau_target = photon._target_optical_depth;

    myassert(tau_done < tau_target, "tau_done: " << tau_done
                                                 << ", target: " << tau_target);

    const double cross_section = photon._photoionization_cross_section;
    const double photon_weight = photon._weight;
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
    photon._current_optical_depth = tau_done;
    photon._position[0] = position[0] + _anchor[0];
    photon._position[1] = position[1] + _anchor[1];
    photon._position[2] = position[2] + _anchor[2];
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
   * @brief Get the intensity integral for the cell with the given indices.
   *
   * @param ix X index of the cell.
   * @param iy Y index of the cell.
   * @param iz Z index of the cell.
   * @return Luminosity integral for the cell with the given indices.
   */
  inline const double get_intensity_integral(const int ix, const int iy,
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
                 << "\n";
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
        }
      }
    }
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
  inline const unsigned long get_computational_cost() const {
    return _computational_cost;
  }

  /**
   * @brief Get the dependency lock for this subgrid.
   *
   * @return Pointer to the dependency lock for this subgrid.
   */
  inline Lock *get_dependency() { return &_dependency; }
};

#endif // DENSITYSUBGRID_HPP
