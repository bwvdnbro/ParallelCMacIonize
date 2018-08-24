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
 * @file Source.hpp
 *
 * @brief Photon source.
 *
 * @author Bert Vandenbroucke (bv7@st-andrews.ac.uk)
 */
#ifndef SOURCE_HPP
#define SOURCE_HPP

#include <vector>

/**
 * @brief Photon source.
 */
class Source {
private:
  /*! @brief Position of the source (in m). */
  const double _position[3];

  /*! @brief Ionising luminosity of the source (in s^-1). */
  const double _luminosity;

  /*! @brief Indices of the subgrids that contain the source. */
  std::vector< unsigned int > _subgrid_indices;

public:
  /**
   * @brief Empty constructor.
   */
  inline Source() : _position{0., 0., 0.}, _luminosity(0.) {}

  /**
   * @brief Constructor.
   *
   * @param position Source position (in m).
   * @param luminosity Ionising luminosity of the source (in s^-1).
   */
  inline Source(const double position[3], const double luminosity)
      : _position{position[0], position[1], position[2]},
        _luminosity(luminosity) {}

  /**
   * @brief Add an index of a subgrid that contains the source.
   *
   * @param subgrid_index Index of a subgrid that contains the source.
   */
  inline void add_subgrid_index(const unsigned int subgrid_index) {
    _subgrid_indices.push_back(subgrid_index);
  }

  /**
   * @brief Get the position of the source.
   *
   * @param position Array to store the position of the source in (in m).
   */
  inline void get_position(double position[3]) const {
    position[0] = _position[0];
    position[1] = _position[1];
    position[2] = _position[2];
  }

  /**
   * @brief Get the ionising luminosity of the source.
   *
   * @return Ionising luminosity of the source (in s^-1).
   */
  inline double get_luminosity() const { return _luminosity; }

  /**
   * @brief Get a (random) index of a subgrid that contains the source.
   *
   * @param random_generator RandomGenerator.
   * @return Random index of a subgrid that contains the source.
   */
  inline unsigned int
  get_subgrid_index(RandomGenerator &random_generator) const {
    return _subgrid_indices[static_cast< unsigned int >(
        random_generator.get_uniform_random_double() *
        _subgrid_indices.size())];
  }

  /**
   * @brief Get the original subgrid that contains the source.
   *
   * @return Index of the original subgrid that contains the source.
   */
  inline unsigned int get_original_subgrid_index() const {
    return _subgrid_indices[0];
  }
};

#endif // SOURCE_HPP
