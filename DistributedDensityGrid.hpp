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
 * @file DistributedDensityGrid.hpp
 *
 * @brief Density grid that is distributed into (many) smaller parts.
 *
 * Each MPI rank stores a complete list of subgrids, but only the subgrids that
 * are present on the local rank are actually allocated. All neighbour relations
 * between subgrids use the indices in this global subgrid list.
 *
 * @author Bert Vandenbroucke (bv7@st-andrews.ac.uk)
 */
#ifndef DISTRIBUTEDDENSITYGRID_HPP
#define DISTRIBUTEDDENSITYGRID_HPP

#include "Atomic.hpp"
#include "DensitySubGrid.hpp"

#include <vector>

/**
 * @brief Density grid that is distributed into (many) smaller parts.
 */
class DistributedDensityGrid {
private:
  /*! @brief Actual subgrids. Only contains local subgrids. */
  std::vector< DensitySubGrid * > _gridvec;

public:
  /**
   * @brief Create the distributed grid.
   *
   * This sets up a grid of subgrids with correct internal neighbour relations.
   *
   * @param box Box containing the entire grid (in m).
   * @param ncell Number of cells in each coordinate dimension.
   * @param num_subgrid Number of subgrids in each coordinate dimension.
   * @param levels Level of each subgrid, i.e. how many copies each subgrid has.
   * @param MPI_mask Filter that marks out the subgrids present on the local
   * MPI rank.
   */
  inline void create(const double box[6], const int ncell[3],
                     const int num_subgrid[3],
                     std::vector< unsigned char > &levels,
                     std::vector< bool > &MPI_mask) {

    // get the total number of original)subgrids and the dimensions and
    // number of cells for each subgrid

    const unsigned int original_num_subgrid =
        num_subgrid[0] * num_subgrid[1] * num_subgrid[2];

    const double subbox_side[3] = {box[3] / num_subgrid[0],
                                   box[4] / num_subgrid[1],
                                   box[5] / num_subgrid[2]};
    const int subbox_ncell[3] = {ncell[0] / num_subgrid[0],
                                 ncell[1] / num_subgrid[1],
                                 ncell[2] / num_subgrid[2]};

    // now figure out how many additional subgrids have to be created because
    // of copies, and figure out the correct neighbour relations
    // to this end, we store the offsets of the first copy of each original (if
    // the original has copies) and the index of the original for each copy
    std::vector< unsigned int > copies(original_num_subgrid, 0xffffffff);
    std::vector< unsigned int > originals;
    unsigned int tot_num_subgrid = original_num_subgrid;
    for (unsigned int i = 0; i < original_num_subgrid; ++i) {
      const unsigned char level = levels[i];
      const unsigned int number_of_copies = 1 << level;
      if (number_of_copies > 1) {
        copies[i] = tot_num_subgrid;
        tot_num_subgrid += number_of_copies;
        for (unsigned int j = 1; j < number_of_copies; ++j) {
          originals.push_back(i);
        }
      }
    }

    // now set up the empty subgrid vector
    _gridvec.resize(tot_num_subgrid, nullptr);

    // set up the subgrids (in parallel)
    Atomic< unsigned int > atomic_index(0);
#pragma omp parallel default(shared)
    {
      // id of this specific thread
      const int thread_id = omp_get_thread_num();
      // double conditional loop to get a safe, thread unique index
      while (atomic_index.value() < tot_num_subgrid) {
        const unsigned int index = atomic_index.post_increment();
        if (index < tot_num_subgrid && MPI_mask[index]) {
          // the index tells us which subgrid to set up, but not whether this
          // is an original or a copy. We figure that out here
          unsigned int original_index = index;
          if (index >= original_num_subgrid) {
            original_index = originals[index - original_num_subgrid];
          }

          // the index of the original subgrid encodes the position
          const int ix = original_index / (num_subgrid[1] * num_subgrid[2]);
          const int iy =
              (original_index - ix * num_subgrid[1] * num_subgrid[2]) /
              num_subgrid[2];
          const int iz = original_index - ix * num_subgrid[1] * num_subgrid[2] -
                         iy * num_subgrid[2];
          const double subbox[6] = {box[0] + ix * subbox_side[0],
                                    box[1] + iy * subbox_side[1],
                                    box[2] + iz * subbox_side[2],
                                    subbox_side[0],
                                    subbox_side[1],
                                    subbox_side[2]};
          _gridvec[index] = new DensitySubGrid(subbox, subbox_ncell);

          // now set up the grid variables
          DensitySubGrid &this_grid = *_gridvec[index];
          this_grid.set_owning_thread(thread_id);
          // set up neighbouring information. We first make sure all
          // neighbours are initialized to NEIGHBOUR_OUTSIDE, indicating no
          // neighbour
          for (int i = 0; i < TRAVELDIRECTION_NUMBER; ++i) {
            this_grid.set_neighbour(i, NEIGHBOUR_OUTSIDE);
            this_grid.set_active_buffer(i, NEIGHBOUR_OUTSIDE);
          }

          // now set up the correct neighbour relations for the neighbours
          // that exist
          for (int nix = -1; nix < 2; ++nix) {
            for (int niy = -1; niy < 2; ++niy) {
              for (int niz = -1; niz < 2; ++niz) {
                // get neighbour corrected indices
                const int cix = ix + nix;
                const int ciy = iy + niy;
                const int ciz = iz + niz;
                // if the indices above point to a real subgrid: set up the
                // neighbour relations
                if (cix >= 0 && cix < num_subgrid[0] && ciy >= 0 &&
                    ciy < num_subgrid[1] && ciz >= 0 && ciz < num_subgrid[2]) {
                  // we use get_output_direction() to get the correct index
                  // for the neighbour
                  // the three_index components will either be
                  //  - -ncell --> negative --> lower limit
                  //  - 0 --> in range --> inside
                  //  - ncell --> upper limit
                  const int three_index[3] = {nix * subbox_ncell[0],
                                              niy * subbox_ncell[1],
                                              niz * subbox_ncell[2]};
                  const int ngbi = this_grid.get_output_direction(three_index);
                  // now get the actual ngb index
                  const unsigned int ngb_index =
                      cix * num_subgrid[1] * num_subgrid[2] +
                      ciy * num_subgrid[2] + ciz;

                  // figure out which subgrid to use as neighbour
                  if (index == original_index) {
                    // original subgrids always have an original as neighbour
                    this_grid.set_neighbour(ngbi, ngb_index);
                  } else {
                    // this is a copy, which neighbour to use depends on the
                    // level of the neighbour
                    const unsigned char level = levels[original_index];
                    const unsigned char ngb_level = levels[ngb_index];
                    if (ngb_level == level) {
                      // same, easy: just make copies mutual neighbours
                      this_grid.set_neighbour(ngbi, copies[ngb_index] + index -
                                                        copies[original_index]);
                    } else {
                      // not the same: there are 2 options
                      if (level > ngb_level) {
                        // we have less neighbour copies, so some of our copies
                        // need to share the same neighbour
                        // some of our copies might also need to share the
                        // original neighbour
                        const unsigned int number_of_ngb_copies =
                            1 << (level - ngb_level);
                        const unsigned int copy_index =
                            (index - copies[original_index]) /
                            number_of_ngb_copies;
                        const unsigned int ngb_copy =
                            (copy_index > 0)
                                ? copies[ngb_index] + copy_index - 1
                                : ngb_index;
                        this_grid.set_neighbour(ngbi, ngb_copy);
                      } else {
                        // we have more neighbour copies: pick a subset
                        const unsigned int number_of_own_copies =
                            1 << (ngb_level - level);
                        this_grid.set_neighbour(
                            ngbi, copies[ngb_index] +
                                      (index - copies[original_index]) *
                                          number_of_own_copies);
                      }
                    }
                  }

                } // if ci
              }   // for niz
            }     // for niy
          }       // for nix
        }         // if local index
      }
    } // end parallel region
  }
};

#endif // DISTRIBUTEDDENSITYGRID_HPP
