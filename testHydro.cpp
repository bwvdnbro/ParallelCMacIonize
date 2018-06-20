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
 * @file testHydro.cpp
 *
 * @brief Test for the hydro.
 *
 * @author Bert Vandenbroucke (bv7@st-andrews.ac.uk)
 */

/*! @brief Hydro only test (without DensitySubGrid calls). */
//#define DO_HYDRO_ONLY_TEST

/*! @brief Hydro test through DensitySubGrid. */
//#define DO_HYDRO_SUBGRID_TEST

/*! @brief Ascii output of the result. */
#define ASCII_OUTPUT

/*! @brief Second order hydro scheme. */
#define SECOND_ORDER

#include "Atomic.hpp"
#include "DensitySubGrid.hpp"
#include "Hydro.hpp"
#include "HydroIC.hpp"
#include "Task.hpp"
#include "Timer.hpp"

#include <fstream>
#include <vector>

// global variables, as we need them in the log macro
int MPI_rank, MPI_size;

/**
 * @brief Output the subgrid values for inspection of the physical result.
 *
 * @param gridvec Subgrids.
 * @param tot_num_subgrid Total number of original subgrids.
 */
inline void output_result(const std::vector< DensitySubGrid * > &gridvec,
                          const unsigned int tot_num_subgrid) {

  const size_t blocksize = gridvec[0]->get_output_size();

  MemoryMap file("hydro_result.dat", tot_num_subgrid * blocksize);

  // write the subgrids using multiple threads
  Atomic< unsigned int > igrid(0);
#pragma omp parallel default(shared)
  {
    while (igrid.value() < tot_num_subgrid) {
      const unsigned int this_igrid = igrid.post_increment();
      // only write local subgrids
      if (this_igrid < tot_num_subgrid) {
        const size_t offset = this_igrid * blocksize;
        gridvec[this_igrid]->output_intensities(offset, file);
      }
    }
  }
}

/**
 * @brief Make the hydro tasks for the given subgrid.
 *
 * @param next_task Index of the next task that will be created (updated).
 * @param tasks Task vector.
 * @param igrid Index of the subgrid.
 * @param gridvec Subgrids.
 */
inline void make_hydro_tasks(size_t &next_task, std::vector< Task > &tasks,
                             const unsigned int igrid,
                             std::vector< DensitySubGrid * > &gridvec) {

  DensitySubGrid &this_grid = *gridvec[igrid];
  const unsigned int ngbx = this_grid.get_neighbour(TRAVELDIRECTION_FACE_X_P);
  const unsigned int ngby = this_grid.get_neighbour(TRAVELDIRECTION_FACE_Y_P);
  const unsigned int ngbz = this_grid.get_neighbour(TRAVELDIRECTION_FACE_Z_P);

  /// gradient computation and prediction

  // internal gradient sweep
  {
    Task &task = tasks[next_task];
    task.set_type(TASKTYPE_GRADIENTSWEEP_INTERNAL);
    task.set_dependency(this_grid.get_dependency());
    task.set_subgrid(igrid);
    this_grid.set_hydro_task(0, next_task);
    ++next_task;
  }
  // external gradient sweeps
  // x
  if (this_grid.get_neighbour(TRAVELDIRECTION_FACE_X_N) == NEIGHBOUR_OUTSIDE) {
    Task &task = tasks[next_task];
    task.set_type(TASKTYPE_GRADIENTSWEEP_EXTERNAL_BOUNDARY);
    task.set_dependency(this_grid.get_dependency());
    task.set_subgrid(igrid);
    task.set_interaction_direction(TRAVELDIRECTION_FACE_X_N);
    this_grid.set_hydro_task(1, next_task);
    ++next_task;
  } else {
    this_grid.set_hydro_task(1, tasks.size());
  }
  if (ngbx == NEIGHBOUR_OUTSIDE) {
    Task &task = tasks[next_task];
    task.set_type(TASKTYPE_GRADIENTSWEEP_EXTERNAL_BOUNDARY);
    task.set_dependency(this_grid.get_dependency());
    task.set_subgrid(igrid);
    task.set_interaction_direction(TRAVELDIRECTION_FACE_X_P);
    this_grid.set_hydro_task(2, next_task);
    ++next_task;
  } else {
    Task &task = tasks[next_task];
    task.set_type(TASKTYPE_GRADIENTSWEEP_EXTERNAL_NEIGHBOUR);
    task.set_dependency(this_grid.get_dependency());
    task.set_extra_dependency(gridvec[ngbx]->get_dependency());
    task.set_subgrid(igrid);
    task.set_buffer(ngbx);
    task.set_interaction_direction(TRAVELDIRECTION_FACE_X_P);
    this_grid.set_hydro_task(2, next_task);
    ++next_task;
  }
  // y
  if (this_grid.get_neighbour(TRAVELDIRECTION_FACE_Y_N) == NEIGHBOUR_OUTSIDE) {
    Task &task = tasks[next_task];
    task.set_type(TASKTYPE_GRADIENTSWEEP_EXTERNAL_BOUNDARY);
    task.set_dependency(this_grid.get_dependency());
    task.set_subgrid(igrid);
    task.set_interaction_direction(TRAVELDIRECTION_FACE_Y_N);
    this_grid.set_hydro_task(3, next_task);
    ++next_task;
  } else {
    this_grid.set_hydro_task(3, tasks.size());
  }
  if (ngby == NEIGHBOUR_OUTSIDE) {
    Task &task = tasks[next_task];
    task.set_type(TASKTYPE_GRADIENTSWEEP_EXTERNAL_BOUNDARY);
    task.set_dependency(this_grid.get_dependency());
    task.set_subgrid(igrid);
    task.set_interaction_direction(TRAVELDIRECTION_FACE_Y_P);
    this_grid.set_hydro_task(4, next_task);
    ++next_task;
  } else {
    Task &task = tasks[next_task];
    task.set_type(TASKTYPE_GRADIENTSWEEP_EXTERNAL_NEIGHBOUR);
    task.set_dependency(this_grid.get_dependency());
    task.set_extra_dependency(gridvec[ngby]->get_dependency());
    task.set_subgrid(igrid);
    task.set_buffer(ngby);
    task.set_interaction_direction(TRAVELDIRECTION_FACE_Y_P);
    this_grid.set_hydro_task(4, next_task);
    ++next_task;
  }
  // z
  if (this_grid.get_neighbour(TRAVELDIRECTION_FACE_Z_N) == NEIGHBOUR_OUTSIDE) {
    Task &task = tasks[next_task];
    task.set_type(TASKTYPE_GRADIENTSWEEP_EXTERNAL_BOUNDARY);
    task.set_dependency(this_grid.get_dependency());
    task.set_subgrid(igrid);
    task.set_interaction_direction(TRAVELDIRECTION_FACE_Z_N);
    this_grid.set_hydro_task(5, next_task);
    ++next_task;
  } else {
    this_grid.set_hydro_task(5, tasks.size());
  }
  if (ngbz == NEIGHBOUR_OUTSIDE) {
    Task &task = tasks[next_task];
    task.set_type(TASKTYPE_GRADIENTSWEEP_EXTERNAL_BOUNDARY);
    task.set_dependency(this_grid.get_dependency());
    task.set_subgrid(igrid);
    task.set_interaction_direction(TRAVELDIRECTION_FACE_Z_P);
    this_grid.set_hydro_task(6, next_task);
    ++next_task;
  } else {
    Task &task = tasks[next_task];
    task.set_type(TASKTYPE_GRADIENTSWEEP_EXTERNAL_NEIGHBOUR);
    task.set_dependency(this_grid.get_dependency());
    task.set_extra_dependency(gridvec[ngbz]->get_dependency());
    task.set_subgrid(igrid);
    task.set_buffer(ngbz);
    task.set_interaction_direction(TRAVELDIRECTION_FACE_Z_P);
    this_grid.set_hydro_task(6, next_task);
    ++next_task;
  }
  // slope limiter
  {
    Task &task = tasks[next_task];
    task.set_type(TASKTYPE_SLOPE_LIMITER);
    task.set_dependency(this_grid.get_dependency());
    task.set_subgrid(igrid);
    this_grid.set_hydro_task(7, next_task);
    ++next_task;
  }
  // primitive variable prediction
  {
    Task &task = tasks[next_task];
    task.set_type(TASKTYPE_PREDICT_PRIMITIVES);
    task.set_dependency(this_grid.get_dependency());
    task.set_subgrid(igrid);
    this_grid.set_hydro_task(8, next_task);
    ++next_task;
  }

  /// flux exchange and primitive variable update

  // internal flux sweep
  {
    Task &task = tasks[next_task];
    task.set_type(TASKTYPE_FLUXSWEEP_INTERNAL);
    task.set_dependency(this_grid.get_dependency());
    task.set_subgrid(igrid);
    this_grid.set_hydro_task(9, next_task);
    ++next_task;
  }
  // external flux sweeps
  // x
  if (this_grid.get_neighbour(TRAVELDIRECTION_FACE_X_N) == NEIGHBOUR_OUTSIDE) {
    Task &task = tasks[next_task];
    task.set_type(TASKTYPE_FLUXSWEEP_EXTERNAL_BOUNDARY);
    task.set_dependency(this_grid.get_dependency());
    task.set_subgrid(igrid);
    task.set_interaction_direction(TRAVELDIRECTION_FACE_X_N);
    this_grid.set_hydro_task(10, next_task);
    ++next_task;
  } else {
    this_grid.set_hydro_task(10, tasks.size());
  }
  if (ngbx == NEIGHBOUR_OUTSIDE) {
    Task &task = tasks[next_task];
    task.set_type(TASKTYPE_FLUXSWEEP_EXTERNAL_BOUNDARY);
    task.set_dependency(this_grid.get_dependency());
    task.set_subgrid(igrid);
    task.set_interaction_direction(TRAVELDIRECTION_FACE_X_P);
    this_grid.set_hydro_task(11, next_task);
    ++next_task;
  } else {
    Task &task = tasks[next_task];
    task.set_type(TASKTYPE_FLUXSWEEP_EXTERNAL_NEIGHBOUR);
    task.set_dependency(this_grid.get_dependency());
    task.set_extra_dependency(gridvec[ngbx]->get_dependency());
    task.set_subgrid(igrid);
    task.set_buffer(ngbx);
    task.set_interaction_direction(TRAVELDIRECTION_FACE_X_P);
    this_grid.set_hydro_task(11, next_task);
    ++next_task;
  }
  // y
  if (this_grid.get_neighbour(TRAVELDIRECTION_FACE_Y_N) == NEIGHBOUR_OUTSIDE) {
    Task &task = tasks[next_task];
    task.set_type(TASKTYPE_FLUXSWEEP_EXTERNAL_BOUNDARY);
    task.set_dependency(this_grid.get_dependency());
    task.set_subgrid(igrid);
    task.set_interaction_direction(TRAVELDIRECTION_FACE_Y_N);
    this_grid.set_hydro_task(12, next_task);
    ++next_task;
  } else {
    this_grid.set_hydro_task(12, tasks.size());
  }
  if (ngby == NEIGHBOUR_OUTSIDE) {
    Task &task = tasks[next_task];
    task.set_type(TASKTYPE_FLUXSWEEP_EXTERNAL_BOUNDARY);
    task.set_dependency(this_grid.get_dependency());
    task.set_subgrid(igrid);
    task.set_interaction_direction(TRAVELDIRECTION_FACE_Y_P);
    this_grid.set_hydro_task(13, next_task);
    ++next_task;
  } else {
    Task &task = tasks[next_task];
    task.set_type(TASKTYPE_FLUXSWEEP_EXTERNAL_NEIGHBOUR);
    task.set_dependency(this_grid.get_dependency());
    task.set_extra_dependency(gridvec[ngby]->get_dependency());
    task.set_subgrid(igrid);
    task.set_buffer(ngby);
    task.set_interaction_direction(TRAVELDIRECTION_FACE_Y_P);
    this_grid.set_hydro_task(13, next_task);
    ++next_task;
  }
  // z
  if (this_grid.get_neighbour(TRAVELDIRECTION_FACE_Z_N) == NEIGHBOUR_OUTSIDE) {
    Task &task = tasks[next_task];
    task.set_type(TASKTYPE_FLUXSWEEP_EXTERNAL_BOUNDARY);
    task.set_dependency(this_grid.get_dependency());
    task.set_subgrid(igrid);
    task.set_interaction_direction(TRAVELDIRECTION_FACE_Z_N);
    this_grid.set_hydro_task(14, next_task);
    ++next_task;
  } else {
    this_grid.set_hydro_task(14, tasks.size());
  }
  if (ngbz == NEIGHBOUR_OUTSIDE) {
    Task &task = tasks[next_task];
    task.set_type(TASKTYPE_FLUXSWEEP_EXTERNAL_BOUNDARY);
    task.set_dependency(this_grid.get_dependency());
    task.set_subgrid(igrid);
    task.set_interaction_direction(TRAVELDIRECTION_FACE_Z_P);
    this_grid.set_hydro_task(15, next_task);
    ++next_task;
  } else {
    Task &task = tasks[next_task];
    task.set_type(TASKTYPE_FLUXSWEEP_EXTERNAL_NEIGHBOUR);
    task.set_dependency(this_grid.get_dependency());
    task.set_extra_dependency(gridvec[ngbz]->get_dependency());
    task.set_subgrid(igrid);
    task.set_buffer(ngbz);
    task.set_interaction_direction(TRAVELDIRECTION_FACE_Z_P);
    this_grid.set_hydro_task(15, next_task);
    ++next_task;
  }
  // conserved variable update
  {
    Task &task = tasks[next_task];
    task.set_type(TASKTYPE_UPDATE_CONSERVED);
    task.set_dependency(this_grid.get_dependency());
    task.set_subgrid(igrid);
    this_grid.set_hydro_task(16, next_task);
    ++next_task;
  }
  // primitive variable update
  {
    Task &task = tasks[next_task];
    task.set_type(TASKTYPE_UPDATE_PRIMITIVES);
    task.set_dependency(this_grid.get_dependency());
    task.set_subgrid(igrid);
    this_grid.set_hydro_task(17, next_task);
    ++next_task;
  }
}

/**
 * @brief Reset the hydro tasks for the given subgrid.
 *
 * @param tasks Tasks.
 * @param this_grid Subgrid.
 */
inline void reset_hydro_tasks(std::vector< Task > &tasks,
                              DensitySubGrid &this_grid) {

  // gradient sweeps
  // internal
  tasks[this_grid.get_hydro_task(0)].set_number_of_unfinished_parents(0);
  unsigned char num_ngb = 4;
  // external
  if (this_grid.get_hydro_task(1) < tasks.size()) {
    tasks[this_grid.get_hydro_task(1)].set_number_of_unfinished_parents(0);
    ++num_ngb;
  }
  tasks[this_grid.get_hydro_task(2)].set_number_of_unfinished_parents(0);
  if (this_grid.get_hydro_task(3) < tasks.size()) {
    tasks[this_grid.get_hydro_task(3)].set_number_of_unfinished_parents(0);
    ++num_ngb;
  }
  tasks[this_grid.get_hydro_task(4)].set_number_of_unfinished_parents(0);
  if (this_grid.get_hydro_task(5) < tasks.size()) {
    tasks[this_grid.get_hydro_task(5)].set_number_of_unfinished_parents(0);
    ++num_ngb;
  }
  tasks[this_grid.get_hydro_task(6)].set_number_of_unfinished_parents(0);

  // slope limiter
  tasks[this_grid.get_hydro_task(7)].set_number_of_unfinished_parents(num_ngb);
  // primitive variable prediction
  tasks[this_grid.get_hydro_task(8)].set_number_of_unfinished_parents(1);

  // flux sweeps
  // internal
  tasks[this_grid.get_hydro_task(9)].set_number_of_unfinished_parents(1);
  // external
  if (this_grid.get_hydro_task(10) < tasks.size()) {
    tasks[this_grid.get_hydro_task(10)].set_number_of_unfinished_parents(1);
  }
  if (tasks[this_grid.get_hydro_task(11)].get_type() ==
      TASKTYPE_FLUXSWEEP_EXTERNAL_BOUNDARY) {
    tasks[this_grid.get_hydro_task(11)].set_number_of_unfinished_parents(1);
  } else {
    tasks[this_grid.get_hydro_task(11)].set_number_of_unfinished_parents(2);
  }
  if (this_grid.get_hydro_task(12) < tasks.size()) {
    tasks[this_grid.get_hydro_task(12)].set_number_of_unfinished_parents(1);
  }
  if (tasks[this_grid.get_hydro_task(13)].get_type() ==
      TASKTYPE_FLUXSWEEP_EXTERNAL_BOUNDARY) {
    tasks[this_grid.get_hydro_task(13)].set_number_of_unfinished_parents(1);
  } else {
    tasks[this_grid.get_hydro_task(13)].set_number_of_unfinished_parents(2);
  }
  if (this_grid.get_hydro_task(14) < tasks.size()) {
    tasks[this_grid.get_hydro_task(14)].set_number_of_unfinished_parents(1);
  }
  if (tasks[this_grid.get_hydro_task(15)].get_type() ==
      TASKTYPE_FLUXSWEEP_EXTERNAL_BOUNDARY) {
    tasks[this_grid.get_hydro_task(15)].set_number_of_unfinished_parents(1);
  } else {
    tasks[this_grid.get_hydro_task(15)].set_number_of_unfinished_parents(2);
  }

  // conserved variable update
  tasks[this_grid.get_hydro_task(16)].set_number_of_unfinished_parents(num_ngb);
  // primitive variable update
  tasks[this_grid.get_hydro_task(17)].set_number_of_unfinished_parents(1);
}

inline void set_dependencies() {
  // START AGAIN HERE!
}

/**
 * @brief Test for the hydro.
 *
 * @param argc Number of command line arguments.
 * @param argv Command line arguments.
 * @return Exit code: 0 on success.
 */
int main(int argc, char **argv) {

#ifdef DO_HYDRO_ONLY_TEST
  /// Hydro only
  {
    Hydro hydro;

    double conserved[500];
    double delta_conserved[500];
    double primitives[500];
    double gradients[1500];
    double limiters[1000];

    for (unsigned int i = 0; i < 100; ++i) {
      if (i < 50) {
        primitives[5 * i] = 1.;
        primitives[5 * i + 4] = 1.;
      } else {
        primitives[5 * i] = 0.125;
        primitives[5 * i + 4] = 0.1;
      }
      primitives[5 * i + 1] = 0.;
      primitives[5 * i + 2] = 0.;
      primitives[5 * i + 3] = 0.;

      delta_conserved[5 * i] = 0.;
      delta_conserved[5 * i + 1] = 0.;
      delta_conserved[5 * i + 2] = 0.;
      delta_conserved[5 * i + 3] = 0.;
      delta_conserved[5 * i + 4] = 0.;
      for (unsigned int j = 0; j < 15; ++j) {
        gradients[15 * i + j] = 0.;
      }
      for (unsigned int j = 0; j < 5; ++j) {
        limiters[10 * i + 2 * j] = DBL_MAX;
        limiters[10 * i + 2 * j + 1] = -DBL_MAX;
      }
    }

    for (unsigned int i = 0; i < 100; ++i) {
      hydro.get_conserved_variables(
          primitives[5 * i], &primitives[5 * i + 1], primitives[5 * i + 4],
          0.01, conserved[5 * i], &conserved[5 * i + 1], conserved[5 * i + 4]);
    }

    const double dt = 0.001;
    double dx[3] = {0.01, 0., 0.};
    for (unsigned int istep = 0; istep < 100; ++istep) {
      for (unsigned int i = 0; i < 100; ++i) {
        unsigned int inext = (i + 1) % 100;
        hydro.do_gradient_calculation(
            0, &primitives[5 * i], &primitives[5 * inext], 100.,
            &gradients[15 * i], &limiters[10 * i], &gradients[15 * inext],
            &limiters[10 * inext]);
      }
      for (unsigned int i = 0; i < 100; ++i) {
        hydro.apply_slope_limiter(&primitives[5 * i], &gradients[15 * i],
                                  &limiters[10 * i], dx);
      }
      for (unsigned int i = 0; i < 100; ++i) {
        unsigned int inext = (i + 1) % 100;
        hydro.do_flux_calculation(
            0, &primitives[5 * i], &gradients[15 * i], &primitives[5 * inext],
            &gradients[15 * inext], 0.01, 1., &delta_conserved[5 * i],
            &delta_conserved[5 * inext]);
      }
      for (unsigned int i = 0; i < 500; ++i) {
        conserved[i] += delta_conserved[i] * dt;
      }
      for (unsigned int i = 0; i < 100; ++i) {
        hydro.get_primitive_variables(
            conserved[5 * i], &conserved[5 * i + 1], conserved[5 * i + 4], 100.,
            primitives[5 * i], &primitives[5 * i + 1], primitives[5 * i + 4]);
      }

      for (unsigned int i = 0; i < 100; ++i) {
        delta_conserved[5 * i] = 0.;
        delta_conserved[5 * i + 1] = 0.;
        delta_conserved[5 * i + 2] = 0.;
        delta_conserved[5 * i + 3] = 0.;
        delta_conserved[5 * i + 4] = 0.;
        for (unsigned int j = 0; j < 15; ++j) {
          gradients[15 * i + j] = 0.;
        }
        for (unsigned int j = 0; j < 5; ++j) {
          limiters[10 * i + 2 * j] = DBL_MAX;
          limiters[10 * i + 2 * j + 1] = -DBL_MAX;
        }
      }
    }

    std::ofstream ofile("hydro_only.txt");
    for (unsigned int i = 0; i < 100; ++i) {
      ofile << (i + 0.5) * 0.01 << "\t" << primitives[5 * i] << "\n";
    }
  }
#endif // DO_HYDRO_ONLY_TEST

#ifdef DO_HYDRO_SUBGRID_TEST
  /// DensitySubGrid hydro
  {
    const double box1[6] = {-0.5, -0.25, -0.25, 0.5, 0.5, 0.5};
    const double box2[6] = {0., -0.25, -0.25, 0.5, 0.5, 0.5};
    const int ncell[3] = {50, 3, 3};
    const unsigned int tot_ncell = ncell[0] * ncell[1] * ncell[2];
    DensitySubGrid test_grid1(box1, ncell);
    DensitySubGrid test_grid2(box2, ncell);

    for (unsigned int i = 0; i < tot_ncell; ++i) {
      double density, velocity[3], pressure;
      density = 1.;
      velocity[0] = 0.;
      velocity[1] = 0.;
      velocity[2] = 0.;
      pressure = 1.;
      test_grid1.set_primitive_variables(i, density, velocity, pressure);

      density = 0.125;
      velocity[0] = 0.;
      velocity[1] = 0.;
      velocity[2] = 0.;
      pressure = 0.1;
      test_grid2.set_primitive_variables(i, density, velocity, pressure);
    }

    const double dt = 0.001;
    const Hydro hydro;
    const InflowHydroBoundary inflow_boundary;
    const ReflectiveHydroBoundary reflective_boundary;

    test_grid1.initialize_conserved_variables(hydro);
    test_grid2.initialize_conserved_variables(hydro);
    Timer hydro_timer;
    hydro_timer.start();
    for (unsigned int istep = 0; istep < 300; ++istep) {

#ifdef SECOND_ORDER
      // gradient calculations
      test_grid1.inner_gradient_sweep(hydro);
      test_grid2.inner_gradient_sweep(hydro);
      test_grid1.outer_gradient_sweep(TRAVELDIRECTION_FACE_X_P, hydro,
                                      test_grid2);
      test_grid1.outer_ghost_gradient_sweep(TRAVELDIRECTION_FACE_X_N, hydro,
                                            inflow_boundary);
      test_grid2.outer_ghost_gradient_sweep(TRAVELDIRECTION_FACE_X_P, hydro,
                                            reflective_boundary);

      test_grid1.outer_ghost_gradient_sweep(TRAVELDIRECTION_FACE_Y_N, hydro,
                                            inflow_boundary);
      test_grid1.outer_ghost_gradient_sweep(TRAVELDIRECTION_FACE_Y_P, hydro,
                                            inflow_boundary);
      test_grid1.outer_ghost_gradient_sweep(TRAVELDIRECTION_FACE_Z_N, hydro,
                                            inflow_boundary);
      test_grid1.outer_ghost_gradient_sweep(TRAVELDIRECTION_FACE_Z_P, hydro,
                                            inflow_boundary);
      test_grid2.outer_ghost_gradient_sweep(TRAVELDIRECTION_FACE_Y_N, hydro,
                                            inflow_boundary);
      test_grid2.outer_ghost_gradient_sweep(TRAVELDIRECTION_FACE_Y_P, hydro,
                                            inflow_boundary);
      test_grid2.outer_ghost_gradient_sweep(TRAVELDIRECTION_FACE_Z_N, hydro,
                                            inflow_boundary);
      test_grid2.outer_ghost_gradient_sweep(TRAVELDIRECTION_FACE_Z_P, hydro,
                                            inflow_boundary);

      // slope limiting
      test_grid1.apply_slope_limiter(hydro);
      test_grid2.apply_slope_limiter(hydro);

      // second order time prediction
      test_grid1.predict_primitive_variables(hydro, 0.5 * dt);
      test_grid2.predict_primitive_variables(hydro, 0.5 * dt);
#endif

      // flux exchanges
      test_grid1.inner_flux_sweep(hydro);
      test_grid2.inner_flux_sweep(hydro);
      test_grid1.outer_ghost_flux_sweep(TRAVELDIRECTION_FACE_X_N, hydro,
                                        inflow_boundary);
      test_grid1.outer_flux_sweep(TRAVELDIRECTION_FACE_X_P, hydro, test_grid2);
      test_grid2.outer_ghost_flux_sweep(TRAVELDIRECTION_FACE_X_P, hydro,
                                        reflective_boundary);

      test_grid1.outer_ghost_flux_sweep(TRAVELDIRECTION_FACE_Y_N, hydro,
                                        inflow_boundary);
      test_grid1.outer_ghost_flux_sweep(TRAVELDIRECTION_FACE_Y_P, hydro,
                                        inflow_boundary);
      test_grid1.outer_ghost_flux_sweep(TRAVELDIRECTION_FACE_Z_N, hydro,
                                        inflow_boundary);
      test_grid1.outer_ghost_flux_sweep(TRAVELDIRECTION_FACE_Z_P, hydro,
                                        inflow_boundary);
      test_grid2.outer_ghost_flux_sweep(TRAVELDIRECTION_FACE_Y_N, hydro,
                                        inflow_boundary);
      test_grid2.outer_ghost_flux_sweep(TRAVELDIRECTION_FACE_Y_P, hydro,
                                        inflow_boundary);
      test_grid2.outer_ghost_flux_sweep(TRAVELDIRECTION_FACE_Z_N, hydro,
                                        inflow_boundary);
      test_grid2.outer_ghost_flux_sweep(TRAVELDIRECTION_FACE_Z_P, hydro,
                                        inflow_boundary);

      // conserved variable update
      test_grid1.update_conserved_variables(dt);
      test_grid2.update_conserved_variables(dt);

      // primitive variable update
      test_grid1.update_primitive_variables(hydro);
      test_grid2.update_primitive_variables(hydro);
    }
    hydro_timer.stop();

    cmac_status("Total time: %g s", hydro_timer.value());

#ifdef ASCII_OUTPUT
    std::ofstream ofile("hydro.txt");
    test_grid1.print_intensities(ofile);
    test_grid2.print_intensities(ofile);
#endif
    MemoryMap file("hydro.dat", 2 * test_grid1.get_output_size());
    test_grid1.output_intensities(0, file);
    test_grid2.output_intensities(test_grid1.get_output_size(), file);
  }
#endif // DO_HYDRO_SUBGRID_TEST

  /// The real stuff: task based hydro

  const SodShockHydroIC sodshock_ic;
  const double box[6] = {-0.5, -0.5, -0.5, 1., 1., 1.};
  const int ncell[3] = {64, 64, 64};
  const int num_subgrid[3] = {8, 8, 8};
  const unsigned int tot_num_subgrid =
      num_subgrid[0] * num_subgrid[1] * num_subgrid[2];

  std::vector< DensitySubGrid * > gridvec(tot_num_subgrid, nullptr);

  const double subbox_side[3] = {box[3] / num_subgrid[0],
                                 box[4] / num_subgrid[1],
                                 box[5] / num_subgrid[2]};
  const int subbox_ncell[3] = {ncell[0] / num_subgrid[0],
                               ncell[1] / num_subgrid[1],
                               ncell[2] / num_subgrid[2]};

  // set up the subgrids (in parallel)
  Atomic< unsigned int > atomic_index(0);
#pragma omp parallel default(shared)
  {
    // id of this specific thread
    const int thread_id = omp_get_thread_num();
    while (atomic_index.value() < tot_num_subgrid) {
      const unsigned int index = atomic_index.post_increment();
      if (index < tot_num_subgrid) {
        const int ix = index / (num_subgrid[1] * num_subgrid[2]);
        const int iy =
            (index - ix * num_subgrid[1] * num_subgrid[2]) / num_subgrid[2];
        const int iz =
            index - ix * num_subgrid[1] * num_subgrid[2] - iy * num_subgrid[2];
        const double subbox[6] = {box[0] + ix * subbox_side[0],
                                  box[1] + iy * subbox_side[1],
                                  box[2] + iz * subbox_side[2],
                                  subbox_side[0],
                                  subbox_side[1],
                                  subbox_side[2]};
        gridvec[index] = new DensitySubGrid(subbox, subbox_ncell);
        gridvec[index]->set_primitive_variables(sodshock_ic);
        DensitySubGrid &this_grid = *gridvec[index];
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
                this_grid.set_neighbour(ngbi, ngb_index);
              } // if ci
            }   // for niz
          }     // for niy
        }       // for nix
      }         // if local index
    }
  } // end parallel region

  const Hydro hydro;
  const InflowHydroBoundary inflow_boundary;

  std::vector< Task > tasks(18 * tot_num_subgrid);

  size_t next_task = 0;
  for (unsigned int igrid = 0; igrid < tot_num_subgrid; ++igrid) {
    gridvec[igrid]->initialize_conserved_variables(hydro);
    make_hydro_tasks(next_task, tasks, igrid, gridvec);
  }

  const double dt = 0.001;
  Timer hydro_loop_timer;
  hydro_loop_timer.start();
  for (unsigned int istep = 0; istep < 100; ++istep) {

    // gradient computation and prediction
    for (unsigned int igrid = 0; igrid < tot_num_subgrid; ++igrid) {
      DensitySubGrid &this_grid = *gridvec[igrid];
      // inner sweep
      this_grid.inner_gradient_sweep(hydro);
      // outer sweeps
      if (this_grid.get_neighbour(TRAVELDIRECTION_FACE_X_N) ==
          NEIGHBOUR_OUTSIDE) {
        this_grid.outer_ghost_gradient_sweep(TRAVELDIRECTION_FACE_X_N, hydro,
                                             inflow_boundary);
      }
      const unsigned int ngbx =
          this_grid.get_neighbour(TRAVELDIRECTION_FACE_X_P);
      if (ngbx == NEIGHBOUR_OUTSIDE) {
        this_grid.outer_ghost_gradient_sweep(TRAVELDIRECTION_FACE_X_P, hydro,
                                             inflow_boundary);
      } else {
        this_grid.outer_gradient_sweep(TRAVELDIRECTION_FACE_X_P, hydro,
                                       *gridvec[ngbx]);
      }
      if (this_grid.get_neighbour(TRAVELDIRECTION_FACE_Y_N) ==
          NEIGHBOUR_OUTSIDE) {
        this_grid.outer_ghost_gradient_sweep(TRAVELDIRECTION_FACE_Y_N, hydro,
                                             inflow_boundary);
      }
      const unsigned int ngby =
          this_grid.get_neighbour(TRAVELDIRECTION_FACE_Y_P);
      if (ngby == NEIGHBOUR_OUTSIDE) {
        this_grid.outer_ghost_gradient_sweep(TRAVELDIRECTION_FACE_Y_P, hydro,
                                             inflow_boundary);
      } else {
        this_grid.outer_gradient_sweep(TRAVELDIRECTION_FACE_Y_P, hydro,
                                       *gridvec[ngby]);
      }
      if (this_grid.get_neighbour(TRAVELDIRECTION_FACE_Z_N) ==
          NEIGHBOUR_OUTSIDE) {
        this_grid.outer_ghost_gradient_sweep(TRAVELDIRECTION_FACE_Z_N, hydro,
                                             inflow_boundary);
      }
      const unsigned int ngbz =
          this_grid.get_neighbour(TRAVELDIRECTION_FACE_Z_P);
      if (ngbz == NEIGHBOUR_OUTSIDE) {
        this_grid.outer_ghost_gradient_sweep(TRAVELDIRECTION_FACE_Z_P, hydro,
                                             inflow_boundary);
      } else {
        this_grid.outer_gradient_sweep(TRAVELDIRECTION_FACE_Z_P, hydro,
                                       *gridvec[ngbz]);
      }
    }
    for (unsigned int igrid = 0; igrid < tot_num_subgrid; ++igrid) {
      gridvec[igrid]->apply_slope_limiter(hydro);
    }
    for (unsigned int igrid = 0; igrid < tot_num_subgrid; ++igrid) {
      gridvec[igrid]->predict_primitive_variables(hydro, 0.5 * dt);
    }
    // flux exchange
    for (unsigned int igrid = 0; igrid < tot_num_subgrid; ++igrid) {
      DensitySubGrid &this_grid = *gridvec[igrid];
      // inner sweep
      this_grid.inner_flux_sweep(hydro);
      // outer sweeps
      if (this_grid.get_neighbour(TRAVELDIRECTION_FACE_X_N) ==
          NEIGHBOUR_OUTSIDE) {
        this_grid.outer_ghost_flux_sweep(TRAVELDIRECTION_FACE_X_N, hydro,
                                         inflow_boundary);
      }
      const unsigned int ngbx =
          this_grid.get_neighbour(TRAVELDIRECTION_FACE_X_P);
      if (ngbx == NEIGHBOUR_OUTSIDE) {
        this_grid.outer_ghost_flux_sweep(TRAVELDIRECTION_FACE_X_P, hydro,
                                         inflow_boundary);
      } else {
        this_grid.outer_flux_sweep(TRAVELDIRECTION_FACE_X_P, hydro,
                                   *gridvec[ngbx]);
      }
      if (this_grid.get_neighbour(TRAVELDIRECTION_FACE_Y_N) ==
          NEIGHBOUR_OUTSIDE) {
        this_grid.outer_ghost_flux_sweep(TRAVELDIRECTION_FACE_Y_N, hydro,
                                         inflow_boundary);
      }
      const unsigned int ngby =
          this_grid.get_neighbour(TRAVELDIRECTION_FACE_Y_P);
      if (ngby == NEIGHBOUR_OUTSIDE) {
        this_grid.outer_ghost_flux_sweep(TRAVELDIRECTION_FACE_Y_P, hydro,
                                         inflow_boundary);
      } else {
        this_grid.outer_flux_sweep(TRAVELDIRECTION_FACE_Y_P, hydro,
                                   *gridvec[ngby]);
      }
      if (this_grid.get_neighbour(TRAVELDIRECTION_FACE_Z_N) ==
          NEIGHBOUR_OUTSIDE) {
        this_grid.outer_ghost_flux_sweep(TRAVELDIRECTION_FACE_Z_N, hydro,
                                         inflow_boundary);
      }
      const unsigned int ngbz =
          this_grid.get_neighbour(TRAVELDIRECTION_FACE_Z_P);
      if (ngbz == NEIGHBOUR_OUTSIDE) {
        this_grid.outer_ghost_flux_sweep(TRAVELDIRECTION_FACE_Z_P, hydro,
                                         inflow_boundary);
      } else {
        this_grid.outer_flux_sweep(TRAVELDIRECTION_FACE_Z_P, hydro,
                                   *gridvec[ngbz]);
      }
    }
    for (unsigned int igrid = 0; igrid < tot_num_subgrid; ++igrid) {
      gridvec[igrid]->update_conserved_variables(dt);
    }
    for (unsigned int igrid = 0; igrid < tot_num_subgrid; ++igrid) {
      gridvec[igrid]->update_primitive_variables(hydro);
    }
  }
  hydro_loop_timer.stop();

  cmac_status("Total hydro loop time: %g s", hydro_loop_timer.value());

  output_result(gridvec, tot_num_subgrid);

  return 0;
}
