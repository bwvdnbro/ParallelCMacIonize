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

/*! @brief Output log level. The higher the value, the more stuff is printed to
 *  the stderr. Comment to disable logging altogether. */
#define LOG_OUTPUT 1

/*! @brief Hydro only test (without DensitySubGrid calls). */
//#define DO_HYDRO_ONLY_TEST

/*! @brief Hydro test through DensitySubGrid. */
//#define DO_HYDRO_SUBGRID_TEST

/*! @brief Ascii output of the result. */
#define ASCII_OUTPUT

/*! @brief Second order hydro scheme. */
#define SECOND_ORDER

/*! @brief Uncomment this to enable run time assertions. */
#define DO_ASSERTS

/*! @brief Make task plots. */
#define TASK_PLOT

#include "Assert.hpp"
#include "Atomic.hpp"
#include "CommandLineParser.hpp"
#include "DensitySubGrid.hpp"
#include "Hydro.hpp"
#include "HydroIC.hpp"
#include "Log.hpp"
#include "Queue.hpp"
#include "Task.hpp"
#include "ThreadSafeVector.hpp"
#include "Timer.hpp"

#include <fstream>
#include <sstream>
#include <vector>

// global variables, as we need them in the log macro
int MPI_rank, MPI_size;

/**
 * @brief Write a file with the start and end times of all tasks.
 *
 * @param iloop Iteration number (added to file name).
 * @param tasks Tasks to print.
 * @param iteration_start Start CPU cycle count of the iteration on this
 * process.
 * @param iteration_end End CPU cycle count of the iteration on this process.
 */
inline void output_tasks(const unsigned int iloop,
                         ThreadSafeVector< Task > &tasks,
                         const unsigned long iteration_start,
                         const unsigned long iteration_end) {

#ifdef TASK_PLOT
  {
    // compose the file name
    std::stringstream filename;
    filename << "hydro_tasks_";
    filename.fill('0');
    filename.width(2);
    filename << iloop;
    filename << ".txt";

    // now output
    // each process outputs its own tasks in turn, process 0 is responsible for
    // creating the file and the other processes append to it
    for (int irank = 0; irank < MPI_size; ++irank) {
      // only the active process writes
      if (irank == MPI_rank) {
        // the file mode depends on the rank
        std::ios_base::openmode mode;
        if (irank == 0) {
          // rank 0 creates (or overwrites) the file
          mode = std::ofstream::trunc;
        } else {
          // all other ranks append to it
          mode = std::ofstream::app;
        }
        // now open the file
        std::ofstream ofile(filename.str(), mode);

        // rank 0 writes the header
        if (irank == 0) {
          ofile << "# rank\tthread\tstart\tstop\ttype\n";
        }

        // write the start and end CPU cycle count
        // this is a dummy task executed by thread 0 (so that the min or max
        // thread count is not affected), but with non-existing type -1
        ofile << MPI_rank << "\t0\t" << iteration_start << "\t" << iteration_end
              << "\t-1\n";

        // write the task info
        const size_t tsize = tasks.size();
        for (size_t i = 0; i < tsize; ++i) {
          const Task &task = tasks[i];
          myassert(task.done(), "Task was never executed!");
          int type, thread_id;
          unsigned long start, end;
          task.get_timing_information(type, thread_id, start, end);
          ofile << MPI_rank << "\t" << thread_id << "\t" << start << "\t" << end
                << "\t" << type << "\n";
        }
      }
      // only one process at a time is allowed to write
      MPI_Barrier(MPI_COMM_WORLD);
    }
  }
#endif
}

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
 * @param tasks Task vector.
 * @param igrid Index of the subgrid.
 * @param gridvec Subgrids.
 */
inline void make_hydro_tasks(ThreadSafeVector< Task > &tasks,
                             const unsigned int igrid,
                             std::vector< DensitySubGrid * > &gridvec) {

  DensitySubGrid &this_grid = *gridvec[igrid];
  const unsigned int ngbx = this_grid.get_neighbour(TRAVELDIRECTION_FACE_X_P);
  const unsigned int ngby = this_grid.get_neighbour(TRAVELDIRECTION_FACE_Y_P);
  const unsigned int ngbz = this_grid.get_neighbour(TRAVELDIRECTION_FACE_Z_P);

  /// gradient computation and prediction

  // internal gradient sweep
  {
    const size_t next_task = tasks.get_free_element();
    Task &task = tasks[next_task];
    task.set_type(TASKTYPE_GRADIENTSWEEP_INTERNAL);
    task.set_dependency(this_grid.get_dependency());
    task.set_subgrid(igrid);
    this_grid.set_hydro_task(0, next_task);
  }
  // external gradient sweeps
  // x
  // positive: always apply
  if (ngbx == NEIGHBOUR_OUTSIDE) {
    const size_t next_task = tasks.get_free_element();
    Task &task = tasks[next_task];
    task.set_type(TASKTYPE_GRADIENTSWEEP_EXTERNAL_BOUNDARY);
    task.set_dependency(this_grid.get_dependency());
    task.set_subgrid(igrid);
    task.set_interaction_direction(TRAVELDIRECTION_FACE_X_P);
    this_grid.set_hydro_task(1, next_task);
  } else {
    const size_t next_task = tasks.get_free_element();
    Task &task = tasks[next_task];
    task.set_type(TASKTYPE_GRADIENTSWEEP_EXTERNAL_NEIGHBOUR);
    // avoid dining philosophers by sorting the dependencies on subgrid index
    if (igrid < ngbx) {
      task.set_dependency(this_grid.get_dependency());
      task.set_extra_dependency(gridvec[ngbx]->get_dependency());
    } else {
      task.set_dependency(gridvec[ngbx]->get_dependency());
      task.set_extra_dependency(this_grid.get_dependency());
    }
    task.set_subgrid(igrid);
    task.set_buffer(ngbx);
    task.set_interaction_direction(TRAVELDIRECTION_FACE_X_P);
    this_grid.set_hydro_task(1, next_task);
  }
  // negative: only apply if we have a non-periodic boundary
  if (this_grid.get_neighbour(TRAVELDIRECTION_FACE_X_N) == NEIGHBOUR_OUTSIDE) {
    const size_t next_task = tasks.get_free_element();
    Task &task = tasks[next_task];
    task.set_type(TASKTYPE_GRADIENTSWEEP_EXTERNAL_BOUNDARY);
    task.set_dependency(this_grid.get_dependency());
    task.set_subgrid(igrid);
    task.set_interaction_direction(TRAVELDIRECTION_FACE_X_N);
    this_grid.set_hydro_task(2, next_task);
  } else {
    this_grid.set_hydro_task(2, NO_TASK);
  }
  // y
  if (ngby == NEIGHBOUR_OUTSIDE) {
    const size_t next_task = tasks.get_free_element();
    Task &task = tasks[next_task];
    task.set_type(TASKTYPE_GRADIENTSWEEP_EXTERNAL_BOUNDARY);
    task.set_dependency(this_grid.get_dependency());
    task.set_subgrid(igrid);
    task.set_interaction_direction(TRAVELDIRECTION_FACE_Y_P);
    this_grid.set_hydro_task(3, next_task);
  } else {
    const size_t next_task = tasks.get_free_element();
    Task &task = tasks[next_task];
    task.set_type(TASKTYPE_GRADIENTSWEEP_EXTERNAL_NEIGHBOUR);
    if (igrid < ngby) {
      task.set_dependency(this_grid.get_dependency());
      task.set_extra_dependency(gridvec[ngby]->get_dependency());
    } else {
      task.set_dependency(gridvec[ngby]->get_dependency());
      task.set_extra_dependency(this_grid.get_dependency());
    }
    task.set_subgrid(igrid);
    task.set_buffer(ngby);
    task.set_interaction_direction(TRAVELDIRECTION_FACE_Y_P);
    this_grid.set_hydro_task(3, next_task);
  }
  if (this_grid.get_neighbour(TRAVELDIRECTION_FACE_Y_N) == NEIGHBOUR_OUTSIDE) {
    const size_t next_task = tasks.get_free_element();
    Task &task = tasks[next_task];
    task.set_type(TASKTYPE_GRADIENTSWEEP_EXTERNAL_BOUNDARY);
    task.set_dependency(this_grid.get_dependency());
    task.set_subgrid(igrid);
    task.set_interaction_direction(TRAVELDIRECTION_FACE_Y_N);
    this_grid.set_hydro_task(4, next_task);
  } else {
    this_grid.set_hydro_task(4, NO_TASK);
  }
  // z
  if (ngbz == NEIGHBOUR_OUTSIDE) {
    const size_t next_task = tasks.get_free_element();
    Task &task = tasks[next_task];
    task.set_type(TASKTYPE_GRADIENTSWEEP_EXTERNAL_BOUNDARY);
    task.set_dependency(this_grid.get_dependency());
    task.set_subgrid(igrid);
    task.set_interaction_direction(TRAVELDIRECTION_FACE_Z_P);
    this_grid.set_hydro_task(5, next_task);
  } else {
    const size_t next_task = tasks.get_free_element();
    Task &task = tasks[next_task];
    task.set_type(TASKTYPE_GRADIENTSWEEP_EXTERNAL_NEIGHBOUR);
    if (igrid < ngbz) {
      task.set_dependency(this_grid.get_dependency());
      task.set_extra_dependency(gridvec[ngbz]->get_dependency());
    } else {
      task.set_dependency(gridvec[ngbz]->get_dependency());
      task.set_extra_dependency(this_grid.get_dependency());
    }
    task.set_subgrid(igrid);
    task.set_buffer(ngbz);
    task.set_interaction_direction(TRAVELDIRECTION_FACE_Z_P);
    this_grid.set_hydro_task(5, next_task);
  }
  if (this_grid.get_neighbour(TRAVELDIRECTION_FACE_Z_N) == NEIGHBOUR_OUTSIDE) {
    const size_t next_task = tasks.get_free_element();
    Task &task = tasks[next_task];
    task.set_type(TASKTYPE_GRADIENTSWEEP_EXTERNAL_BOUNDARY);
    task.set_dependency(this_grid.get_dependency());
    task.set_subgrid(igrid);
    task.set_interaction_direction(TRAVELDIRECTION_FACE_Z_N);
    this_grid.set_hydro_task(6, next_task);
  } else {
    this_grid.set_hydro_task(6, NO_TASK);
  }
  // slope limiter
  {
    const size_t next_task = tasks.get_free_element();
    Task &task = tasks[next_task];
    task.set_type(TASKTYPE_SLOPE_LIMITER);
    task.set_dependency(this_grid.get_dependency());
    task.set_subgrid(igrid);
    this_grid.set_hydro_task(7, next_task);
  }
  // primitive variable prediction
  {
    const size_t next_task = tasks.get_free_element();
    Task &task = tasks[next_task];
    task.set_type(TASKTYPE_PREDICT_PRIMITIVES);
    task.set_dependency(this_grid.get_dependency());
    task.set_subgrid(igrid);
    this_grid.set_hydro_task(8, next_task);
  }

  /// flux exchange and primitive variable update

  // internal flux sweep
  {
    const size_t next_task = tasks.get_free_element();
    Task &task = tasks[next_task];
    task.set_type(TASKTYPE_FLUXSWEEP_INTERNAL);
    task.set_dependency(this_grid.get_dependency());
    task.set_subgrid(igrid);
    this_grid.set_hydro_task(9, next_task);
  }
  // external flux sweeps
  // x
  // positive: always do flux exchange
  if (ngbx == NEIGHBOUR_OUTSIDE) {
    const size_t next_task = tasks.get_free_element();
    Task &task = tasks[next_task];
    task.set_type(TASKTYPE_FLUXSWEEP_EXTERNAL_BOUNDARY);
    task.set_dependency(this_grid.get_dependency());
    task.set_subgrid(igrid);
    task.set_interaction_direction(TRAVELDIRECTION_FACE_X_P);
    this_grid.set_hydro_task(10, next_task);
  } else {
    const size_t next_task = tasks.get_free_element();
    Task &task = tasks[next_task];
    task.set_type(TASKTYPE_FLUXSWEEP_EXTERNAL_NEIGHBOUR);
    if (igrid < ngbx) {
      task.set_dependency(this_grid.get_dependency());
      task.set_extra_dependency(gridvec[ngbx]->get_dependency());
    } else {
      task.set_dependency(gridvec[ngbx]->get_dependency());
      task.set_extra_dependency(this_grid.get_dependency());
    }
    task.set_subgrid(igrid);
    task.set_buffer(ngbx);
    task.set_interaction_direction(TRAVELDIRECTION_FACE_X_P);
    this_grid.set_hydro_task(10, next_task);
  }
  // negative: only do flux exchange when we need to apply a non-periodic
  // boundary condition
  if (this_grid.get_neighbour(TRAVELDIRECTION_FACE_X_N) == NEIGHBOUR_OUTSIDE) {
    const size_t next_task = tasks.get_free_element();
    Task &task = tasks[next_task];
    task.set_type(TASKTYPE_FLUXSWEEP_EXTERNAL_BOUNDARY);
    task.set_dependency(this_grid.get_dependency());
    task.set_subgrid(igrid);
    task.set_interaction_direction(TRAVELDIRECTION_FACE_X_N);
    this_grid.set_hydro_task(11, next_task);
  } else {
    this_grid.set_hydro_task(11, NO_TASK);
  }
  // y
  if (ngby == NEIGHBOUR_OUTSIDE) {
    const size_t next_task = tasks.get_free_element();
    Task &task = tasks[next_task];
    task.set_type(TASKTYPE_FLUXSWEEP_EXTERNAL_BOUNDARY);
    task.set_dependency(this_grid.get_dependency());
    task.set_subgrid(igrid);
    task.set_interaction_direction(TRAVELDIRECTION_FACE_Y_P);
    this_grid.set_hydro_task(12, next_task);
  } else {
    const size_t next_task = tasks.get_free_element();
    Task &task = tasks[next_task];
    task.set_type(TASKTYPE_FLUXSWEEP_EXTERNAL_NEIGHBOUR);
    if (igrid < ngby) {
      task.set_dependency(this_grid.get_dependency());
      task.set_extra_dependency(gridvec[ngby]->get_dependency());
    } else {
      task.set_dependency(gridvec[ngby]->get_dependency());
      task.set_extra_dependency(this_grid.get_dependency());
    }
    task.set_subgrid(igrid);
    task.set_buffer(ngby);
    task.set_interaction_direction(TRAVELDIRECTION_FACE_Y_P);
    this_grid.set_hydro_task(12, next_task);
  }
  if (this_grid.get_neighbour(TRAVELDIRECTION_FACE_Y_N) == NEIGHBOUR_OUTSIDE) {
    const size_t next_task = tasks.get_free_element();
    Task &task = tasks[next_task];
    task.set_type(TASKTYPE_FLUXSWEEP_EXTERNAL_BOUNDARY);
    task.set_dependency(this_grid.get_dependency());
    task.set_subgrid(igrid);
    task.set_interaction_direction(TRAVELDIRECTION_FACE_Y_N);
    this_grid.set_hydro_task(13, next_task);
  } else {
    this_grid.set_hydro_task(13, NO_TASK);
  }
  // z
  if (ngbz == NEIGHBOUR_OUTSIDE) {
    const size_t next_task = tasks.get_free_element();
    Task &task = tasks[next_task];
    task.set_type(TASKTYPE_FLUXSWEEP_EXTERNAL_BOUNDARY);
    task.set_dependency(this_grid.get_dependency());
    task.set_subgrid(igrid);
    task.set_interaction_direction(TRAVELDIRECTION_FACE_Z_P);
    this_grid.set_hydro_task(14, next_task);
  } else {
    const size_t next_task = tasks.get_free_element();
    Task &task = tasks[next_task];
    task.set_type(TASKTYPE_FLUXSWEEP_EXTERNAL_NEIGHBOUR);
    if (igrid < ngbz) {
      task.set_dependency(this_grid.get_dependency());
      task.set_extra_dependency(gridvec[ngbz]->get_dependency());
    } else {
      task.set_dependency(gridvec[ngbz]->get_dependency());
      task.set_extra_dependency(this_grid.get_dependency());
    }
    task.set_subgrid(igrid);
    task.set_buffer(ngbz);
    task.set_interaction_direction(TRAVELDIRECTION_FACE_Z_P);
    this_grid.set_hydro_task(14, next_task);
  }
  if (this_grid.get_neighbour(TRAVELDIRECTION_FACE_Z_N) == NEIGHBOUR_OUTSIDE) {
    const size_t next_task = tasks.get_free_element();
    Task &task = tasks[next_task];
    task.set_type(TASKTYPE_FLUXSWEEP_EXTERNAL_BOUNDARY);
    task.set_dependency(this_grid.get_dependency());
    task.set_subgrid(igrid);
    task.set_interaction_direction(TRAVELDIRECTION_FACE_Z_N);
    this_grid.set_hydro_task(15, next_task);
  } else {
    this_grid.set_hydro_task(15, NO_TASK);
  }
  // conserved variable update
  {
    const size_t next_task = tasks.get_free_element();
    Task &task = tasks[next_task];
    task.set_type(TASKTYPE_UPDATE_CONSERVED);
    task.set_dependency(this_grid.get_dependency());
    task.set_subgrid(igrid);
    this_grid.set_hydro_task(16, next_task);
  }
  // primitive variable update
  {
    const size_t next_task = tasks.get_free_element();
    Task &task = tasks[next_task];
    task.set_type(TASKTYPE_UPDATE_PRIMITIVES);
    task.set_dependency(this_grid.get_dependency());
    task.set_subgrid(igrid);
    this_grid.set_hydro_task(17, next_task);
  }
}

/**
 * @brief Reset the hydro tasks for the given subgrid.
 *
 * @param tasks Tasks.
 * @param this_grid Subgrid.
 */
inline void reset_hydro_tasks(ThreadSafeVector< Task > &tasks,
                              DensitySubGrid &this_grid) {

  // gradient sweeps
  // internal
  tasks[this_grid.get_hydro_task(0)].set_number_of_unfinished_parents(0);
  // external
  tasks[this_grid.get_hydro_task(1)].set_number_of_unfinished_parents(0);
  if (this_grid.get_hydro_task(2) != NO_TASK) {
    tasks[this_grid.get_hydro_task(2)].set_number_of_unfinished_parents(0);
  }
  tasks[this_grid.get_hydro_task(3)].set_number_of_unfinished_parents(0);
  if (this_grid.get_hydro_task(4) != NO_TASK) {
    tasks[this_grid.get_hydro_task(4)].set_number_of_unfinished_parents(0);
  }
  tasks[this_grid.get_hydro_task(5)].set_number_of_unfinished_parents(0);
  if (this_grid.get_hydro_task(6) != NO_TASK) {
    tasks[this_grid.get_hydro_task(6)].set_number_of_unfinished_parents(0);
  }

  // slope limiter
  tasks[this_grid.get_hydro_task(7)].set_number_of_unfinished_parents(7);
  // primitive variable prediction
  tasks[this_grid.get_hydro_task(8)].set_number_of_unfinished_parents(1);

  // flux sweeps
  // internal
  tasks[this_grid.get_hydro_task(9)].set_number_of_unfinished_parents(1);
  // external
  if (tasks[this_grid.get_hydro_task(10)].get_type() ==
      TASKTYPE_FLUXSWEEP_EXTERNAL_BOUNDARY) {
    tasks[this_grid.get_hydro_task(10)].set_number_of_unfinished_parents(1);
  } else {
    tasks[this_grid.get_hydro_task(10)].set_number_of_unfinished_parents(2);
  }
  if (this_grid.get_hydro_task(11) != NO_TASK) {
    tasks[this_grid.get_hydro_task(11)].set_number_of_unfinished_parents(1);
  }
  if (tasks[this_grid.get_hydro_task(12)].get_type() ==
      TASKTYPE_FLUXSWEEP_EXTERNAL_BOUNDARY) {
    tasks[this_grid.get_hydro_task(12)].set_number_of_unfinished_parents(1);
  } else {
    tasks[this_grid.get_hydro_task(12)].set_number_of_unfinished_parents(2);
  }
  if (this_grid.get_hydro_task(13) != NO_TASK) {
    tasks[this_grid.get_hydro_task(13)].set_number_of_unfinished_parents(1);
  }
  if (tasks[this_grid.get_hydro_task(14)].get_type() ==
      TASKTYPE_FLUXSWEEP_EXTERNAL_BOUNDARY) {
    tasks[this_grid.get_hydro_task(14)].set_number_of_unfinished_parents(1);
  } else {
    tasks[this_grid.get_hydro_task(14)].set_number_of_unfinished_parents(2);
  }
  if (this_grid.get_hydro_task(15) != NO_TASK) {
    tasks[this_grid.get_hydro_task(15)].set_number_of_unfinished_parents(1);
  }

  // conserved variable update
  tasks[this_grid.get_hydro_task(16)].set_number_of_unfinished_parents(7);
  // primitive variable update
  tasks[this_grid.get_hydro_task(17)].set_number_of_unfinished_parents(1);
}

/**
 * @brief Set the task dependencies for the given subgrid.
 *
 * @param igrid Subgrid index.
 * @param gridvec Subgrids.
 * @param tasks Tasks.
 */
inline void set_dependencies(const unsigned int igrid,
                             const std::vector< DensitySubGrid * > &gridvec,
                             ThreadSafeVector< Task > &tasks) {

  const DensitySubGrid &this_grid = *gridvec[igrid];

  // gradient sweeps unlock slope limiter tasks
  const size_t igg = this_grid.get_hydro_task(0);
  const size_t igxp = this_grid.get_hydro_task(1);
  size_t igxn = this_grid.get_hydro_task(2);
  if (igxn == NO_TASK) {
    igxn = gridvec[this_grid.get_neighbour(TRAVELDIRECTION_FACE_X_N)]
               ->get_hydro_task(1);
  }
  const size_t igyp = this_grid.get_hydro_task(3);
  size_t igyn = this_grid.get_hydro_task(4);
  if (igyn == NO_TASK) {
    igyn = gridvec[this_grid.get_neighbour(TRAVELDIRECTION_FACE_Y_N)]
               ->get_hydro_task(3);
  }
  const size_t igzp = this_grid.get_hydro_task(5);
  size_t igzn = this_grid.get_hydro_task(6);
  if (igzn == NO_TASK) {
    igzn = gridvec[this_grid.get_neighbour(TRAVELDIRECTION_FACE_Z_N)]
               ->get_hydro_task(5);
  }

  const size_t isl = this_grid.get_hydro_task(7);
  tasks[igg].add_child(isl);
  tasks[igxp].add_child(isl);
  tasks[igxn].add_child(isl);
  tasks[igyp].add_child(isl);
  tasks[igyn].add_child(isl);
  tasks[igzp].add_child(isl);
  tasks[igzn].add_child(isl);

  // the slope limiter task unlocks the gradient prediction task
  const size_t ipp = this_grid.get_hydro_task(8);
  tasks[isl].add_child(ipp);

  // the gradient prediction task unlocks the flux task for this cell and all
  // neighbouring cell pair flux tasks
  const size_t iff = this_grid.get_hydro_task(9);
  tasks[ipp].add_child(iff);
  // neighbours: the positive one is (always) stored in this subgrid
  const size_t ifxp = this_grid.get_hydro_task(10);
  tasks[ipp].add_child(ifxp);
  // the negative one is only stored in this subgrid if it is a non-periodic
  // boundary
  size_t ifxn = this_grid.get_hydro_task(11);
  if (ifxn == NO_TASK) {
    ifxn = gridvec[this_grid.get_neighbour(TRAVELDIRECTION_FACE_X_N)]
               ->get_hydro_task(10);
  }
  tasks[ipp].add_child(ifxn);
  const size_t ifyp = this_grid.get_hydro_task(12);
  tasks[ipp].add_child(ifyp);
  size_t ifyn = this_grid.get_hydro_task(13);
  if (ifyn == NO_TASK) {
    ifyn = gridvec[this_grid.get_neighbour(TRAVELDIRECTION_FACE_Y_N)]
               ->get_hydro_task(12);
  }
  tasks[ipp].add_child(ifyn);
  const size_t ifzp = this_grid.get_hydro_task(14);
  tasks[ipp].add_child(ifzp);
  size_t ifzn = this_grid.get_hydro_task(15);
  if (ifzn == NO_TASK) {
    ifzn = gridvec[this_grid.get_neighbour(TRAVELDIRECTION_FACE_Z_N)]
               ->get_hydro_task(14);
  }
  tasks[ipp].add_child(ifzn);

  // the flux tasks unlock the conserved variable update
  const size_t icu = this_grid.get_hydro_task(16);
  tasks[iff].add_child(icu);
  tasks[ifxp].add_child(icu);
  tasks[ifxn].add_child(icu);
  tasks[ifyp].add_child(icu);
  tasks[ifyn].add_child(icu);
  tasks[ifzp].add_child(icu);
  tasks[ifzn].add_child(icu);

  // the conserved variable update unlocks the primitive variable update
  const size_t ipu = this_grid.get_hydro_task(17);
  tasks[icu].add_child(ipu);
}

/**
 * @brief Execute a task.
 *
 * @param itask Task index.
 * @param gridvec Subgrids.
 * @param tasks Tasks.
 * @param timestep System time step (in s).
 * @param hydro Hydro instance to use.
 * @param boundary HydroBoundary to use.
 */
template < typename _boundary_ >
inline void execute_task(const size_t itask,
                         std::vector< DensitySubGrid * > gridvec,
                         ThreadSafeVector< Task > &tasks, const double timestep,
                         const Hydro &hydro, const _boundary_ &boundary) {

  const Task &task = tasks[itask];
  switch (task.get_type()) {
  case TASKTYPE_GRADIENTSWEEP_INTERNAL:
    gridvec[task.get_subgrid()]->inner_gradient_sweep(hydro);
    break;
  case TASKTYPE_GRADIENTSWEEP_EXTERNAL_NEIGHBOUR:
    gridvec[task.get_subgrid()]->outer_gradient_sweep(
        task.get_interaction_direction(), hydro, *gridvec[task.get_buffer()]);
    break;
  case TASKTYPE_GRADIENTSWEEP_EXTERNAL_BOUNDARY:
    gridvec[task.get_subgrid()]->outer_ghost_gradient_sweep(
        task.get_interaction_direction(), hydro, boundary);
    break;
  case TASKTYPE_SLOPE_LIMITER:
    gridvec[task.get_subgrid()]->apply_slope_limiter(hydro);
    break;
  case TASKTYPE_PREDICT_PRIMITIVES:
    gridvec[task.get_subgrid()]->predict_primitive_variables(hydro,
                                                             0.5 * timestep);
    break;
  case TASKTYPE_FLUXSWEEP_INTERNAL:
    gridvec[task.get_subgrid()]->inner_flux_sweep(hydro);
    break;
  case TASKTYPE_FLUXSWEEP_EXTERNAL_NEIGHBOUR:
    gridvec[task.get_subgrid()]->outer_flux_sweep(
        task.get_interaction_direction(), hydro, *gridvec[task.get_buffer()]);
    break;
  case TASKTYPE_FLUXSWEEP_EXTERNAL_BOUNDARY:
    gridvec[task.get_subgrid()]->outer_ghost_flux_sweep(
        task.get_interaction_direction(), hydro, boundary);
    break;
  case TASKTYPE_UPDATE_CONSERVED:
    gridvec[task.get_subgrid()]->update_conserved_variables(timestep);
    break;
  case TASKTYPE_UPDATE_PRIMITIVES:
    gridvec[task.get_subgrid()]->update_primitive_variables(hydro);
    break;
  default:
    cmac_error("Unknown hydro task: %i", task.get_type());
  }
}

/**
 * @brief Initialize MPI.
 *
 * @param argc Number of command line arguments.
 * @param argv Command line arguments.
 * @param MPI_rank Variable to store the active MPI rank in.
 * @param MPI_size Variable to store the total MPI size in.
 */
inline void initialize_MPI(int &argc, char **argv, int &MPI_rank,
                           int &MPI_size) {

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &MPI_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &MPI_size);

  if (MPI_rank == 0) {
    if (MPI_size > 1) {
      logmessage("Running on " << MPI_size << " processes.", 0);
    } else {
      logmessage("Running on a single process.", 0);
    }
  }
}

/**
 * @brief Parse the command line options.
 *
 * @param argc Number of command line options.
 * @param argv Command line options.
 * @param num_threads_request Variable to store the requested number of threads
 * in.
 * @param paramfile_name Variable to store the parameter file name in.
 */
inline void parse_command_line(int argc, char **argv, int &num_threads_request,
                               std::string &paramfile_name) {

  CommandLineParser commandlineparser("testDensitySubGrid");
  commandlineparser.add_required_option< int_fast32_t >(
      "threads", 't', "Number of shared memory threads to use.");
  commandlineparser.add_required_option< std::string >(
      "params", 'p', "Name of the parameter file.");
  commandlineparser.parse_arguments(argc, argv);

  num_threads_request = commandlineparser.get_value< int_fast32_t >("threads");
  paramfile_name = commandlineparser.get_value< std::string >("params");
}

/**
 * @brief Set the number of threads to use during the simulation.
 *
 * We first determine the number of threads available (either by system default,
 * or because the user has set the OMP_NUM_THREADS environment variable). We
 * then check if a number of threads was specified on the command line. We don't
 * allow setting the number of threads to a value larger than available, and use
 * the available number as default if no value was given on the command line. If
 * the requested number of threads is larger than what is available, we display
 * a message.
 *
 * @param num_threads_request Requested number of threads.
 * @param num_threads Variable to store the actual number of threads that will
 * be used in.
 */
inline void set_number_of_threads(int num_threads_request, int &num_threads) {

  // check how many threads are available
  int num_threads_available;
#pragma omp parallel
  {
#pragma omp single
    num_threads_available = omp_get_num_threads();
  }

  // now check if this is compatible with what was requested
  if (num_threads_request > num_threads_available) {
    // NO: warn the user
    logmessage("More threads requested ("
                   << num_threads_request << ") than available ("
                   << num_threads_available
                   << "). Resetting to maximum available number of threads.",
               0);
    num_threads_request = num_threads_available;
  }

  // set the number of threads to the requested/maximal allowed value
  omp_set_num_threads(num_threads_request);
  num_threads = num_threads_request;

  logmessage("Running with " << num_threads << " threads.", 0);
}

/**
 * @brief Test for the hydro.
 *
 * @param argc Number of command line arguments.
 * @param argv Command line arguments.
 * @return Exit code: 0 on success.
 */
int main(int argc, char **argv) {

  initialize_MPI(argc, argv, MPI_rank, MPI_size);

  MPI_Barrier(MPI_COMM_WORLD);
  // start timing when all processes are at the same point (to make sure the
  // timelines are compatible)
  Timer program_timer;
  program_timer.start();

  unsigned long program_start, program_end;
  cpucycle_tick(program_start);

  int num_threads_request;
  std::string paramfile_name;
  parse_command_line(argc, argv, num_threads_request, paramfile_name);

  int num_threads;
  set_number_of_threads(num_threads_request, num_threads);

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
  const int ncell[3] = {128, 128, 128};
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

  ThreadSafeVector< Task > tasks(18 * tot_num_subgrid);

  for (unsigned int igrid = 0; igrid < tot_num_subgrid; ++igrid) {
    gridvec[igrid]->initialize_conserved_variables(hydro);
    make_hydro_tasks(tasks, igrid, gridvec);
  }
  // all hydro tasks have been made, link the dependencies
  for (unsigned int igrid = 0; igrid < tot_num_subgrid; ++igrid) {
    set_dependencies(igrid, gridvec, tasks);
  }

  Queue hydro_queue(18 * tot_num_subgrid);

  const double dt = 0.001;
  Timer hydro_loop_timer;
  hydro_loop_timer.start();
  for (unsigned int istep = 0; istep < 100; ++istep) {

    logmessage("Step " << istep, 0);

    myassert(hydro_queue.size() == 0, "Wrong queue size!");

    // reset the hydro tasks and add them to the queue
    for (unsigned int igrid = 0; igrid < tot_num_subgrid; ++igrid) {
      reset_hydro_tasks(tasks, *gridvec[igrid]);
      for (int i = 0; i < 18; ++i) {
        const size_t itask = gridvec[igrid]->get_hydro_task(i);
        if (itask != NO_TASK &&
            tasks[itask].get_number_of_unfinished_parents() == 0) {
          hydro_queue.add_task(itask);
        }
      }
    }

    unsigned long iteration_start, iteration_end;
    cpucycle_tick(iteration_start);
#pragma omp parallel default(shared)
    {
      const int thread_id = omp_get_thread_num();
      size_t current_task = hydro_queue.get_task(tasks);
      while (current_task != NO_TASK) {
        tasks[current_task].start(thread_id);
        execute_task(current_task, gridvec, tasks, dt, hydro, inflow_boundary);
        tasks[current_task].stop();
        tasks[current_task].unlock_dependency();
        const unsigned char numchild =
            tasks[current_task].get_number_of_children();
        for (unsigned char i = 0; i < numchild; ++i) {
          const size_t ichild = tasks[current_task].get_child(i);
          myassert(ichild != NO_TASK, "Child task does not exist!");
          if (tasks[ichild].decrement_number_of_unfinished_parents() == 0) {
            hydro_queue.add_task(ichild);
          }
        }
        current_task = hydro_queue.get_task(tasks);
      }
    }
    cpucycle_tick(iteration_end);

    output_tasks(istep, tasks, iteration_start, iteration_end);

    myassert(hydro_queue.size() == 0, "Queue not empty!");
  }
  hydro_loop_timer.stop();

  // stop the timers (after all processes synchronize)
  MPI_Barrier(MPI_COMM_WORLD);
  program_timer.stop();
  cpucycle_tick(program_end);

  // write the start and end CPU cycle count for each process, and the total
  // program time (for tick to time conversion)
  {
    std::string filename = "program_time.txt";
    for (int irank = 0; irank < MPI_size; ++irank) {
      // only the active process writes
      if (irank == MPI_rank) {
        // the file mode depends on the rank
        std::ios_base::openmode mode;
        if (irank == 0) {
          // rank 0 creates (or overwrites) the file
          mode = std::ofstream::trunc;
        } else {
          // all other ranks append to it
          mode = std::ofstream::app;
        }
        // now open the file
        std::ofstream ofile(filename, mode);

        // rank 0 writes the header
        if (irank == 0) {
          ofile << "# rank\tstart\tstop\ttime\n";
        }
        ofile << MPI_rank << "\t" << program_start << "\t" << program_end
              << "\t" << program_timer.value() << "\n";
      }
      // only one process at a time is allowed to write
      MPI_Barrier(MPI_COMM_WORLD);
    }
  }

  cmac_status("Total hydro loop time: %g s", hydro_loop_timer.value());

  output_result(gridvec, tot_num_subgrid);

  const int MPI_exit_code = MPI_Finalize();
  return MPI_exit_code;
}
