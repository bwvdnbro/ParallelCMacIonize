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
 * @file CostVector.hpp
 *
 * @brief Object that keeps track of computational costs and tries to balance
 * them.
 *
 * @author Bert Vandenbroucke (bv7@st-andrews.ac.uk)
 */
#ifndef COSTVECTOR_HPP
#define COSTVECTOR_HPP

#include "Assert.hpp"

#include <metis.h>

#include <algorithm>
#include <vector>

/**
 * @brief Sort the given array with given size by argument.
 *
 * @param v Array to sort.
 * @param size Size of the array.
 * @return std::vector<size_t> containing the indices of the array in an order
 * that would sort the array.
 */
template < typename _datatype_ >
inline std::vector< size_t > argsort(const _datatype_ *v, const size_t size) {
  std::vector< size_t > indices(size, 0);
  std::iota(indices.begin(), indices.end(), 0);
  std::sort(indices.begin(), indices.end(),
            [v](size_t i1, size_t i2) { return v[i1] < v[i2]; });
  return indices;
}

/**
 * @brief Object that keeps track of computational costs and tries to balance
 * them.
 */
class CostVector {
private:
  /*! @brief Number of elements. */
  size_t _size;

  /*! @brief Number of threads. */
  const int _number_of_threads;

  /*! @brief Number of MPI processes. */
  const int _number_of_processes;

  /*! @brief Computational costs for each element. */
  unsigned long *_costs;

  /*! @brief Source cost for each element. */
  unsigned int *_source_costs;

  /*! @brief Thread list that links elements to threads in an optimal balancing
   *  scheme. */
  int *_thread_list;

  /*! @brief Process list that links elements to MPI processes in an optimal
   *  balancing scheme. */
  int *_process_list;

public:
  /**
   * @brief Constructor.
   *
   * @param size Number of elements.
   * @param number_of_threads Number of threads.
   * @param number_of_proceses Number of MPI processes.
   */
  inline CostVector(const size_t size, const int number_of_threads,
                    const int number_of_processes)
      : _size(size), _number_of_threads(number_of_threads),
        _number_of_processes(number_of_processes) {

    _costs = new unsigned long[size];
    _source_costs = new unsigned int[size];
    _thread_list = new int[size];
    _process_list = new int[size];
    for (size_t i = 0; i < size; ++i) {
      _costs[i] = 0;
      _source_costs[i] = 0;
      // our initial decomposition
      _thread_list[i] = i % number_of_threads;
      _process_list[i] = i % number_of_processes;
    }
  }

  /**
   * @brief Destructor.
   */
  inline ~CostVector() {
    delete[] _costs;
    delete[] _source_costs;
    delete[] _thread_list;
    delete[] _process_list;
  }

  /**
   * @brief Reset the size of the cost vector.
   *
   * @param size New size.
   */
  inline void reset(const size_t size) {
    _size = size;

    delete[] _costs;
    delete[] _source_costs;
    delete[] _thread_list;
    delete[] _process_list;

    _costs = new unsigned long[size];
    _source_costs = new unsigned int[size];
    _thread_list = new int[size];
    _process_list = new int[size];

    for (size_t i = 0; i < size; ++i) {
      _costs[i] = 0;
      _source_costs[i] = 0;
      // our initial decomposition
      _thread_list[i] = i % _number_of_threads;
      _process_list[i] = i % _number_of_processes;
    }
  }

  /**
   * @brief Get the thread id for the given element.
   *
   * @param index Index of an element.
   * @return Thread id that owns that element.
   */
  inline const int get_thread(const size_t index) const {
    return _thread_list[index];
  }

  /**
   * @brief Get the process id for the given element.
   *
   * @param index Index of an element.
   * @return Process rank that owns that element.
   */
  inline const int get_process(const size_t index) const {
    return _process_list[index];
  }

  /**
   * @brief Add the given computational cost to the given element.
   *
   * @param index Index of an element.
   * @param cost Computational cost to add.
   */
  inline void add_cost(const size_t index, const unsigned long cost) {
    _costs[index] += cost;
  }

  /**
   * @brief Set the computational cost for the given element.
   *
   * @param index Index of an element.
   * @param cost New computation cost.
   */
  inline void set_cost(const size_t index, const unsigned long cost) {
    _costs[index] = cost;
  }

  /**
   * @brief Get the computational cost for the given element.
   *
   * @param index Index of an element.
   * @return Computational cost for that element.
   */
  inline const unsigned long get_cost(const size_t index) const {
    return _costs[index];
  }

  /**
   * @brief Add the given source cost to the given element.
   *
   * @param index Index of an element.
   * @param cost Source cost to add.
   */
  inline void add_source_cost(const size_t index, const unsigned int cost) {
    _source_costs[index] += cost;
  }

  /**
   * @brief Set the source cost for the given element.
   *
   * @param index Index of an element.
   * @param cost New source cost.
   */
  inline void set_source_cost(const size_t index, const unsigned int cost) {
    _source_costs[index] = cost;
  }

  /**
   * @brief Get the source cost for the given element.
   *
   * @param index Index of an element.
   * @return Source cost for that element.
   */
  inline const unsigned int get_source_cost(const size_t index) const {
    return _source_costs[index];
  }

  /**
   * @brief Redistribute the elements such that the computational cost for each
   * thread is as close as possible to the average.
   *
   * @param ngbs Neighbour graph that stores the communication neighbours for
   * each subgrid (and copy) in the list on the local processor.
   */
  inline void
  redistribute(const std::vector< std::vector< unsigned int > > &ngbs) {

    /// first step: MPI distribution

    if (_number_of_processes > 1) {

      myassert(ngbs.size() == _size, "Graph has wrong size!");

      idx_t nvert = _size;
      idx_t nedge = 0;
      for (size_t igrid = 0; igrid < _size; ++igrid) {
        nedge += ngbs[igrid].size();
      }
      // we counted each edge twice
      nedge >>= 1;

      // construct the METIS graph from the ngbs
      // we have 3 weights (see below)
      idx_t ncon = 3;
      // edge offsets: xadj[0] stores the offset of the edge list of vertex 0 in
      //  adjncy, xadj[1] is the offset of vertex 1... the extra element gives
      //  the
      //  total number of edges (times 2, as an edge from a to b is counted
      //  twice:
      //  a-b and b-a)
      idx_t *xadj = new idx_t[nvert + 1];
      // actual edges: adjncy[0] stores the vertex on the other side of the
      // first
      //  edge of vertex 0, etc.
      idx_t *adjncy = new idx_t[2 * nedge];
      // vertex weights: the first 3 elements correspond to the 3 weights for
      //  vertex 0, etc.
      idx_t *vwgt = new idx_t[ncon * nvert];
      // edge weights: weight for every entry in adjncy. We act under the
      //  assumption that both entries for the same edge need to have the same
      //  weight, although we did not test if not doing this actually results in
      //  an error (and the documentation does not mention this)
      idx_t *adjwgt = new idx_t[2 * nedge];

      // the first offset is always trivially zero
      xadj[0] = 0;
      for (size_t igrid = 0; igrid < _size; ++igrid) {

        // we have 3 weights that we want to equally distribute:
        //  - the computational cost
        vwgt[3 * igrid + 0] = _costs[igrid];
        //  - the memory (each subgrid has the same memory requirements for now)
        vwgt[3 * igrid + 1] = 1;
        //  - the number of sources per domain
        vwgt[3 * igrid + 2] = _source_costs[igrid];

        // xadj[igrid] points to the beginning of the edge list for this vertex
        // in adjncy
        // xadj[igrid+1] points to the element beyond the edge list for this
        // vertex (similar to iterator::end())
        xadj[igrid + 1] = xadj[igrid] + ngbs[igrid].size();

        for (unsigned int ingb = 0; ingb < ngbs[igrid].size(); ++ingb) {
          adjncy[xadj[igrid] + ingb] = ngbs[igrid][ingb];
          // all edges have the same weight for now (we should base this on the
          //  communication during the previous step, but we're not entirely
          //  clear on how to store this correctly)
          adjwgt[xadj[igrid] + ingb] = 1;
        }
      }
      myassert(xadj[nvert] == 2 * nedge, "Wrong number of edges!");

      idx_t *part = new idx_t[nvert];
      idx_t nparts = _number_of_processes;
      idx_t edgecut;

      int metis_status = METIS_PartGraphKway(&nvert, &ncon, xadj, adjncy, vwgt,
                                             nullptr, adjwgt, &nparts, nullptr,
                                             nullptr, nullptr, &edgecut, part);

      if (metis_status != METIS_OK) {
        cmac_error("Metis error!");
      }

      for (size_t igrid = 0; igrid < _size; ++igrid) {
        _process_list[igrid] = part[igrid];
      }

      // clean up METIS arrays
      delete[] xadj;
      delete[] adjncy;
      delete[] vwgt;
      delete[] adjwgt;
      delete[] part;

    } else { // if more than 1 process

      for (size_t igrid = 0; igrid < _size; ++igrid) {
        _process_list[igrid] = 0;
      }
    }

    // argsort the elements based on cost
    std::vector< size_t > indices = argsort(_costs, _size);

    const size_t max_index = _size - 1;
    // loop over the processes and load balance the threads on each process
    for (int irank = 0; irank < _number_of_processes; ++irank) {
      size_t index = 0;
      while (index < _size && _process_list[index] != irank) {
        ++index;
      }
      myassert(index < _size, "Not enough subgrids!");
      std::vector< unsigned long > threadcost(_number_of_threads, 0);
      for (int ithread = 0; ithread < _number_of_threads; ++ithread) {
        const size_t current_index = indices[max_index - index];
        _thread_list[current_index] = ithread;
        threadcost[ithread] += _costs[current_index];
        ++index;
        while (index < _size && _process_list[index] != irank) {
          ++index;
        }
        myassert(index < _size, "Not enough subgrids!");
      }
      for (; index < _size; ++index) {
        if (_process_list[index] == irank) {
          const size_t current_index = indices[max_index - index];
          // find the thread where this cost has the lowest impact
          int cmatch_thread = -1;
          unsigned long cmatch = threadcost[0] + _costs[current_index];
          for (int ithread = 0; ithread < _number_of_threads; ++ithread) {
            const unsigned long cvalue =
                threadcost[ithread] + _costs[current_index];
            if (cvalue <= cmatch) {
              cmatch = cvalue;
              cmatch_thread = ithread;
            }
          }
          myassert(cmatch_thread >= 0, "No closest match!");
          _thread_list[current_index] = cmatch_thread;
          threadcost[cmatch_thread] += _costs[current_index];
        }
      }
    }

    // reset costs
    for (size_t i = 0; i < _size; ++i) {
      _costs[i] = 0;
    }
  }
};

#endif // COSTVECTOR_HPP
