#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <mpi.h>
#include <sstream>
#include <string>
#include <vector>

/**
 * This file contains a very simple parallel algorithm with uncoordinated
 * communication. The idea is that we have a 1D grid of cells, with each cell
 * having 2 neighbouring cells (periodially wrapped at the edges).
 * Each cell holds an integer value, which is initialised to the same value for
 * all cells. Each cell also has a unique random sequence assigned to it (you
 * could think of it as each cell having a unique random generator that will
 * always produce the same random sequence for that specific cell).
 *
 * The algorithm we perform is very simple:
 * as long as a cell has a positive non-zero value for its integer variable:
 *   - we subtract 1 from the integer variable
 *   - we draw a random number (the next random number in the sequence)
 *   - what happens next depends on that random number:
 *       - if that random number is smaller than some constant value, we
 *         increase the integer variable for one of its 2 neighbours with 1,
 *         with a 50% probability of choosing one or the other
 *       - if that random number is larger, we do nothing
 * For each iteration of the above loop, we add 1 to an interaction counter.
 * Once all cells have completed the loop, we compute the average number of
 * interactions per cell. Theoretically, this number should be
 * \f[
 *   \langle{} N_{int} \rangle{} = \sum_{i=0}^{N} N p^i,
 * \f]
 * with \f$N\f$ the starting value of the integer variable, and \f$p\f$ the
 * threshold probability for increasing a neighbour counter.
 *
 * This algorithm is tricky to parallelize, as we have neighbour interactions
 * across nodes between some cells, and we do not a priori know how many of
 * those we will have. Furthermore, cells or even entire processes that already
 * finished could be woken up again by an interaction with a neighbouring cell.
 * We do know how many interactions have been generated at any given time, and
 * how many completed, and we can use this information to deduce how many cells
 * are still active. This can be used to set up an uncoordinated communication
 * scheme.
 *
 * The way this works is as follows: as long as a process has work to do, it
 * does work and occasionally (e.g. after a cell has been processed) checks if
 * an MPI communication is coming in (non-blocking). As soon as there is no more
 * work, the process notifies a master process (process 0) that it is ready, and
 * then enters a blocking communication loop (since a process can only be woken
 * up again through an incoming communication). The master process gathers the
 * number of interactions that has been created (either from the start, or by
 * interaction with a neighbour), and the number of interactions that has been
 * completed, globally for all processes. If the master process has no more
 * work, it performs a preliminary check on these values to see if they are
 * equal. If they are, there is a possibility that we finished, and a more
 * strict test needs to be performed (since the values might accidentally be
 * equal for other reasons). The more strict test is initiated by the master by
 * sending a dedicated message to all other processes. All other processes
 * receive this message when they have time and upon receiving it enter a
 * blocking collective communication to gather the actual total number of
 * created and completed interactions. If these values match, the master sends
 * a final signal to all processes to inform them that the simulation finished.
 * If not, the master goes into a blocking receive until at least another
 * process finished, or it receives more interactions itself, after which the
 * whole process repeats.
 *
 * There are some key aspects for this algorithm to work:
 *   - we know how many interactions need to be done at any given time, and how
 *     many completed. If one process has access to this information for all
 *     processes, it can decide whether we finished or not.
 *   - a process can get a good guess of whether a finish is likely by gathering
 *     these numbers from individual processes, without the need for a global
 *     (blocking) communication
 *   - we need to keep track of how many interactions were created and completed
 *     between individual sends to master to make sure the master does not need
 *     to store an individual counter per process: a contribution that has been
 *     send to master cannot be send again
 *   - the master needs to inform the other processes that we really finished in
 *     a separate message, so that we have a clean way of exiting the
 *     communication loop (and so that we can use a unilateral collective
 *     communication)
 */

/*! @brief Threshold probability \f$p\f$ of a cell to affect one of its
 *  neighbours during an interaction. */
#define GLOBAL_PROBABILITY 0.4

/*! @brief Total number of cells. */
#define GLOBAL_NUMBER_OF_CELLS 10000u

/*! @brief Starting value \f$N\f$ for the cells. */
#define GLOBAL_STARTING_VALUE 1000u

/**
 * @brief Generate a uniform random double using the default C++ random
 * generator.
 *
 * @return Uniform random double in the range [0, 1[.
 */
double random_double() { return ((double)rand()) / ((double)RAND_MAX); }

/**
 * @brief Single cell of the 1D grid.
 */
class Cell {
public:
  /*! @brief Integer variable that determines whether a cell is allowed to
   *  interact or not. */
  unsigned int _integer_variable;

  /*! @brief Index of the neighbouring cell on the "left". */
  int _left_neighbour;

  /*! @brief Index of the neighbouring cell on the "right". */
  int _right_neighbour;

  /*! @brief Interaction counter \f$N_{int}\f$. */
  unsigned int _number_of_interactions;

  /*! @brief Flag that tells us whether this cell is local or not. */
  bool _local;

  /*! @brief Current value of the random seed, used to keep track of the random
   *  sequence for this cell. */
  unsigned int _seed;

  /**
   * @brief Constructor.
   *
   * @param left_neighbour Index of the neighbouring cell on the "left".
   * @param right_neighbour Index of the neighbouring cell on the "right".
   * @param local Flag that tells us whether this cell is local or not.
   * @param seed Initial value of the random seed, used to set the random
   * sequence for this cell.
   */
  Cell(int left_neighbour, int right_neighbour, bool local, unsigned int seed)
      : _integer_variable(GLOBAL_STARTING_VALUE),
        _left_neighbour(left_neighbour), _right_neighbour(right_neighbour),
        _number_of_interactions(0), _local(local), _seed(seed) {}

  /**
   * @brief Interaction function.
   *
   * @return Index of the cell to update after the interaction, or -1 if no cell
   * needs to be updated, or -2 if the cell cannot interact any more.
   */
  int do_something() {
    if (_integer_variable > 0) {
      srand(_seed);
      _seed = rand();
      ++_number_of_interactions;
      --_integer_variable;
      double x = random_double();
      if (x < GLOBAL_PROBABILITY) {
        x = random_double();
        if (x < 0.5) {
          return _left_neighbour;
        } else {
          return _right_neighbour;
        }
      } else {
        return -1;
      }
    } else {
      return -2;
    }
  }
};

/**
 * @brief Reduce the number of interactions that has been created and has
 * completed over all processes.
 *
 * @param created Number of interactions that has been created locally.
 * @param completed Number of interactions that has completed locally.
 * @param global_buffer Array to store the result of the reduction in.
 */
void do_reduction(unsigned int created, unsigned int completed,
                  unsigned int *global_buffer) {
  unsigned int buffer[2] = {created, completed};
  std::cerr << "Initiating reduction..." << std::endl;
  MPI_Reduce(buffer, global_buffer, 2, MPI_UNSIGNED, MPI_SUM, 0,
             MPI_COMM_WORLD);
}

/**
 * @brief Main program.
 *
 * @param argc Number of command line arguments.
 * @param argv Command line arguments.
 * @return Exit code: 0 on success.
 */
int main(int argc, char **argv) {

  // initialise MPI
  MPI_Init(&argc, &argv);

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  std::stringstream rankstream;
  rankstream << rank << "/" << size << ": ";
  std::string rankstr = rankstream.str();

  // distribute the number of cells equally among all domains
  unsigned int domain_size =
      std::ceil(((double)GLOBAL_NUMBER_OF_CELLS) / ((double)size));
  unsigned int imin = rank * domain_size;
  unsigned int imax =
      std::min((rank + 1) * domain_size, GLOBAL_NUMBER_OF_CELLS);

  // initialise the send and receive counters we use to check if all
  // communications were properly executed
  unsigned int num_send = 0;
  unsigned int num_recv = 0;

  // initialise the cells
  // we initialise all cells on all processes, to make sure the random
  // sequences are the same irrespective of the number of processes used
  srand(42);
  std::vector< Cell > cells;
  cells.reserve(GLOBAL_NUMBER_OF_CELLS);
  for (unsigned int i = 0; i < GLOBAL_NUMBER_OF_CELLS; ++i) {
    int left_neighbour = GLOBAL_NUMBER_OF_CELLS - 1;
    if (i > 0) {
      left_neighbour = i - 1;
    }
    int right_neighbour = 0;
    if (i < GLOBAL_NUMBER_OF_CELLS - 1) {
      right_neighbour = i + 1;
    }
    // the third argument makes sure only local cells are assigned to this
    // specific process
    cells.push_back(
        Cell(left_neighbour, right_neighbour, i >= imin && i < imax, rand()));
  }

  // now do the actual loop
  // we first initialise the number of interactions: this will be at least
  // the global starting value per cell times the number of cells assigned to
  // this process
  unsigned int number_active = imax - imin;
  unsigned int created = number_active * GLOBAL_STARTING_VALUE;
  // none of the interactions have completed yet
  unsigned int completed = 0;
  // this is the number of interactions created and completed since the last
  // time we checked
  unsigned int created_since_last = created;
  unsigned int completed_since_last = completed;
  unsigned int creacombuf[2] = {created_since_last, completed_since_last};
  // this is the total number of interactions created and completed across all
  // processes, needed to check possible finishes on the master rank
  unsigned int total_created = 0;
  unsigned int total_completed = 0;
  // this is the exit flag for our loop
  bool global_finished = false;
  while (!global_finished) {
    // if this process has work left to do: do work
    if (number_active > 0) {
      // loop over all cells and do something for each cell
      for (unsigned int i = imin; i < imax; ++i) {
        int neighbour = cells[i].do_something();
        while (neighbour != -2) {
          ++completed;
          ++completed_since_last;
          if (neighbour >= 0) {
            // new interaction is created
            ++created;
            ++created_since_last;
            ++cells[neighbour]._integer_variable;
            // need to check if this neighbour is local
            // if not, we need to do communication
            if (!cells[neighbour]._local) {
              MPI_Request request;
              int tag, destination;
              if ((neighbour < i && neighbour != 0) ||
                  (i == 0 && neighbour == GLOBAL_NUMBER_OF_CELLS - 1)) {
                tag = 0;
                destination = (rank + size - 1) % size;
              } else {
                tag = 1;
                destination = (rank + 1) % size;
              }
              MPI_Isend(NULL, 0, MPI_INT, destination, tag, MPI_COMM_WORLD,
                        &request);
              ++num_send;
              // we do not care about storing request information in this case,
              // as the communication tag itself contains all the data we want
              // to communicate
              MPI_Request_free(&request);
            }
          }
          neighbour = cells[i].do_something();
        }
        // check for incoming messages after each cell has been handled
        MPI_Status status;
        int flag;
        MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &flag, &status);
        while (flag) {
          int source = status.MPI_SOURCE;
          int tag = status.MPI_TAG;

          if (tag == 0) {
            // receive the (empty) message
            MPI_Recv(NULL, 0, MPI_INT, source, tag, MPI_COMM_WORLD, &status);
            ++num_recv;
            // increase the relevant neighbour: the rightmost one
            ++cells[imax - 1]._integer_variable;
          }

          if (tag == 1) {
            // receive the (empty) message
            MPI_Recv(NULL, 0, MPI_INT, source, tag, MPI_COMM_WORLD, &status);
            ++num_recv;
            // increase the relevant neighbour: the leftmost one
            ++cells[imin]._integer_variable;
          }

          if (tag == 2) {
            // should only happen on rank 0
            if (rank != 0) {
              std::cerr << "Only rank 0 should receive messages with tag 2!"
                        << std::endl;
              MPI_Abort(MPI_COMM_WORLD, 1);
            }
            unsigned int buffer[2];
            MPI_Recv(buffer, 2, MPI_UNSIGNED, source, tag, MPI_COMM_WORLD,
                     &status);
            ++num_recv;
            total_created += buffer[0];
            total_completed += buffer[1];
          }

          if (tag == 3) {
            // receive the (empty) message
            MPI_Recv(NULL, 0, MPI_INT, source, tag, MPI_COMM_WORLD, &status);
            ++num_recv;
            unsigned int global_buffer[2];
            do_reduction(created, completed, global_buffer);
          }

          if (tag == 4) {
            // receive the (empty) message
            MPI_Recv(NULL, 0, MPI_INT, source, tag, MPI_COMM_WORLD, &status);
            ++num_recv;
            // set the global finished flag and break from the probe loop
            global_finished = true;
          }

          MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &flag,
                     &status);
        }
      }
      number_active = 0;
      for (unsigned int i = imin; i < imax; ++i) {
        if (cells[i]._integer_variable > 0) {
          ++number_active;
        }
      }
    }
    // note that the value of number_active could have changed since the last
    // time we checked, so we need to check again
    // we might be able to replace the first condition with a while though...
    // doesn't really matter...
    if (number_active == 0) {
      // no more local work to do, notify master...
      if (rank != 0) {
        // send statistics to the master process (rank 0)
        if (created_since_last > 0 || completed_since_last > 0) {
          creacombuf[0] = created_since_last;
          creacombuf[1] = completed_since_last;
          created_since_last = 0;
          completed_since_last = 0;
          MPI_Request request;
          MPI_Isend(creacombuf, 2, MPI_UNSIGNED, 0, 2, MPI_COMM_WORLD,
                    &request);
          ++num_send;
          MPI_Request_free(&request);
        }
        // okay, done, now keep probing for incoming messages until we either
        // receive new interactions, or the master notifies us that we can stop
      } else {
        // we are the master: compute new totals and check if we could have
        // finished by doing a global reduce
        total_created += created_since_last;
        total_completed += completed_since_last;
        created_since_last = 0;
        completed_since_last = 0;
        std::cout << rankstr << "Totals: " << total_created << " "
                  << total_completed << std::endl;
        if (total_created == total_completed) {
          // possibly finished: ask all other processes to provide their totals
          for (int irank = 1; irank < size; ++irank) {
            MPI_Request request;
            MPI_Isend(NULL, 0, MPI_INT, irank, 3, MPI_COMM_WORLD, &request);
            ++num_send;
            MPI_Request_free(&request);
          }
          // now add the master values to the totals
          unsigned int global_buffer[2];
          do_reduction(created, completed, global_buffer);
          // check if we really finished
          if (global_buffer[0] == global_buffer[1]) {
            // we really finished: tell all other processes to stop
            for (int irank = 1; irank < size; ++irank) {
              MPI_Request request;
              MPI_Isend(NULL, 0, MPI_INT, irank, 4, MPI_COMM_WORLD, &request);
              ++num_send;
              MPI_Request_free(&request);
            }
            global_finished = true;
          }
        }
      }
      // if we have nothing to do, but we're not finished, we need to wait until
      // we either get new interactions, or we finished completely
      // all these involve at least one communication, so we wait for that to
      // arrive
      // (note that we use a blocking MPI_Probe rather than a non-blocking
      //  MPI_Iprobe in this case)
      if (!global_finished) {
        MPI_Status status;
        MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        int source = status.MPI_SOURCE;
        int tag = status.MPI_TAG;

        if (tag == 0) {
          // receive the (empty) message
          MPI_Recv(NULL, 0, MPI_INT, source, tag, MPI_COMM_WORLD, &status);
          ++num_recv;
          // increase nummertjes for the rightmost kotje
          ++cells[imax - 1]._integer_variable;
          ++number_active;
        }

        if (tag == 1) {
          // receive the (empty) message
          MPI_Recv(NULL, 0, MPI_INT, source, tag, MPI_COMM_WORLD, &status);
          ++num_recv;
          // increase nummertjes for the leftmost kotje
          ++cells[imin]._integer_variable;
          ++number_active;
        }

        if (tag == 2) {
          // should only happen on rank 0
          if (rank != 0) {
            std::cerr << "Only rank 0 should receive messages with tag 2!"
                      << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
          }
          unsigned int buffer[2];
          MPI_Recv(buffer, 2, MPI_UNSIGNED, source, tag, MPI_COMM_WORLD,
                   &status);
          ++num_recv;
          total_created += buffer[0];
          total_completed += buffer[1];
        }

        if (tag == 3) {
          // receive the (empty) message
          MPI_Recv(NULL, 0, MPI_INT, source, tag, MPI_COMM_WORLD, &status);
          ++num_recv;
          unsigned int globbuffer[2];
          do_reduction(created, completed, globbuffer);
        }

        if (tag == 4) {
          // receive the (empty) message
          MPI_Recv(NULL, 0, MPI_INT, source, tag, MPI_COMM_WORLD, &status);
          ++num_recv;
          // set the global finished flag and break from the probe loop
          global_finished = true;
        }
      }
    }
  }

  // okay, done. Now compute the averages and output everything to screen.
  double avg = 0;
  unsigned int local_number_of_interactions[GLOBAL_NUMBER_OF_CELLS];
  for (unsigned int i = 0; i < GLOBAL_NUMBER_OF_CELLS; ++i) {
    if (cells[i]._local) {
      avg += cells[i]._number_of_interactions;
      local_number_of_interactions[i] = cells[i]._number_of_interactions;
    } else {
      local_number_of_interactions[i] = 0;
    }
  }
  unsigned int global_number_of_interactions[GLOBAL_NUMBER_OF_CELLS];
  MPI_Allreduce(local_number_of_interactions, global_number_of_interactions,
                GLOBAL_NUMBER_OF_CELLS, MPI_UNSIGNED, MPI_SUM, MPI_COMM_WORLD);

  // gather the global average
  double global_avg;
  MPI_Allreduce(&avg, &global_avg, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  avg = global_avg;
  // gather the global communication statistics
  unsigned int global_num_send, global_num_recv;
  MPI_Allreduce(&num_send, &global_num_send, 1, MPI_UNSIGNED, MPI_SUM,
                MPI_COMM_WORLD);
  MPI_Allreduce(&num_recv, &global_num_recv, 1, MPI_UNSIGNED, MPI_SUM,
                MPI_COMM_WORLD);
  if (rank == 0) {
    // we output the average, the theoretical expectation and the values for
    // some cells (corresponding to the boundary cells for multiple of 4 world
    // sizes). We also output the total communication statistics.
    std::cout << rankstr << "Average: " << (avg / GLOBAL_NUMBER_OF_CELLS)
              << std::endl;
    double theory = 0.;
    for (unsigned int i = 0; i < GLOBAL_STARTING_VALUE; ++i) {
      theory += GLOBAL_STARTING_VALUE * std::pow(GLOBAL_PROBABILITY, i);
    }
    std::cout << rankstr << "Theory: " << theory << std::endl;
    std::cout << "0: " << global_number_of_interactions[0] << std::endl;
    std::cout << "2500: " << global_number_of_interactions[2500] << std::endl;
    std::cout << "4999: " << global_number_of_interactions[4999] << std::endl;
    std::cout << "5000: " << global_number_of_interactions[5000] << std::endl;
    std::cout << "7500: " << global_number_of_interactions[7500] << std::endl;
    std::cout << "9999: " << global_number_of_interactions[9999] << std::endl;
    std::cout << "Total number of messages sent: " << global_num_send
              << std::endl;
    std::cout << "Total number of messages received: " << global_num_recv
              << std::endl;
    // write the values for all cells to a file
    std::ofstream ofile("cells.txt");
    for (unsigned int i = 0; i < GLOBAL_NUMBER_OF_CELLS; ++i) {
      ofile << i << "\t" << global_number_of_interactions[i] << std::endl;
    }
  }

  // output per process communication statistics in an orderly way
  for (int i = 0; i < size; ++i) {
    if (i == rank) {
      std::cout << rankstr << "Num_send: " << num_send << std::endl;
      std::cout << rankstr << "Num_recv: " << num_recv << std::endl;
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }

  return MPI_Finalize();
}
