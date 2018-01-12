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
 * @file testCostVector.cpp
 *
 * @brief Unit test for the CostVector class.
 *
 * @author Bert Vandenbroucke (bv7@st-andrews.ac.uk)
 */

/*! @brief Output log level. The higher the value, the more stuff is printed to
 *  the stderr. Comment to disable logging altogether. */
#define LOG_OUTPUT 1

#include "CostVector.hpp"
#include "Log.hpp"
#include "RandomGenerator.hpp"

#include <fstream>

/**
 * @brief Unit test for the CostVector class.
 *
 * @param argc Number of command line arguments.
 * @param argv Command line arguments.
 * @return Exit code: 0 on success.
 */
int main(int argc, char **argv) {

  RandomGenerator random_generator;
  CostVector costs(100, 16, 4);
  unsigned long cost_list[100];
  for (size_t i = 0; i < 100; ++i) {
    cost_list[i] = random_generator.get_uniform_random_double() * 0xffffffff;
    costs.add_cost(i, cost_list[i]);
  }
  // add an element with a ridiculously high cost to see how the algorithm
  // copes
  cost_list[42] = 0xfffffffff;
  costs.set_cost(42, cost_list[42]);
  costs.redistribute();
  std::ofstream ofile("cost_test.txt");
  ofile << "# element\tcost\trank\tthread\n";
  for (size_t i = 0; i < 100; ++i) {
    ofile << i << "\t" << cost_list[i] << "\t" << costs.get_process(i) << "\t"
          << costs.get_thread(i) << "\n";
  }

  logmessage("Cost distribution written to cost_test.txt.", 0);

  return 0;
}
