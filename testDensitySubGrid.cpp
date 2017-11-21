#include "DensitySubGrid.hpp"
#include "RandomGenerator.hpp"

#include <cmath>
#include <fstream>
#include <iostream>

int main(int argc, char **argv) {

  const double box[6] = {-1.543e17, -1.543e17, -1.543e17,
                         3.086e17,  3.086e17,  3.086e17};
  const int ncell[3] = {64, 64, 64};
  DensitySubGrid grid(box, ncell);

  const unsigned int num_photon = 1000000;

  Photon *photons = new Photon[num_photon];
  RandomGenerator random_generator;

  for (unsigned int iloop = 0; iloop < 10; ++iloop) {
    for (unsigned int i = 0; i < num_photon; ++i) {
      Photon &photon = photons[i];
      photon._position[0] = 0.;
      photon._position[1] = 0.;
      photon._position[2] = 0.;
      const double cost =
          2. * random_generator.get_uniform_random_double() - 1.;
      const double sint = std::sqrt(std::max(1. - cost * cost, 0.));
      const double phi =
          2. * M_PI * random_generator.get_uniform_random_double();
      const double cosp = std::cos(phi);
      const double sinp = std::sin(phi);
      photon._direction[0] = sint * cosp;
      photon._direction[1] = sint * sinp;
      photon._direction[2] = cost;
      photon._weight = 1.;
      photon._current_optical_depth = 0.;
      photon._target_optical_depth =
          -std::log(random_generator.get_uniform_random_double());
      photon._photoionization_cross_section = 6.3e-22;
    }

    double output[27];
    for (unsigned int i = 0; i < 27; ++i) {
      output[i] = 0.;
    }
    for (unsigned int i = 0; i < num_photon; ++i) {
      const double inverse_direction[3] = {1. / photons[i]._direction[0],
                                           1. / photons[i]._direction[1],
                                           1. / photons[i]._direction[2]};
      const int result = grid.interact(photons[i], inverse_direction);
      output[result] += 1.;
    }
    for (unsigned int i = 0; i < 27; ++i) {
      output[i] *= 100. / num_photon;
    }

    std::cout << "Loop " << iloop << ":" << std::endl;
    std::cout << output[TRAVELDIRECTION_CORNER_NPP] << "\t"
              << output[TRAVELDIRECTION_EDGE_X_PP] << "\t"
              << output[TRAVELDIRECTION_CORNER_PPP] << "\n";
    std::cout << output[TRAVELDIRECTION_EDGE_Y_NP] << "\t"
              << output[TRAVELDIRECTION_FACE_Z_P] << "\t"
              << output[TRAVELDIRECTION_EDGE_Y_PP] << "\n";
    std::cout << output[TRAVELDIRECTION_CORNER_NNP] << "\t"
              << output[TRAVELDIRECTION_EDGE_X_NP] << "\t"
              << output[TRAVELDIRECTION_CORNER_PNP] << "\n";
    std::cout << std::endl;
    std::cout << output[TRAVELDIRECTION_EDGE_Z_NP] << "\t"
              << output[TRAVELDIRECTION_FACE_Y_P] << "\t"
              << output[TRAVELDIRECTION_EDGE_Z_PP] << "\n";
    std::cout << output[TRAVELDIRECTION_FACE_X_N] << "\t"
              << output[TRAVELDIRECTION_INSIDE] << "\t"
              << output[TRAVELDIRECTION_FACE_X_P] << "\n";
    std::cout << output[TRAVELDIRECTION_EDGE_Z_NN] << "\t"
              << output[TRAVELDIRECTION_FACE_Y_N] << "\t"
              << output[TRAVELDIRECTION_EDGE_Z_PN] << "\n";
    std::cout << std::endl;
    std::cout << output[TRAVELDIRECTION_CORNER_NPN] << "\t"
              << output[TRAVELDIRECTION_EDGE_X_PN] << "\t"
              << output[TRAVELDIRECTION_CORNER_PPN] << "\n";
    std::cout << output[TRAVELDIRECTION_EDGE_Y_NN] << "\t"
              << output[TRAVELDIRECTION_FACE_Z_N] << "\t"
              << output[TRAVELDIRECTION_EDGE_Y_PN] << "\n";
    std::cout << output[TRAVELDIRECTION_CORNER_NNN] << "\t"
              << output[TRAVELDIRECTION_EDGE_X_NN] << "\t"
              << output[TRAVELDIRECTION_CORNER_PNN] << "\n";

    grid.compute_neutral_fraction(num_photon);
  }

  delete[] photons;

  std::ofstream ofile("intensities.txt");
  grid.print_intensities(ofile);
  ofile.close();

  std::ofstream bfile("intensities.dat");
  grid.output_intensities(bfile);
  bfile.close();

  return 0;
}
