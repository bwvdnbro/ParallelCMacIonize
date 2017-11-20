#include "DensitySubGrid.hpp"
#include "RandomGenerator.hpp"

#include <cmath>
#include <fstream>

int main(int argc, char **argv){

  const double box[6] = {0., 0., 0., 1., 1., 1.};
  const int ncell[3] = {128, 128, 128};
  DensitySubGrid grid(box, ncell);

  const unsigned int num_photon = 1000000;

  Photon *photons = new Photon[num_photon];
  RandomGenerator random_generator;
  for(unsigned int i = 0; i < num_photon; ++i){
    Photon &photon = photons[i];
    photon._position[0] = 0.5;
    photon._position[1] = 0.5;
    photon._position[2] = 0.5;
    const double cost = 2. * random_generator.get_uniform_random_double() - 1.;
    const double sint = std::sqrt(std::max(1. - cost * cost, 0.));
    const double phi = 2. * M_PI * random_generator.get_uniform_random_double();
    const double cosp = std::cos(phi);
    const double sinp = std::sin(phi);
    photon._direction[0] = sint * cosp;
    photon._direction[1] = sint * sinp;
    photon._direction[2] = cost;
    photon._weight = 1.;
    photon._current_optical_depth = 0.;
    photon._target_optical_depth = -std::log(random_generator.get_uniform_random_double());
    photon._photoionization_cross_section = 1.e-10;
  }

  for(unsigned int i = 0; i < num_photon; ++i){
    const double inverse_direction[3] = {1./photons[i]._direction[0], 1./photons[i]._direction[1], 1./photons[i]._direction[2]};
    grid.interact(photons[i], inverse_direction);
  }

  delete [] photons;

  std::ofstream ofile("intensities.txt");
  grid.print_intensities(ofile);
  ofile.close();

  std::ofstream bfile("intensities.dat");
  grid.output_intensities(bfile);
  bfile.close();

  return 0;
}
