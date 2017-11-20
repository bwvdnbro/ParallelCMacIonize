#include "DensitySubGrid.hpp"
#include "RandomGenerator.hpp"

#include <cmath>
#include <fstream>

int main(int argc, char **argv){

  const double box[6] = {0., 0., 0., 1., 1., 1.};
  const int ncell[3] = {128, 128, 128};
  DensitySubGrid grid(box, ncell);

  RandomGenerator random_generator;
  for(unsigned int i = 0; i < 1000000; ++i){
    Photon photon;
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
    const double inverse_direction[3] = {1./photon._direction[0], 1./photon._direction[1], 1./photon._direction[2]};
    photon._weight = 1.;
    photon._current_optical_depth = 0.;
    photon._target_optical_depth = -std::log(random_generator.get_uniform_random_double());
    photon._photoionization_cross_section = 1.e-10;
    grid.interact(photon, inverse_direction);
  }

  std::ofstream ofile("intensities.txt");
  grid.print_intensities(ofile);
  ofile.close();

  return 0;
}
