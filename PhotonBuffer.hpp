#ifndef PHOTONBUFFER_HPP
#define PHOTONBUFFER_HPP

#include "Photon.hpp"

#define PHOTONBUFFER_SIZE 1000u

class PhotonBuffer {
public:
  int _direction;
  unsigned int _actual_size;
  Photon _photons[PHOTONBUFFER_SIZE];
  bool _is_inside_box;
};

#endif // PHOTONBUFFER_HPP
