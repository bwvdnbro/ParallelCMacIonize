#ifndef PHOTON_HPP
#define PHOTON_HPP

class Photon {
public:
  double _position[3];
  double _direction[3];
  double _current_optical_depth;
  double _target_optical_depth;
  double _photoionization_cross_section;
  double _weight;
};

#endif // PHOTON_HPP
