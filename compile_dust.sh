#! /bin/bash

mpic++ -std=c++11 -Wall -Werror -O3 -march=native -ffast-math -funroll-loops \
  -ftree-vectorize -fopenmp -o testDensitySubGrid_dust \
  testDensitySubGrid_dust.cpp
