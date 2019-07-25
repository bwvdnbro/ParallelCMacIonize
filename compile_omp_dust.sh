#! /bin/bash

debug=

while [ "$1" != "" ]
do
  case $1 in
    -d | --debug ) debug=1;;
  esac
  shift
done

# flags that need to be present
cflags="-std=c++11 -fopenmp -march=native"

# Extra flags: choose one of the two below
flags="-Wall -Werror -Wignored-qualifiers"
if [ "$debug" = "1" ]
then
  # optional (debug) flags
  flags="$flags -g -fsanitize=address -fno-omit-frame-pointer"
  echo "Compiling with debug flags"
else
  # default (optimization) flags
  flags="$flags -O3 -ftree-vectorize -funroll-loops -ffast-math"
fi

# library dependencies
incs="-I/usr/include/hdf5/serial"
libs="-L/usr/lib/x86_64-linux-gnu/hdf5/serial -lhdf5"

for f in testDensitySubGrid_omp_dust.cpp CommandLineOption.cpp CommandLineParser.cpp
do
  g++ $flags $cflags $incs -c $f
done

g++ $flags $cflags -o testDensitySubGrid_omp_dust \
  testDensitySubGrid_omp_dust.o CommandLineOption.o CommandLineParser.o $libs
