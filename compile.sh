#! /bin/bash

# flags that need to be present
cflags="-std=c++11 -fopenmp"

# Extra flags: choose one of the two below
# optional (optimization) flags
#flags="-Wall -Werror -O3 -ftree-vectorize -funroll-loops -ffast-math"
# optional (debug) flags
flags="-Wall -Werror -g -fsanitize=address -fno-omit-frame-pointer"

# library dependencies
libs="-lparmetis -lmetis"

for f in testDensitySubGrid.cpp CommandLineOption.cpp CommandLineParser.cpp
do
  mpic++ $flags $cflags -c $f
done

mpic++ $flags $cflags -o testDensitySubGrid \
       testDensitySubGrid.o CommandLineOption.o CommandLineParser.o $libs
