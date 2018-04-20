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
cflags="-std=c++11 -fopenmp"

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
libs="-lmetis"

for f in testDensitySubGrid.cpp CommandLineOption.cpp CommandLineParser.cpp
do
  mpic++ $flags $cflags -c $f
done

mpic++ $flags $cflags -o testDensitySubGrid \
       testDensitySubGrid.o CommandLineOption.o CommandLineParser.o $libs
