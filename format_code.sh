#! /bin/bash

for f in *.[ch]pp
do
  clang-format-6.0 -i $f
done
