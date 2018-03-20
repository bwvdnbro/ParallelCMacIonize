#! /bin/bash

for f in *.[ch]pp
do
  clang-format-3.8 -i $f
done
