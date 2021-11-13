#!/bin/bash
make clean
make

srun -c6 -n2 ./hw2b experiment.png 10000 -0.5 0.5 -0.5 0.5 491 935
make clean

rm experiment.png