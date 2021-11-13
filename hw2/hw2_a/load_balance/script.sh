#!/bin/bash
#SBATCH -c 8
#SBATCH -n 1
make clean
make

#srun -c12 ./hw2a experiment.png 10000 -0.5 0.5 -0.5 0.5 491 935
srun -c12 ./hw2a experiment.png 54564 -0.34499 -0.34501 -0.61249 -0.61251 800 800
make clean

rm experiment.png