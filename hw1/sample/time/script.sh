#!/bin/bash
#SBATCH -n 12
#SBATCH -N 1
make clean
make
srun ./hw1 536869888 33.in 33.out
make clean