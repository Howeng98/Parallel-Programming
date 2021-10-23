#!/bin/bash
#SBATCH -n 24
#SBATCH -N 2
make clean
make
srun ./hw1 12347 21.in 21.out