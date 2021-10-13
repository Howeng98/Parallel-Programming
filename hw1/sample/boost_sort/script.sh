#!/bin/bash
#SBATCH -n 28
#SBATCH -N 4
make clean
make
srun ./hw1 21 03.in 03.out