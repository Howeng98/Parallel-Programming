#!/bin/bash
#SBATCH -n 1
#SBATCH -N 1
make clean
make
srun ./hw1 50 04.in 04.out