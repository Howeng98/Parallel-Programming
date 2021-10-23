#!/bin/bash
#SBATCH -n 15
#SBATCH -N 3
make clean
make
srun ./hw1 15 02.in 02.out