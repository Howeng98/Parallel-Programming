#!/bin/bash
#SBATCH -n 10
#SBATCH -N 2
make clean
make
srun ./hw1 65536 06.in 06.out