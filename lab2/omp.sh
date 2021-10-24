#!/bin/bash
#SBATCH -c 4
#SBATCH -n 1
make clean
make
srun ./lab2_omp 5 100 