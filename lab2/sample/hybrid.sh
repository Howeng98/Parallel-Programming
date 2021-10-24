#!/bin/bash
#SBATCH -c 4
#SBATCH -n 6
#SBATCH -N 2
make clean
make
srun ./lab2_hybrid 5 21