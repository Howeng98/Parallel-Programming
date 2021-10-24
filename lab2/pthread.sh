#!/bin/bash
#SBATCH -c 4
#SBATCH -n 1
make clean
make
srun ./lab2_pthread 5 21  