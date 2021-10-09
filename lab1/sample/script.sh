#!/bin/bash
#SBATCH -n 1
#SBATCH -N 1
make clear
make
srun ./lab1 2147483647 2147483647