#!/bin/bash
#SBATCH -c 4
#SBATCH -n 1
make clean
make
srun ./hw2a ./imgs/fast01.png 2602 -3 0.2 -3 0.2 979 2355