#!/bin/bash
#SBATCH -c 4
#SBATCH -n 1
rm ./imgs/fast01.png
make clean
make
srun ./hw2a ./imgs/fast01.png 10000 -2 2 -2 2 1000 1000