#!/bin/bash
#SBATCH -c 4
#SBATCH -n 1
rm ./imgs/fast01.png
make clean
make

echo -e "\ntestcase = fast01"
srun ./hw2a ./imgs/fast01.png 2602 -3 0.2 -3 0.2 979 2355
hw2-diff ./imgs/fast01.png ../../testcases/fast01.png  

echo -e "\ntestcase = fast04"
srun ./hw2a ./imgs/fast04.png 1813 -2 0 -2 0 1347 1651
hw2-diff ./imgs/fast04.png ../../testcases/fast04.png

echo -e "\ntestcase = fast08"
srun ./hw2a ./imgs/fast08.png 1274 -1.55555 -1.55515 -0.0002 0.0002 1731 1731
hw2-diff ./imgs/fast08.png ../../testcases/fast08.png
