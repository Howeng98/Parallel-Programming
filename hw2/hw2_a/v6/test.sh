#!/bin/bash

mpicxx -lm -O3 -lpng test.cpp -o test
srun -n1 ./test
rm test