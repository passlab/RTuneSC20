#!/bin/bash

#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=1

export OMP_NUM_THREADS=1

for i in 15 30 45

do
    srun lulesh2.0 -s $i -i 10000
done
