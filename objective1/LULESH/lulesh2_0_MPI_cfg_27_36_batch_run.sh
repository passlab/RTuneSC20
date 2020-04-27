#!/bin/bash
  
#SBATCH --ntasks=27
#SBATCH --ntasks-per-node=1

export OMP_NUM_THREADS=36

for i in 10 20 30

do
    srun lulesh2.0 -s $i -i 10000
done
