#!/bin/bash
for i in {2..36..2}
do 
    export OMP_NUM_THREADS=$i
    ./jacobi-clean 64 64 
done

for i in {2..36..2}
do 
    export OMP_NUM_THREADS=$i
    ./jacobi-clean 128 128  
done

for i in {2..36..2}
do 
    export OMP_NUM_THREADS=$i
    ./jacobi-clean 256 256  
done

for i in {2..36..2}
do
    export OMP_NUM_THREADS=$i
    ./jacobi-clean 512 512  
done
