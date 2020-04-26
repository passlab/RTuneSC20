#!/bin/bash
for i in {1..10..1}
do 
    export OMP_NUM_THREADS=36
    ./jacobi-rtune 64 64 
done

for i in {1..10..1}
do
    export OMP_NUM_THREADS=36
    ./jacobi-rtune 128 128
done

for i in {1..10..1}
do
    export OMP_NUM_THREADS=36
    ./jacobi-rtune 256 256
done

for i in {1..10..1}
do
    export OMP_NUM_THREADS=36
    ./jacobi-rtune 512 512
done
