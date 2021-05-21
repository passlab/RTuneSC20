
# Tuning for the tradeoff between computation and precision (Jacobi)

## Prerequisite

- OpenMPI 4.0
- Clang/LLVM 10.x or later

## Files

List
* Jacobi MPI version:  `jacobi_mpi.c`

Given a predefined percision and maximum number of iterations, RTune will collect a sample every 5 iterations. It tracks the gradient of last two samples. If the gradient is smaller than a threshold, it increases the counter by one.
If there are 15 such recorded gradients within every 200 iterations, RTune will consider that the converging is almost done and the computing can be terminated without losing too much percision.
The code has been tested on the Pascal cluster of LLNL.

## Build

```bash
make
```
By default, this will build the Jacobi MPI version
* jacobi_mpi.out

## Run

The program takes the problem size and number of MPI processors as parameters. It runs 50000 iterations.

The program prints the total execution time, tuning time, and the predicted number of saved iterations.

```bash
# mpirun -np <number_MPI_processor> ./jacobi_mpi.out <problem_size> <problem_size>
mpirun -np 4 ./jacobi_mpi.out 1024 1024
...
Saved iterations based on mits: 37420 out of 50000
Fitting curve is: y = -5.41254e-06 * x + -8.17812
Predicted finishing iteration is: 336603
Saved iterations based on error tolerance: 324023 out of 336603
Computing Time: 57280, Tuning Time: 0.52026, Total Time: 57280
Total Number of Iterations: 12580
Residual: 5.67204468968116e-09
...
```
