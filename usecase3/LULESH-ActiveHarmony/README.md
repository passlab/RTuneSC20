# ActiveHarmony-enhanced version of LULESH for tuning thread count

## Prerequisite

- OpenMPI 4.0
- Clang/LLVM 10.x or later with OpenMP support
- Active Harmony 4.6

## Files

List

* lulesh.cc - where most of the timed functionality lies including ActiveHarmony functions
* lulesh-comm.cc - MPI functionality
* lulesh-init.cc - Setup code
* lulesh-viz.cc  - Support for visualization option
* lulesh-util.cc - Non-timed functions

The problem size are the inputs of the excutable. With different number of problem sizes, thread counts are tuned according to the execution time of one run of loop body.
The search space for thread counts is from 4 to 32 with stride of 2.
As for advanced configuration of ActiveHarmony, the search strategy is Nelder-Mead method and there are no aggregation layers.

## Build

```bash
make
```
By default, this will build ActiveHarmony-enhanced version of LULESH

* lulesh2.0-ah

## Run

The program is executed with three different problem sizes, 20x20x20, 30x30x30, and 40x40x40.
The tuning results are reflected by total execution time.

For running all tests:

```bash
./lulesh-ah-run.sh
```
Or, for customized test:

```bash
#./lulesh2.0-ah -s <problem_size>
./lulesh2.0-ah -s 20
```
One sample output is:

```bash
Run completed:
   Problem size        =  20
   MPI tasks           =  1
   Iteration count     =  575
   Final Origin Energy =  9.668856e+04
   Testing Plane 0 of Energy Array on rank 0:
        MaxAbsDiff   = 2.910383e-11
        TotalAbsDiff = 2.310265e-10
        MaxRelDiff   = 1.748265e-12

Elapsed time         =        2.7 (s)
Grind time (us/z/c)  = 0.59525203 (per dom)  ( 2.7381593 overall)
FOM                  =  1679.9607 (z/s)
```