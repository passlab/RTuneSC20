# ActiveHarmony-enhanced version of jacobi for tuning thread count

## Prerequisite

- Clang/LLVM 10.x or later with OpenMP support
- Active Harmony 4.6

## Files

List

* jacobi-ah.c - ActiveHarmony-enhanced version of jacobi

The problem size are the inputs of the excutable. With different number of problem sizes, thread counts are tuned according to the execution time of one run of loop body.
The search space for thread counts is from 4 to 32 with stride of 2.
As for advanced configuration of ActiveHarmony, the search strategy is Nelder-Mead method and there are no aggregation layers.

## Build

```bash
make
```
By default, this will build ActiveHarmony-enhanced version of jacobi

* jacobi-ah

## Run

The program is executed with four different problem sizes, 64x64, 128x128, 256x256, and 512x512.
The tuning results are reflected by total execution time.

For running all tests:

```bash
./jacobi-ah-run.sh
```
Or, for customized test:

```bash
#./jacobi-ah <n> <m>
./jacobi-ah 256 256
```
One sample output is:

```bash
OpenMP (10 threads) elapsed time(ms):        695.1
MFLOPS:         6033
Solution Error: 0.00207517
```