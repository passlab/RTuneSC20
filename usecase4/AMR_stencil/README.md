
# Automatically select CPU or GPU based on the problem size of each iteration of AMR (2D stencil kernel)

## Prerequisite

- NVIDIA GPU with CUDA toolkit 10.1 or later
- Clang/LLVM 10.x or later with omp target offloading support

## Files

List
* self-contained all-in-one version:  `amr_stencil_rtune.c`
* Testing script: `test_random_size.sh`

the main function will run a stencil kernel 5000 times with randomized stencil sizes. 
* first 20+20 iterations will be used to profile CPU and GPU executions
  * 20 iterations (data points) are used for each type of devices. 
* using a set of predefined problem sizes to sample exeuction times of CPU vs. GPU loops. 
```
    int profiling_size[PROFILE_NUM] = {
        20, 40, 60, 80, 100, 120, 140, 160, 180, 200,
        300, 400, 500, 600, 700, 800, 900, 1000, 1500, 2000,
    };
```    

The 41-th and followed iterations will use the trained models to guide the device selection. 
 

## Build

```bash
make
```
this will build the all-in-one version
* amr_stencil_rtune.out

## Run

The program takes a max problem size and the execution mode as parameters. It runs 5000 iterations. The first 40 iterations run with assigned problem sizes for profiling and modelling.
For the rest iterations, random problem size is generated for each iteration becuase in the AMR applications the problem size for each iteration could be different due to refinement and coarsening.

The range of probelm size is from 1 to the given max problem size.

The program prints the generated model functions and the total execution time of CPU, GPU, and linear regression model to `stderr`.
The max problem size and three execution times are printed to `stdout`.

Three modes can be selected:
1. Default mode: the AMR application runs with RTune to obtain the best performance;
2. Overhead mearsuring mode: the AMR application runs with RTune. The total time cost and overhead are measured;
3. Evaluation mode: besides running with RTune, the AMR application also runs on purely CPU or GPU. The exectuion time of three versions will be printed.

```bash
# ./amr_stencil_rtune.out <problem_size> <execution_mode>
./amr_stencil_rtune.out 512 2
512
Linear Regression CPU: y = 0.0450573 * x + -8.92249
Linear Regression GPU: y = 0.0187996 * x + -3.46672
CPU total time (ms): 10726.4
GPU total time (ms): 5863.13
Linear Regression total time (ms): 6071.73
```


## Test

A script is used to test different max problem sizes in batch, then we can get an overall performance of three models.
The script takes up to 5 parameters: prefix of output csv file, starting max problem size, ending max problem size, step, and executable postfix.

Be default, if no parameters are provided, it tests from max problem size 32 to 512 with step 32 (32, 64, 96, ..., 512). The output csv file is `rtune_random.csv`. The executable name is `amr_stencil_rtune.out`.

```bash
#./test_random_size.sh <output_prefix> <starting_max_problem_size> <ending_max_problem_size> <step> <executable_postfix>
./test_random_size.sh
```
