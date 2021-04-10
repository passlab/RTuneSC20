# homp-metadirective
Implementing metadirective extension using HOMP 

# Using `metadirective` to guide the computing of 2D stencil kernel

## Prerequisite

- NVIDIA GPU with CUDA toolkit 10.1
- Clang/LLVM 10.x with omp target offloading support

## Files

List
* self-contained all-in-one version:  stencil_metadirective_online.c
* separated adapative code and runtime support files
  * stencil_metadirective_main.c 
  * stencil_metadirective_models.c

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

41 iterations and after will use the trained models to guide the device selection. 

* compiler-translation: prototype of refined compiler translation
 

## Build

```bash
make
```
this will build the all-in-one version
* stencil_online.out

## Run

The program takes a max problem size as parameter. It runs 5000 iterations. The first 40 iterations run with assigned problem sizes for profiling and modelling.
For the rest iterations, random problem size is generated for each iteration becuase in the AMR applications the problem size for each iteration could be different due to refinement and coarsening.

The range of probelm size is from 1 to the given max problem size.

The program prints the generated model functions and the total execution time of CPU, GPU, and three models to `stderr`.
The max problem size and five execution times are printed to `stdout`.

```bash
# ./stencil_online.out <problem_size>
./stencil_online.out 512

Or if you use Livermore Computing's Lassen supercomputer
* bsub ./stencil_online.out 512

Qilin CPU: y = 0.0787223 * x + -15.5737
Log Linear Regression CPU: y = 22.5622 * ln(x) + -89.711
Log-Log Linear Regression CPU: log(y) = 3.04608 * log(x) + -22.3711
Qilin GPU: y = 0.00759507 * x + -0.871385
Log Linear Regression GPU: y = 2.28097 * ln(x) + -8.95434
Log-Log Linear Regression GPU: log(y) = 2.73217 * log(x) + -22.9684
CPU total time (ms): 18777
GPU total time (ms): 3262
Qilin total time (ms): 4436
Log Linear Regression total time (ms): 3514
Log-Log Linear Regression total time (ms): 3604
512, 18777, 3262, 4436, 3514, 3604

```

Anjia's experiment data

https://docs.google.com/spreadsheets/d/18fq2KkIPx-kAIIrWIKFwsEo6XXFwXtg-g0CDeTmJ00g/edit#gid=1568124873 


## Test

A script is used to test different max problem sizes in batch, then we can get an overall performance of three models.
The script takes up to 5 parameters: prefix of output csv file, starting max problem size, ending max problem size, step, and executable postfix.

Be default, if no parameters are provided, it tests from max problem size 32 to 512 with step 32 (32, 64, 96, ..., 512). The output csv file is `online_random.csv`. The executable name is `stencil_online.out`.

```bash
#./test_random_size.sh <output_prefix> <starting_max_problem_size> <ending_max_problem_size> <step> <executable_postfix>
./test_random_size.sh
```
