## Instructions for reproducing performance results for RTune methods

The results of the experiment presented in the paper were performed using two machines. One  is  a computing  server  that  has  two  Intel  Xeon  E5-2699v3  CPUs for total 36 cores at 2.3 GHz and 256 GB main memory. Ubuntu 18.04.3 LTS with Linux kernel 4.15.0  is  the  OS.  Clang/LLVM  9.0.0  compiler  and  OpenMP runtime  are  used  to  compile  the  programs. Another machine is LLNL Pascal  cluster.  Each  cluster  node  has  two  Intel  Xeon  E5-2695 v4 CPUs for total 36 cores, and 256 GB main memory. Red  Hat  Enterprise  Linux  Server  release  7.7  (Maipo),  kernel 3.10.0,  MVAPICH2  v2.3  and  Intel  19.0.4  compiler  are  the software environment.

Thus a standard Linux machine with OpenMP and MPI support (OpenMP compiler such as GNU or Clang/LLVM compiler)
and MPI library (MPICH, OpenMPI, etc) should be sufficient to run the program. 

## Steps:
1. clone the repository and go to the repository from a Linux terminal

    ```
    git clone https://github.com/passlab/RTuneSC21
    cd RTuneSC21
    ```

### Use case 1: 

1. change to `objective1/LULESH` folder

    ```
    cd objective1/LULESH
    mkdir build
    ```

2. change to `build` folder, build executable `lulesh2.0`
    ```
    cd build
    cmake -DCMAKE_BUILD_TYPE=Release -DMPI_CXX_COMPILER=`which mpicxx` ..
    make
    ```
3. change to `objective1/LULESH` folder, run the .sh scripts to get the results shown in the paper
    ```
    ./lulesh2_0_cutoff_cfg_batch_run.sh
    ./lulesh2_0_MPI_cfg_8_1_batch_run.sh
    ...
    ```

### Use case 2:

1. change to `objective2/jacobi` folder
    ```
    cd objective2/jacobi
    
    ```
2. build the MPI executable `jacobi_mpi`
    ```
    make mpi
    ````
3. run the produced executable with different arguments on clusters
    ```
    # run with the RTune threshold 0.0002 and defualt error tolerance 1.0*10-9.
    mpirun -N2 -n8 ./jacobi_mpi 1024 1024 0.0002
    # run with the RTune threshold 0.0005 and error tolerance 4.0*10-9
    mpirun -N2 -n8 ./jacobi_mpi 1024 1024 0.0005 0.000000004
    ...
    ```

### Use case 3: 

1. change to `objective3` folder and create build and install folder for rtune library
    ```
    cd objective3
    mkdir rtune-build rtune-install
    ```
1. change to `rtune-build` folder, build rtune library and install the library in the `rtune-install` folder
    ```
    cd rtune-build
    cmake -DCMAKE_INSTALL_PREFIX=../rtune-install ../rtune 
    make
    make install

    ```
1. change to `objective3/LULESH` folder, and create both the clean LULESH binary (`lulesh2.0-clean`) and RTune-optimized LULESH binary (`lulesh2.0-rtune`)

    ```
    make -f Makefile-rtune 
    make -f Makefile-clean
    ```
1. execute the `clean_batch_run.sh` and `rtune_batch_run.sh` scripts to generate the results for LULESH shown in the paper

    ```
    ./batch_run.sh
    ```
1. change to `objective3/jacobi` folder, and create both the clean jacobi binary (`jacobi-clean`) and RTune-optimized jacobi binary (`jacobi-rtune`)
    ```
    make
    ```
1. execute the `clean_batch_run.sh` and `rtune_batch_run.sh` scripts to generate the jacobi results shown in the paper
    ```
    ./clean_batch_run.sh
    ./rtune_batch_run.sh
    ```
    
### Use case 4:
