## Instructions for reproducing performance results for RTune methods

The results of the experiment presented in the paper were performed using two machines. One  is  a computing  server  that  has  two  Intel  Xeon  E5-2699v3  CPUs for total 36 cores at 2.3 GHz and 256 GB main memory. Ubuntu 18.04.3 LTS with Linux kernel 4.15.0  is  the  OS.  Clang/LLVM  9.0.0  compiler  and  OpenMP runtime  are  used  to  compile  the  programs. Another machine is LLNL Pascal  cluster.  Each  cluster  node  has  two  Intel  Xeon  E5-2695 v4 CPUs for total 36 cores, and 256 GB main memory. Red  Hat  Enterprise  Linux  Server  release  7.7  (Maipo),  kernel 3.10.0,  MVAPICH2  v2.3  and  Intel  19.0.4  compiler  are  the software environment.

## Steps:
1. clone the repository and go to the repository from a Linux terminal

    ```
    git clone https://github.com/passlab/RTuneSC20
    cd RTuneSC20
    ```

### Objective 1: 

1. change to `objective1/LULESH` folder

    ```
    cd objective1/LULESH
    
    ```

### Objective 2:

1. change to `objective2/jacobi` folder
    ```
    cd objective2/jacobi
    
    ```

### Objective 3: 

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
1. execute the `batch_run.sh` script to generate the results for LULESH shown in the paper

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
    
