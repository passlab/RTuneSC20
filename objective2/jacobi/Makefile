#RTUNE_INSTALL=../../rtune-install
RTUNE_INSTALL=${HOME}/Projects/rtune/rtune-install
LLVM_OPENMP_INSTALL=/home/awang15/Projects/rtune/llvm-openmp-install
# We only need to provide rtune.h file location. librtune.so will be LD_PRELOAD-ed, thus we do not need to link. 
# rtune.h include omp.h header which contains omp_control_too

jacobi: jacobi.c
	#clang -v -fopenmp -I${LLVM_OPENMP_INSTALL}/include -I${RTUNE_INSTALL}/include -g $^ -o $@ -lm
	clang -v ${HOME}/Projects/rtune/llvm-openmp-install/lib/libomp.so -I${LLVM_OPENMP_INSTALL}/include -I${HOME}/Projects/rtune/rtune-install/include -g $^ -o $@ -lm
	objdump -d jacobi >jacobi.objdump

run: jacobi
	LD_PRELOAD=${LLVM_OPENMP_INSTALL}/lib/libomp.so:${RTUNE_INSTALL}/lib/librtune.so ./jacobi 2048 2048

mpi: jacobi_mpi.c
	mpicc $^ -lm -o jacobi_mpi

clean:
	rm jacobi jacobi_mpi
