RTUNE_INSTALL=../../rtune-install
LLVM_OPENMP_INSTALL=/opt/llvm/llvm-openmp-rtune
# We only need to provide rtune.h file location. librtune.so will be LD_PRELOAD-ed, thus we do not need to link. 
# rtune.h include omp.h header which contains omp_control_too

default:jacobi-rtune jacobi-clean

jacobi-rtune: jacobi-rtune.c
	clang -fopenmp -I${RTUNE_INSTALL}/include -L${RTUNE_INSTALL}/lib -lrtune $^ -o $@ -lm
	objdump -d jacobi-rtune >jacobi-rtune.objdump

jacobi-clean: jacobi-clean.c
	clang -fopenmp $^ -o $@ -lm
	objdump -d jacobi-clean >jacobi-clean.objdump

clean:
	rm jacobi-rtune jacobi-clean
