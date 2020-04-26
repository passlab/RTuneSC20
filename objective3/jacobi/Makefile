RTUNE_INSTALL=../rtune-install

default:jacobi-rtune jacobi-clean

jacobi-rtune: jacobi-rtune.c
	clang -fopenmp -I${RTUNE_INSTALL}/include -L${RTUNE_INSTALL}/lib -lrtune $^ -o $@ -lm

jacobi-clean: jacobi-clean.c
	clang -fopenmp $^ -o $@ -lm

clean:
	rm jacobi-rtune jacobi-clean
