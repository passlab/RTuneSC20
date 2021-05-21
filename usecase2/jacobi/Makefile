default: jacobi_mpi.out

jacobi.out: jacobi.c
	clang -v -g $^ -o $@ -lm

run: jacobi_mpi.out
	mpirun -np 4 ./jacobi_mpi.out 1024 1024

jacobi_mpi.out: jacobi_mpi.c
	mpicc $^ -lm -o $@

clean:
	rm *.out
