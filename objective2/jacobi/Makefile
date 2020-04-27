
jacobi: jacobi.c
	clang -v -g $^ -o $@ -lm
	objdump -d jacobi >jacobi.objdump

run: jacobi
	./jacobi 1024 1024

mpi: jacobi_mpi.c
	mpicc $^ -lm -o jacobi_mpi

clean:
	rm jacobi jacobi_mpi
