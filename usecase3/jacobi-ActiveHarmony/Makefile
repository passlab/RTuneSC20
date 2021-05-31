AH_INSTALL=${HOME}/harmony

default:jacobi-ah

jacobi-ah: jacobi-ah.c
	g++ $^ -fopenmp -I${AH_INSTALL}/src ${AH_INSTALL}/src/libharmony.a -o $@ -lm

clean:
	rm jacobi-ah
