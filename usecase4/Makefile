default: stencil_online.out

all: stencil_online.out

clean: 
	rm -rf *.o *.log *.out

stencil_online.out: stencil_metadirective_online.c
	clang -fopenmp -fopenmp-targets=nvptx64 -lm stencil_metadirective_online.c -o stencil_online.out

