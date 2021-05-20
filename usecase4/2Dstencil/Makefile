default: stencil.out

all: stencil_online.out stencil.out

clean: 
	rm -rf *.o *.log *.out

stencil.out: stencil_rtune.c
	clang -fopenmp -fopenmp-targets=nvptx64 -lm stencil_rtune.c -o stencil.out

stencil_online.out: stencil_metadirective_online.c
	clang -fopenmp -fopenmp-targets=nvptx64 -lm stencil_metadirective_online.c -o stencil_online.out

