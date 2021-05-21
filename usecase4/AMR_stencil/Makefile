default: amr_stencil_rtune.out

all: amr_stencil_rtune.out

clean: 
	rm -rf *.o *.log *.out

amr_stencil_rtune.out: amr_stencil_rtune.c
	clang -fopenmp -fopenmp-targets=nvptx64 -lm $^ -o $@

