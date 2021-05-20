#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <omp.h>
#include <sys/time.h>
#include <float.h>

#define REAL float
#define FILTER_HEIGHT 5
#define FILTER_WIDTH 5
#define TEST 5000
#define PROBLEM 256
#define PROBLEM_SIZE 768
#define TEAM_SIZE 128
#define PROFILE_NUM 20

#define MAX_ITER 5000

// clang -fopenmp -fopenmp-targets=nvptx64 -lm stencil_rtune.c -o stencil.out
// Usage: ./stencil.out <size>
// e.g. ./stencil.out 512

void stencil_omp_cpu(const REAL* src, REAL* dst, int width, int height, const float* filter, int flt_width, int flt_height);
void stencil_omp_gpu(const REAL* src, REAL* dst, int width, int height, const float* filter, int flt_width, int flt_height);
void linear_regression(const REAL* src, REAL* dst, int width, int height, const float* filter, int flt_width, int flt_height);

// variables for linear regression model
int linear_regression_cpu_test = 0;
int linear_regression_gpu_test = 0;
double linear_regression_a_cpu, linear_regression_b_cpu, linear_regression_a_gpu, linear_regression_b_gpu;
int linear_regression_profiling_problem_size[PROFILE_NUM];
double linear_regression_cpu_data[PROFILE_NUM];
double linear_regression_gpu_data[PROFILE_NUM];


static double read_timer_ms() {
    struct timeval timer;
    gettimeofday(&timer, NULL);
    return (double)timer.tv_sec * 1000.0 + (double)timer.tv_usec / 1000.0;
}

void print_array(char *title, char* name, double *A, int n) {
    printf("%s:\n", title);
    int i, j;
    for (i = 0; i < n; i++) {
        printf("%s[%d]:%lg, ", name, i, A[i]);
    }
    printf("\n");
}

void initialize(int width, int height, REAL *u) {
    int i;
    int N = width*height;

    for (i = 0; i < N; i++)
        u[i] = rand() % 256;
}

void initialize_problem_size(int iteration_number, int max_problem_size, int* problem_size) {
    int i;
    for (i = 0; i < iteration_number; i++)
        problem_size[i] = rand() % max_problem_size;
}

void fit_linear_regression(int* problem_size, double* execution_time, int amount, double* result) {
    double sumx = 0.0, sumy = 0.0, sumxy = 0.0, sumx2 = 0.0;
    double a, b;
    int i;

    for (i = 0; i < amount; i++) {
        sumx = sumx + problem_size[i];
        sumx2 = sumx2 + problem_size[i] * problem_size[i];
        sumy = sumy + execution_time[i];
        sumxy = sumxy + problem_size[i] * execution_time[i];
    }
    a = ((amount * sumxy - sumx*sumy) * 1.0 /(amount * sumx2-sumx * sumx) * 1.0);
    b = ((sumx2 * sumy - sumx*sumxy) * 1.0 / (amount * sumx2-sumx * sumx) * 1.0);
    result[0] = a;
    result[1] = b;
}

int main(int argc, char *argv[]) {
    int n = PROBLEM;
    int m = PROBLEM;
    int iteration_number = TEST;
    int i, j, k;

    if (argc == 2) {
        n = atoi(argv[1]);
        m = atoi(argv[1]);
    };

    if (m < 2000) {
        m = 2000;
        n = 2000;
    };

    REAL *u = (REAL *) malloc(sizeof(REAL) * n * m);
    REAL *result = (REAL *) malloc(sizeof(REAL) * n * m);
    REAL *result_cpu = (REAL *) malloc(sizeof(REAL) * n * m);
    REAL *result_linear_regression = (REAL *) malloc(sizeof(REAL) * n * m);

    initialize(n, m, u);

    if (argc == 2) {
        n = atoi(argv[1]);
        m = atoi(argv[1]);
    }
    else {
        n = PROBLEM;
        m = PROBLEM;
    };

    const float filter[FILTER_HEIGHT][FILTER_WIDTH] = {
        { 0, 0, 1, 0, 0, },
        { 0, 0, 2, 0, 0, },
        { 3, 4, 5, 6, 7, },
        { 0, 0, 8, 0, 0, },
        { 0, 0, 9, 0, 0, },
    };

    int width = m;
    int height = n;

    // In the actual application, users should make sure the initial 40 (20 for CPU, 20 for GPU) iterations cover representative problem sizes.
    // Otherwise, the model may not be accurate enough.
    int profiling_size[PROFILE_NUM] = {
        20, 40, 60, 80, 100, 120, 140, 160, 180, 200,
        300, 400, 500, 600, 700, 800, 900, 1000, 1500, 2000,
    };

    // initialize the testing problem sizes
    int* problem_size = (int*) malloc(sizeof(int) * iteration_number);
    initialize_problem_size(iteration_number, width, problem_size);

    // fill the first 40 iters with profiling problem sizes.
    if (iteration_number > PROFILE_NUM * 2) {
        memcpy(problem_size, profiling_size, sizeof(int)*20);
        memcpy(problem_size+20, profiling_size, sizeof(int)*20);
    };

    /*
    printf("Problem size: ");
    for (int i = 0; i < iteration_number; i++) {
        printf("%d, ", problem_size[i]);
    };
    printf("\n");
    */

    // warm up the functions
    for (i = 0; i < 8; i++) {
        stencil_omp_gpu(u, result, width, height, filter[0], FILTER_WIDTH, FILTER_HEIGHT);
        stencil_omp_cpu(u, result_cpu, width, height, filter[0], FILTER_WIDTH, FILTER_HEIGHT);
        linear_regression(u, result_linear_regression, width, height, filter[0], FILTER_WIDTH, FILTER_HEIGHT);
    };

    // Reset the models after warming up
    linear_regression_cpu_test = 0;
    linear_regression_gpu_test = 0;

    double elapsed = read_timer_ms();
    double cpu_time = 0.0;
    double gpu_time = 0.0;
    double linear_regression_time = 0.0;
    double dif = 0.0;

    for (k = 0; k < iteration_number; k++) {
        int width2 = problem_size[k];
        int height2 = problem_size[k];

        elapsed = read_timer_ms();
        stencil_omp_gpu(u, result, width2, height2, filter[0], FILTER_WIDTH, FILTER_HEIGHT);
        gpu_time += read_timer_ms() - elapsed;
        elapsed = read_timer_ms();
        stencil_omp_cpu(u, result_cpu, width2, height2, filter[0], FILTER_WIDTH, FILTER_HEIGHT);
        cpu_time += read_timer_ms() - elapsed;
        elapsed = read_timer_ms();
        linear_regression(u, result_linear_regression, width2, height2, filter[0], FILTER_WIDTH, FILTER_HEIGHT);
        linear_regression_time += read_timer_ms() - elapsed;

        dif = 0.0;
        for (int j = 0; j < width2*height2; j++) {
            int x = j % width2;
            int y = j / width2;
            if (x > FILTER_WIDTH/2 && x < width2 - FILTER_WIDTH/2 && y > FILTER_HEIGHT/2 && y < height2 - FILTER_HEIGHT/2) {
                dif += fabs(result_linear_regression[j] - result_cpu[j]);
            };
            if (dif != 0.0) {
                printf("verify dif =%g\n", dif);
            };
        };
    };

    fprintf(stderr, "CPU total time (ms): %g\n", cpu_time);
    fprintf(stderr, "GPU total time (ms): %g\n", gpu_time);
    fprintf(stderr, "Linear Regression total time (ms): %g\n", linear_regression_time);
    printf("%d, %lg, %lg, %lg\n", m, cpu_time, gpu_time, linear_regression_time);

    free(u);
    free(result);
    free(result_cpu);
    free(result_linear_regression);

    return 0;
}


void stencil_omp_gpu(const REAL* src, REAL* dst, int width, int height, const float* filter, int flt_width, int flt_height) {

    int flt_size = flt_width*flt_height;
    int N = width*height;
    int BLOCK_SIZE = 128;

#pragma omp target teams distribute parallel for map(to: src[0:N], filter[0:flt_size]) map(from: dst[0:N]) collapse(2)
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            REAL sum = 0;
            for (int n = 0; n < flt_width; n++) {
                for (int m = 0; m < flt_height; m++) {
                    int x = j + n - flt_width / 2;
                    int y = i + m - flt_height / 2;
                    if (x >= 0 && x < width && y >= 0 && y < height) {
                        int idx = m*flt_width + n;
                        sum += src[y*width + x] * filter[idx];
                    }
                }
            }
            dst[i*width + j] = sum;
        }
    }
}


void stencil_omp_cpu(const REAL* src, REAL* dst, int width, int height, const float* filter, int flt_width, int flt_height) {

#pragma omp parallel for collapse(2) num_threads(8)
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            REAL sum = 0;
            for (int n = 0; n < flt_width; n++) {
                for (int m = 0; m < flt_height; m++) {
                    int x = j + n - flt_width / 2;
                    int y = i + m - flt_height / 2;
                    if (x >= 0 && x < width && y >= 0 && y < height) {
                        int idx = m*flt_width + n;
                        sum += src[y*width + x] * filter[idx];
                    }
                }
            }
            dst[i*width + j] = sum;
        }
    }
}



void linear_regression(const REAL* src, REAL* dst, int width, int height, const float* filter, int flt_width, int flt_height) {
    double elapsed = read_timer_ms();
    double cpu_predicted_time = 0.0;
    double gpu_predicted_time = 0.0;
    double result[2];

    if (linear_regression_cpu_test < PROFILE_NUM) {
        elapsed = read_timer_ms();
        stencil_omp_cpu(src, dst, width, height, filter, FILTER_WIDTH, FILTER_HEIGHT);
        elapsed = read_timer_ms() - elapsed;
        // if the execution time is too short to measure, it is set to 0.001ms.
        if (elapsed == 0) elapsed = 0.001;
        linear_regression_profiling_problem_size[linear_regression_cpu_test] = width;
        linear_regression_cpu_data[linear_regression_cpu_test] = elapsed;
        linear_regression_cpu_test += 1;
        if (linear_regression_cpu_test == PROFILE_NUM) {
            fit_linear_regression(linear_regression_profiling_problem_size, linear_regression_cpu_data, 20, result);
            linear_regression_a_cpu = result[0];
            linear_regression_b_cpu = result[1];
            //print_array("FPM", "fc", log_linear_regression_cpu_data, 20);
            fprintf(stderr, "Linear Regression CPU: y = %lg * x + %lg\n", linear_regression_a_cpu, linear_regression_b_cpu);
        }
    } else if (linear_regression_gpu_test < PROFILE_NUM) {
        elapsed = read_timer_ms();
        stencil_omp_gpu(src, dst, width, height, filter, FILTER_WIDTH, FILTER_HEIGHT);
        elapsed = read_timer_ms() - elapsed;
        // if the execution time is too short to measure, it is set to 0.001ms.
        if (elapsed == 0) elapsed = 0.001;
        linear_regression_gpu_data[linear_regression_gpu_test] = elapsed;
        linear_regression_gpu_test += 1;
        if (linear_regression_gpu_test == PROFILE_NUM) {
            fit_linear_regression(linear_regression_profiling_problem_size, linear_regression_gpu_data, 20, result);
            linear_regression_a_gpu = result[0];
            linear_regression_b_gpu = result[1];
            //print_array("FPM", "fg", log_linear_regression_gpu_data, 20);
            fprintf(stderr, "Linear Regression GPU: y = %lg * x + %lg\n", linear_regression_a_gpu, linear_regression_b_gpu);
        }
    } else {
        cpu_predicted_time = linear_regression_a_cpu * width + linear_regression_b_cpu;
        gpu_predicted_time = linear_regression_a_gpu * width + linear_regression_b_gpu;
        if (cpu_predicted_time < gpu_predicted_time) {
            //printf("FPM: %d, use CPU\n", width);
            stencil_omp_cpu(src, dst, width, height, filter, FILTER_WIDTH, FILTER_HEIGHT);
        }
        else {
            //printf("FPM: %d, use GPU\n", width);
            stencil_omp_gpu(src, dst, width, height, filter, FILTER_WIDTH, FILTER_HEIGHT);
        };
    };
}

