#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <omp.h>
#include <sys/timeb.h>
#include <float.h>

#define REAL float
#define FILTER_HEIGHT 5
#define FILTER_WIDTH 5
#define TEST 5000
#define PROBLEM 256
#define PROBLEM_SIZE 768
#define TEAM_SIZE 128
#define PROFILE_NUM 20

// clang -fopenmp -fopenmp-targets=nvptx64 -lm stencil_metadirective_online.c -o stencil.out
// Usage: ./stencil.out <size>
// e.g. ./stencil.out 512

void stencil_omp_cpu(const REAL* src, REAL* dst, int width, int height, const float* filter, int flt_width, int flt_height);
void stencil_omp_gpu(const REAL* src, REAL* dst, int width, int height, const float* filter, int flt_width, int flt_height);
void qilin(const REAL* src, REAL* dst, int width, int height, const float* filter, int flt_width, int flt_height);
void log_linear_regression(const REAL* src, REAL* dst, int width, int height, const float* filter, int flt_width, int flt_height);
void log_log_linear_regression(const REAL* src, REAL* dst, int width, int height, const float* filter, int flt_width, int flt_height);

// variables for Qilin model
int qilin_cpu_test = 0;
int qilin_gpu_test = 0;
double qilin_a_cpu, qilin_b_cpu, qilin_a_gpu, qilin_b_gpu;
int profiling_problem_size[PROFILE_NUM];
double qilin_cpu_data[PROFILE_NUM];
double qilin_gpu_data[PROFILE_NUM];

// variables for log linear regression model
int log_linear_regression_cpu_test = 0;
int log_linear_regression_gpu_test = 0;
double log_linear_regression_a_cpu, log_linear_regression_b_cpu, log_linear_regression_a_gpu, log_linear_regression_b_gpu;
int log_linear_regression_profiling_problem_size[PROFILE_NUM];
double log_linear_regression_cpu_data[PROFILE_NUM];
double log_linear_regression_gpu_data[PROFILE_NUM];

// variables for log-log linear regression model
int log_log_linear_regression_cpu_test = 0;
int log_log_linear_regression_gpu_test = 0;
double log_log_linear_regression_a_cpu, log_log_linear_regression_b_cpu, log_log_linear_regression_a_gpu, log_log_linear_regression_b_gpu;
int log_log_linear_regression_profiling_problem_size[PROFILE_NUM];
double log_log_linear_regression_cpu_data[PROFILE_NUM];
double log_log_linear_regression_gpu_data[PROFILE_NUM];


static double read_timer_ms() {
    struct timeb tm;
    ftime(&tm);
    return (double) tm.time * 1000.0 + (double) tm.millitm;
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

void linear_regression(int* problem_size, double* execution_time, int amount, double* result) {
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
    REAL *result_qilin = (REAL *) malloc(sizeof(REAL) * n * m);
    REAL *result_log_linear_regression = (REAL *) malloc(sizeof(REAL) * n * m);
    REAL *result_log_log_linear_regression = (REAL *) malloc(sizeof(REAL) * n * m);

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

    float fc = filter[2][2];
    float fn0 = filter[1][2];
    float fs0 = filter[3][2];
    float fw0 = filter[2][1];
    float fe0 = filter[2][3];
    float fn1 = filter[0][2];
    float fs1 = filter[4][2];
    float fw1 = filter[2][0];
    float fe1 = filter[2][4];

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
        qilin(u, result_qilin, width, height, filter[0], FILTER_WIDTH, FILTER_HEIGHT);
        log_linear_regression(u, result_log_linear_regression, width, height, filter[0], FILTER_WIDTH, FILTER_HEIGHT);
        log_log_linear_regression(u, result_log_log_linear_regression, width, height, filter[0], FILTER_WIDTH, FILTER_HEIGHT);
    };

    // Reset the models after warming up
    qilin_cpu_test = 0;
    qilin_gpu_test = 0;
    log_linear_regression_cpu_test = 0;
    log_linear_regression_gpu_test = 0;
    log_log_linear_regression_cpu_test = 0;
    log_log_linear_regression_gpu_test = 0;

    double elapsed = read_timer_ms();
    double cpu_time = 0.0;
    double gpu_time = 0.0;
    double qilin_time = 0.0;
    double log_linear_regression_time = 0.0;
    double log_log_linear_regression_time = 0.0;
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
        qilin(u, result_qilin, width2, height2, filter[0], FILTER_WIDTH, FILTER_HEIGHT);
        qilin_time += read_timer_ms() - elapsed;
        elapsed = read_timer_ms();
        log_linear_regression(u, result_log_linear_regression, width2, height2, filter[0], FILTER_WIDTH, FILTER_HEIGHT);
        log_linear_regression_time += read_timer_ms() - elapsed;
        elapsed = read_timer_ms();
        log_log_linear_regression(u, result_log_log_linear_regression, width2, height2, filter[0], FILTER_WIDTH, FILTER_HEIGHT);
        log_log_linear_regression_time += read_timer_ms() - elapsed;

        dif = 0.0;
        for (int j = 0; j < width2*height2; j++) {
            int x = j % width2;
            int y = j / width2;
            if (x > FILTER_WIDTH/2 && x < width2 - FILTER_WIDTH/2 && y > FILTER_HEIGHT/2 && y < height2 - FILTER_HEIGHT/2) {
                dif += fabs(result_log_log_linear_regression[j] - result_cpu[j]);
                dif += fabs(result_log_linear_regression[j] - result_qilin[j]);
            };
            if (dif != 0.0) {
                printf("verify dif =%g\n", dif);
            };
        };
    };

    fprintf(stderr, "CPU total time (ms): %g\n", cpu_time);
    fprintf(stderr, "GPU total time (ms): %g\n", gpu_time);
    fprintf(stderr, "Qilin total time (ms): %g\n", qilin_time);
    fprintf(stderr, "Log Linear Regression total time (ms): %g\n", log_linear_regression_time);
    fprintf(stderr, "Log-Log Linear Regression total time (ms): %g\n", log_log_linear_regression_time);
    printf("%d, %lg, %lg, %lg, %lg, %lg\n", m, cpu_time, gpu_time, qilin_time, log_linear_regression_time, log_log_linear_regression_time);

    free(u);
    free(result);
    free(result_cpu);
    free(result_qilin);
    free(result_log_linear_regression);
    free(result_log_log_linear_regression);

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


void qilin(const REAL* src, REAL* dst, int width, int height, const float* filter, int flt_width, int flt_height) {
    double elapsed = read_timer_ms();
    int N = width*height;
    double cpu_predicted_time = 0.0;
    double gpu_predicted_time = 0.0;
    double result[2];

    if (qilin_cpu_test < PROFILE_NUM) {
        elapsed = read_timer_ms();
        stencil_omp_cpu(src, dst, width, height, filter, FILTER_WIDTH, FILTER_HEIGHT);
        elapsed = read_timer_ms() - elapsed;
        if (elapsed == 0) elapsed = 0.001;
        profiling_problem_size[qilin_cpu_test] = width;
        qilin_cpu_data[qilin_cpu_test] = elapsed;
        qilin_cpu_test += 1;
        if (qilin_cpu_test == PROFILE_NUM) {
            linear_regression(profiling_problem_size, qilin_cpu_data, 20, result);
            qilin_a_cpu = result[0];
            qilin_b_cpu = result[1];
            //print_array("CPM", "cc", qilin_cpu_data, 20);
            fprintf(stderr, "Qilin CPU: y = %lg * x + %lg\n", qilin_a_cpu, qilin_b_cpu);
        }
    } else if (qilin_gpu_test < PROFILE_NUM) {
        elapsed = read_timer_ms();
        stencil_omp_gpu(src, dst, width, height, filter, FILTER_WIDTH, FILTER_HEIGHT);
        elapsed = read_timer_ms() - elapsed;
        if (elapsed == 0) elapsed = 0.001;
        qilin_gpu_data[qilin_gpu_test] = elapsed;
        qilin_gpu_test += 1;
        if (qilin_gpu_test == PROFILE_NUM) {
            linear_regression(profiling_problem_size, qilin_gpu_data, 20, result);
            qilin_a_gpu = result[0];
            qilin_b_gpu = result[1];
            //print_array("CPM", "cg", qilin_gpu_data, 20);
            fprintf(stderr, "Qilin GPU: y = %lg * x + %lg\n", qilin_a_gpu, qilin_b_gpu);
        }
    } else {
        cpu_predicted_time = qilin_a_cpu * width + qilin_b_cpu;
        gpu_predicted_time = qilin_a_gpu * width + qilin_b_gpu;
        if (cpu_predicted_time < gpu_predicted_time) {
            //printf("Qilin: %d, use CPU\n", width);
            stencil_omp_cpu(src, dst, width, height, filter, FILTER_WIDTH, FILTER_HEIGHT);
        }
        else {
            //printf("Qilin: %d, use GPU\n", width);
            stencil_omp_gpu(src, dst, width, height, filter, FILTER_WIDTH, FILTER_HEIGHT);
        };
    };
}


void log_linear_regression(const REAL* src, REAL* dst, int width, int height, const float* filter, int flt_width, int flt_height) {
    double elapsed = read_timer_ms();
    double cpu_predicted_time = 0.0;
    double gpu_predicted_time = 0.0;
    double result[2];

    if (log_linear_regression_cpu_test < PROFILE_NUM) {
        elapsed = read_timer_ms();
        stencil_omp_cpu(src, dst, width, height, filter, FILTER_WIDTH, FILTER_HEIGHT);
        elapsed = read_timer_ms() - elapsed;
        if (elapsed == 0) elapsed = 0.001;
        log_linear_regression_profiling_problem_size[log_linear_regression_cpu_test] = log(width);
        log_linear_regression_cpu_data[log_linear_regression_cpu_test] = elapsed;
        log_linear_regression_cpu_test += 1;
        if (log_linear_regression_cpu_test == PROFILE_NUM) {
            linear_regression(log_linear_regression_profiling_problem_size, log_linear_regression_cpu_data, 20, result);
            log_linear_regression_a_cpu = result[0];
            log_linear_regression_b_cpu = result[1];
            //print_array("FPM", "fc", log_linear_regression_cpu_data, 20);
            fprintf(stderr, "Log Linear Regression CPU: y = %lg * ln(x) + %lg\n", log_linear_regression_a_cpu, log_linear_regression_b_cpu);
        }
    } else if (log_linear_regression_gpu_test < PROFILE_NUM) {
        elapsed = read_timer_ms();
        stencil_omp_gpu(src, dst, width, height, filter, FILTER_WIDTH, FILTER_HEIGHT);
        elapsed = read_timer_ms() - elapsed;
        if (elapsed == 0) elapsed = 0.001;
        log_linear_regression_gpu_data[log_linear_regression_gpu_test] = elapsed;
        log_linear_regression_gpu_test += 1;
        if (log_linear_regression_gpu_test == PROFILE_NUM) {
            linear_regression(log_linear_regression_profiling_problem_size, log_linear_regression_gpu_data, 20, result);
            log_linear_regression_a_gpu = result[0];
            log_linear_regression_b_gpu = result[1];
            //print_array("FPM", "fg", log_linear_regression_gpu_data, 20);
            fprintf(stderr, "Log Linear Regression GPU: y = %lg * ln(x) + %lg\n", log_linear_regression_a_gpu, log_linear_regression_b_gpu);
        }
    } else {
        cpu_predicted_time = log_linear_regression_a_cpu * log(width) + log_linear_regression_b_cpu;
        gpu_predicted_time = log_linear_regression_a_gpu * log(width) + log_linear_regression_b_gpu;
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


void log_log_linear_regression(const REAL* src, REAL* dst, int width, int height, const float* filter, int flt_width, int flt_height) {
    double elapsed = read_timer_ms();
    double cpu_predicted_time = 0.0;
    double gpu_predicted_time = 0.0;
    double result[2];

    if (log_log_linear_regression_cpu_test < PROFILE_NUM) {
        elapsed = read_timer_ms();
        stencil_omp_cpu(src, dst, width, height, filter, FILTER_WIDTH, FILTER_HEIGHT);
        elapsed = read_timer_ms() - elapsed;
        if (elapsed == 0) elapsed = 0.001;
        log_log_linear_regression_profiling_problem_size[log_log_linear_regression_cpu_test] = log(width)/log(2);
        log_log_linear_regression_cpu_data[log_log_linear_regression_cpu_test] = log(elapsed)/log(2);
        log_log_linear_regression_cpu_test += 1;
        if (log_log_linear_regression_cpu_test == PROFILE_NUM) {
            linear_regression(log_log_linear_regression_profiling_problem_size, log_log_linear_regression_cpu_data, 20, result);
            log_log_linear_regression_a_cpu = result[0];
            log_log_linear_regression_b_cpu = result[1];
            //print_array("FPM", "fc", log_linear_regression_cpu_data, 20);
            fprintf(stderr, "Log-Log Linear Regression CPU: log(y) = %lg * log(x) + %lg\n", log_log_linear_regression_a_cpu, log_log_linear_regression_b_cpu);
        }
    } else if (log_log_linear_regression_gpu_test < PROFILE_NUM) {
        elapsed = read_timer_ms();
        stencil_omp_gpu(src, dst, width, height, filter, FILTER_WIDTH, FILTER_HEIGHT);
        elapsed = read_timer_ms() - elapsed;
        if (elapsed == 0) elapsed = 0.001;
        log_log_linear_regression_gpu_data[log_log_linear_regression_gpu_test] = log(elapsed)/log(2);
        log_log_linear_regression_gpu_test += 1;
        if (log_log_linear_regression_gpu_test == PROFILE_NUM) {
            linear_regression(log_log_linear_regression_profiling_problem_size, log_log_linear_regression_gpu_data, 20, result);
            log_log_linear_regression_a_gpu = result[0];
            log_log_linear_regression_b_gpu = result[1];
            //print_array("FPM", "fg", log_linear_regression_gpu_data, 20);
            fprintf(stderr, "Log-Log Linear Regression GPU: log(y) = %lg * log(x) + %lg\n", log_log_linear_regression_a_gpu, log_log_linear_regression_b_gpu);
        }
    } else {
        cpu_predicted_time = log_log_linear_regression_a_cpu * (log(width)/log(2)) + log_log_linear_regression_b_cpu;
        gpu_predicted_time = log_log_linear_regression_a_gpu * (log(width)/log(2)) + log_log_linear_regression_b_gpu;
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

