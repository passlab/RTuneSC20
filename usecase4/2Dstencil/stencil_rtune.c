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
#define PROBLEM_SIZE 256
#define TEAM_SIZE 128
#define PROFILE_NUM 20
#define MAX_ITER 5000

// clang -fopenmp -fopenmp-targets=nvptx64 -lm stencil_rtune.c -o stencil.out
// Usage: ./stencil.out <size>
// e.g. ./stencil.out 512

void stencil_omp_cpu(const REAL* src, REAL* dst, int width, int height, const float* filter, int flt_width, int flt_height);
void stencil_omp_gpu(const REAL* src, REAL* dst, int width, int height, const float* filter, int flt_width, int flt_height);
void linear_regression(const REAL* src, REAL* dst, int width, int height, const float* filter, int flt_width, int flt_height);
void amr_stencil_rtune_lowLevelAPI(const REAL* src, REAL* dst, int* width, int* height, const float* filter, int flt_width, int flt_height);
void amr_stencil_rtune_with_tests(const REAL* src, REAL* dst, int* width, int* height, const float* filter, int flt_width, int flt_height);
void amr_stencil_rtune_overhead(const REAL* src, REAL* dst, int* width, int* height, const float* filter, int flt_width, int flt_height);

// variables for linear regression model
int linear_regression_cpu_test = 0;
int linear_regression_gpu_test = 0;
double linear_regression_a_cpu, linear_regression_b_cpu, linear_regression_a_gpu, linear_regression_b_gpu;
int* linear_regression_profiling_problem_size = NULL;
double* linear_regression_cpu_data = NULL;
double* linear_regression_gpu_data = NULL;
int profile_num = 0;
double single_iteration_time = 0;
REAL *result_cpu = NULL;
REAL *result_gpu = NULL;
int max_problem_size = PROBLEM_SIZE;

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
    int n = PROBLEM_SIZE;
    int m = PROBLEM_SIZE;
    int iteration_number = MAX_ITER;
    int i, j, k;

    // because the selected profiling problem size is up to 2000, the initialized data size is at least 2000.
    // otherwise, it will cause out-of-index access
    if (argc > 1) {
        n = atoi(argv[1]);
        m = atoi(argv[1]);
    };
    if (m < 2000) {
        m = 2000;
        n = 2000;
    };
    REAL *u = (REAL *) malloc(sizeof(REAL) * n * m);
    result_cpu = (REAL *) malloc(sizeof(REAL) * n * m);
    result_gpu = (REAL *) malloc(sizeof(REAL) * n * m);
    REAL *result_linear_regression = (REAL *) malloc(sizeof(REAL) * n * m);
    initialize(n, m, u);

    // reset the problem size to the actual value
    if (argc > 1) {
        n = atoi(argv[1]);
        m = atoi(argv[1]);
    }
    else {
        n = PROBLEM_SIZE;
        m = PROBLEM_SIZE;
    };
    max_problem_size = m;

    // mode 0: run with RTune without any measurement or verification (default)
    // mode 1: run with RTune and measure its overhead, but without verification
    // mode 2: run with CPU, GPU, RTune versions and compare their execution time
    int mode = 0;
    if (argc > 2) {
        mode = atoi(argv[2]);
    }

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

    // warm up the computing functions
    for (i = 0; i < 8; i++) {
        stencil_omp_cpu(u, result_cpu, width, height, filter[0], FILTER_WIDTH, FILTER_HEIGHT);
        stencil_omp_gpu(u, result_gpu, width, height, filter[0], FILTER_WIDTH, FILTER_HEIGHT);
    };

    if (mode == 0) {
        amr_stencil_rtune_lowLevelAPI(u, result_linear_regression, problem_size, problem_size, filter[0], FILTER_WIDTH, FILTER_HEIGHT);
    } else if (mode == 1) {
        amr_stencil_rtune_overhead(u, result_linear_regression, problem_size, problem_size, filter[0], FILTER_WIDTH, FILTER_HEIGHT);
    } else {
        amr_stencil_rtune_with_tests(u, result_linear_regression, problem_size, problem_size, filter[0], FILTER_WIDTH, FILTER_HEIGHT);
    }

    free(u);
    free(result_cpu);
    free(result_gpu);
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


void manual_rtune_begin(int* cpu_exec, int problem_size) {

    double cpu_predicted_time = 0.0;
    double gpu_predicted_time = 0.0;

    // start the CPU sampling first
    if (linear_regression_cpu_test < PROFILE_NUM) {
        single_iteration_time = read_timer_ms();
        linear_regression_profiling_problem_size[linear_regression_cpu_test] = problem_size;
        *cpu_exec = 1;
    // if CPU modeling is done, start the GPU sampling
    } else if (linear_regression_gpu_test < PROFILE_NUM) {
        single_iteration_time = read_timer_ms();
        *cpu_exec = 0;
    // predict if both models have been created
    } else {
        cpu_predicted_time = linear_regression_a_cpu * problem_size + linear_regression_b_cpu;
        gpu_predicted_time = linear_regression_a_gpu * problem_size + linear_regression_b_gpu;
        if (cpu_predicted_time < gpu_predicted_time) {
            *cpu_exec = 1;
        }
        else {
            *cpu_exec = 0;
        };
    };
}

void manual_rtune_end() {

    double elapsed = read_timer_ms();
    double result[2];

    // complete the CPU sampling first
    if (linear_regression_cpu_test < PROFILE_NUM) {
        single_iteration_time = read_timer_ms() - single_iteration_time;
        // if the execution time is too short to measure, it is set to 0.001ms.
        if (single_iteration_time == 0) single_iteration_time = 0.001;
        linear_regression_cpu_data[linear_regression_cpu_test] = single_iteration_time;
        linear_regression_cpu_test += 1;
        // if the CPU sampling is done, create the CPU model using simple linear regression
        if (linear_regression_cpu_test == PROFILE_NUM) {
            fit_linear_regression(linear_regression_profiling_problem_size, linear_regression_cpu_data, 20, result);
            linear_regression_a_cpu = result[0];
            linear_regression_b_cpu = result[1];
            fprintf(stderr, "Linear Regression CPU: y = %lg * x + %lg\n", linear_regression_a_cpu, linear_regression_b_cpu);
        }
    // if CPU modeling is done, complete the GPU sampling
    } else if (linear_regression_gpu_test < PROFILE_NUM) {
        single_iteration_time = read_timer_ms() - single_iteration_time;
        // if the execution time is too short to measure, it is set to 0.001ms.
        if (elapsed == 0) elapsed = 0.001;
        linear_regression_gpu_data[linear_regression_gpu_test] = single_iteration_time;
        linear_regression_gpu_test += 1;
        // if the GPU sampling is done, create the GPU model using simple linear regression
        if (linear_regression_gpu_test == PROFILE_NUM) {
            fit_linear_regression(linear_regression_profiling_problem_size, linear_regression_gpu_data, 20, result);
            linear_regression_a_gpu = result[0];
            linear_regression_b_gpu = result[1];
            fprintf(stderr, "Linear Regression GPU: y = %lg * x + %lg\n", linear_regression_a_gpu, linear_regression_b_gpu);
        }
    };
}

void manual_rtune_objective_set_sample_attr(int sample_rate, int num_samples) {
    // implementation: specify the number and collecting frequency of samples
    profile_num = num_samples/2;
    linear_regression_profiling_problem_size = (int*) malloc(sizeof(int) * profile_num);
    linear_regression_cpu_data = (double*) malloc(sizeof(double) * profile_num);
    linear_regression_gpu_data = (double*) malloc(sizeof(double) * profile_num);
};

void amr_stencil_rtune_lowLevelAPI(const REAL* src, REAL* dst, int* width, int* height, const float* filter, int flt_width, int flt_height) {
    /* width == height == problem size */
    int cpu_exec, iter_count = 0;

    /* API for tuning */
    //rtune_region_t * stencil_region = rtune_init_region("stencil kernel region");
    //int * problem_size_var = rtune_var_add_ext(stencil_region, "problem_size", __RTUNE_int, &width, &width, __RTUNE_REGION_BEGIN);
    //double * cpu_exe_time = rtune_var_add_diff(stencil_region, "cpu exe time", __RTUNE_double, get_timer_ms, NULL);
    //double * gpu_exe_time = rtune_var_add_diff(stencil_region, "gpu exe time", __RTUNE_double, get_timer_ms, NULL);
    
    //rtune_model_t rtune_cpu_model = rtune_model_add(stencil_region, "cpu_exe_model_size", cpu_exe_time, problem_size_var, __RTUNE_LINEAR_MODEL /* model used */);
    //rtune_model_t rtune_gpu_model = rtune_model_add(stencil_region, "gpu_exe_model_size", gpu_exe_time, problem_size_var, __RTUNE_LINEAR_MODEL /* model used */);

    //rtune_objective_t threshold_obj = rtune_objective_add_select2(stencil_region, "threshold cpu-gpu", rtune_cpu_model, rtune_gpu_model, _RTUNE_SELECT_MIN, &cpu_exe, &gpu_exe);

    //rtune_objective_set_sample_attr(threshold_obj, 1/* sample_rate */, 40/* num_samples*/, __RTUNE_SAMPLE_DIST_BLOCK);
    manual_rtune_objective_set_sample_attr(1/* sample_rate */, 40/* num_samples*/);

    while (iter_count < MAX_ITER) {
        //rtune_begin(stencil_region);
        manual_rtune_begin(&cpu_exec, width[iter_count]);
        if (cpu_exec) stencil_omp_cpu(src, dst, width[iter_count], height[iter_count], filter, FILTER_WIDTH, FILTER_HEIGHT);
        else stencil_omp_gpu(src, dst, width[iter_count], height[iter_count], filter, FILTER_WIDTH, FILTER_HEIGHT);
        //rtune_end(stencil_region);
        manual_rtune_end();
        iter_count++;
    }
}

// Measure the overhead of using RTune
void amr_stencil_rtune_overhead(const REAL* src, REAL* dst, int* width, int* height, const float* filter, int flt_width, int flt_height) {
    /* width == height == problem size */
    int cpu_exec, iter_count = 0;

    /* API for tuning */
    //rtune_objective_set_sample_attr(threshold_obj, 1/* sample_rate */, 40/* num_samples*/, __RTUNE_SAMPLE_DIST_BLOCK);
    manual_rtune_objective_set_sample_attr(1/* sample_rate */, 40/* num_samples*/);

    double elapsed = read_timer_ms();
    double cpu_time = 0.0;
    double gpu_time = 0.0;
    double sampling_overhead = 0.0;
    double fitting_overhead = 0.0;
    double prediction_overhead = 0.0;

    double total_time = read_timer_ms();
    while (iter_count < MAX_ITER) {
        // sampling is started in the begin region of RTune
        // the prediction is completed by calling the model in the begin region
        elapsed = read_timer_ms();
        manual_rtune_begin(&cpu_exec, width[iter_count]);
        if (iter_count == profile_num - 1 || iter_count == profile_num*2 - 1) {
            fitting_overhead += read_timer_ms() - elapsed;
        } else {
            prediction_overhead += read_timer_ms() - elapsed;
        }

        // computing
        if (cpu_exec) stencil_omp_cpu(src, dst, width[iter_count], height[iter_count], filter, FILTER_WIDTH, FILTER_HEIGHT);
        else stencil_omp_gpu(src, dst, width[iter_count], height[iter_count], filter, FILTER_WIDTH, FILTER_HEIGHT);

        // the fitting happens in the end region of RTune. For this particular iteration, we count one time of sampling overhead into fitting overhead because they happen together in the same iteration.
        // sampling is completed in the end region of RTune
        // the prediction has nothing to do with the end region, but it still triggers a function call that can be considered as overhead
        elapsed = read_timer_ms();
        manual_rtune_end();
        if (iter_count == profile_num - 1 || iter_count == profile_num*2 - 1) {
            fitting_overhead += read_timer_ms() - elapsed;
        } else if (iter_count < profile_num*2) {
            sampling_overhead += read_timer_ms() - elapsed;
        } else {
            prediction_overhead += read_timer_ms() - elapsed;
        }
        iter_count++;
    }
    total_time = read_timer_ms() - total_time;

    // All the overheads are collected from multiple iterations and print the average
    // given 40 samples:
    // 38 iterations of sampling are measured.
    // 2 iterations of fitting are measured.
    // 5000 - 40 = 4960 iterations of prediction are measured.
    fprintf(stderr, "Total sampling overhead (ms): %g; Average sampling overhead per iteration: %g\n", sampling_overhead, sampling_overhead/(profile_num*2-2));
    fprintf(stderr, "Total fitting overhead (ms): %g; Average fitting overhead per iteration: %g\n", fitting_overhead, fitting_overhead/2);
    fprintf(stderr, "Total prediction overhead (ms): %g; Average prediction overhead per iteration: %g\n", prediction_overhead, prediction_overhead/(MAX_ITER - profile_num*2));
    fprintf(stderr, "Total execution time (ms): %g\n", total_time);

}

// Compare the execution time of three versions: CPU, GPU, RTune
void amr_stencil_rtune_with_tests(const REAL* src, REAL* dst, int* width, int* height, const float* filter, int flt_width, int flt_height) {
    /* width == height == problem size */
    int cpu_exec, iter_count = 0;

    /* API for tuning */
    //rtune_objective_set_sample_attr(threshold_obj, 1/* sample_rate */, 40/* num_samples*/, __RTUNE_SAMPLE_DIST_BLOCK);
    manual_rtune_objective_set_sample_attr(1/* sample_rate */, 40/* num_samples*/);

    double elapsed = read_timer_ms();
    double cpu_time = 0.0;
    double gpu_time = 0.0;
    double linear_regression_time = 0.0;
    double dif = 0.0;

    while (iter_count < MAX_ITER) {
        // compute only on GPU
        elapsed = read_timer_ms();
        stencil_omp_gpu(src, result_gpu, width[iter_count], height[iter_count], filter, FILTER_WIDTH, FILTER_HEIGHT);
        gpu_time += read_timer_ms() - elapsed;

        // compute only on CPU
        elapsed = read_timer_ms();
        stencil_omp_cpu(src, result_cpu, width[iter_count], height[iter_count], filter, FILTER_WIDTH, FILTER_HEIGHT);
        cpu_time += read_timer_ms() - elapsed;

        // compute on CPU or GPU guided by RTune
        elapsed = read_timer_ms();
        //rtune_begin(stencil_region);
        manual_rtune_begin(&cpu_exec, width[iter_count]);
        if (cpu_exec) stencil_omp_cpu(src, dst, width[iter_count], height[iter_count], filter, FILTER_WIDTH, FILTER_HEIGHT);
        else stencil_omp_gpu(src, dst, width[iter_count], height[iter_count], filter, FILTER_WIDTH, FILTER_HEIGHT);
        //rtune_end(stencil_region);
        manual_rtune_end();
        linear_regression_time += read_timer_ms() - elapsed;

        // verify the correctness of results computed using RTune
        dif = 0.0;
        for (int j = 0; j < width[iter_count]*height[iter_count]; j++) {
            int x = j % width[iter_count];
            int y = j / width[iter_count];
            if (x > FILTER_WIDTH/2 && x < width[iter_count] - FILTER_WIDTH/2 && y > FILTER_HEIGHT/2 && y < height[iter_count] - FILTER_HEIGHT/2) {
                dif += fabs(dst[j] - result_cpu[j]);
            };
            if (dif != 0.0) {
                printf("verify dif =%g\n", dif);
            };
        };

        iter_count++;
    }

    fprintf(stderr, "CPU total time (ms): %g\n", cpu_time);
    fprintf(stderr, "GPU total time (ms): %g\n", gpu_time);
    fprintf(stderr, "Linear Regression total time (ms): %g\n", linear_regression_time);
    printf("%d, %lg, %lg, %lg\n", max_problem_size, cpu_time, gpu_time, linear_regression_time);
}
