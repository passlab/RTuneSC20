#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/timeb.h>
#include <mpi.h>
#define REAL double


/* compile and run the program using the following command
 *    mpicc jacobi_mpi.c -lm -o jacobi_mpi
 *    mpirun -np 4 ./jacobi_mpi
 */

/* read timer in second */
double read_timer() {
    struct timeb tm;
    ftime(&tm);
    return (double) tm.time + (double) tm.millitm / 1000.0;
}

/* read timer in ms */
double read_timer_ms() {
    struct timeb tm;
    ftime(&tm);
    return (double) tm.time * 1000.0 + (double) tm.millitm;
}

/************************************************************
 * program to solve a finite difference
 * discretization of Helmholtz equation :
 * (d2/dx2)u + (d2/dy2)u - alpha u = f
 * using Jacobi iterative method.
 *
 * Input :  n - grid dimension in x direction
 *          m - grid dimension in y direction
 *          alpha - Helmholtz constant (always greater than 0.0)
 *          tol   - error tolerance for iterative solver
 *          relax - Successice over relaxation parameter
 *          mits  - Maximum iterations for iterative solver
 *
 * On output
 *       : u(n,m) - Dependent variable (solutions)
 *       : f(n,m) - Right hand side function
 *************************************************************/

// flexible between REAL and double
#define DEFAULT_DIMSIZE 256

void fit(REAL* errors, int start_index, int start_iter, int amount, REAL* result) {
    REAL sumx = 0.0, sumy = 0.0, sumxy = 0.0, sumx2 = 0.0;
    REAL a, b;
    int i;
    if (amount > 100) {
        amount = 100;
    };
    REAL* log_errors;
    long int* iterations;
    log_errors = malloc(amount * sizeof(REAL));
    iterations = malloc(amount * sizeof(long int));
    for(i = 0; i <= amount; i++) {
        log_errors[i] = log10(errors[(start_index + i)%100]);
        iterations[i] = start_iter - 5 * (amount - i);
        printf("Data %d: iter: %ld, error: %lg, log_error: %lg\n", i, iterations[i], errors[(start_index + i)%100], log_errors[i]);
    }
    for(i = 0; i <= amount-1; i++) {
        sumx = sumx + iterations[i];
        sumx2 = sumx2 + iterations[i] * iterations[i];
        sumy = sumy + log_errors[i];
        sumxy = sumxy + iterations[i] * log_errors[i];
    }
    b = ((sumx2 * sumy - sumx * sumxy) * 1.0 / (amount * sumx2-sumx * sumx) * 1.0);
    a = ((amount * sumxy - sumx * sumy) * 1.0 / (amount * sumx2-sumx * sumx) * 1.0);
    result[0] = a;
    result[1] = b;
}

void print_array(char * title, char * name, REAL * A, long n, long m) {
    printf("%s:\n", title);
    long i, j;
    for (i = 0; i < n; i++) {
        for (j = 0; j < m; j++) {
            printf("%s[%ld][%ld]:%f  ", name, i, j, A[i * m + j]);
        }
        printf("\n");
    }
    printf("\n");
}


/*      subroutine initialize (n,m,alpha,dx,dy,u,f)
 ******************************************************
 * Initializes data
 * Assumes exact solution is u(x,y) = (1-x^2)*(1-y^2)
 *
 ******************************************************/
void initialize(long n, long m, REAL alpha, REAL dx, REAL dy, REAL * u_p, REAL * f_p) {
    long i;
    long j;
    long xx;
    long yy;
    REAL (*u)[m] = (REAL(*)[m])u_p;
    REAL (*f)[m] = (REAL(*)[m])f_p;
    
    //double PI=3.1415926;
    /* Initialize initial condition and RHS */
    #pragma omp parallel for private(xx,yy,j,i)
    for (i = 0; i < n; i++)
        for (j = 0; j < m; j++) {
            xx = ((int) (-1.0 + (dx * (i - 1))));
            yy = ((int) (-1.0 + (dy * (j - 1))));
            u[i][j] = 0.0;
            f[i][j] = (((((-1.0 * alpha) * (1.0 - (xx * xx)))
                         * (1.0 - (yy * yy))) - (2.0 * (1.0 - (xx * xx))))
                       - (2.0 * (1.0 - (yy * yy))));
        }
}

/*  subroutine error_check (n,m,alpha,dx,dy,u,f)
 implicit none
 ************************************************************
 * Checks error between numerical and exact solution
 *
 ************************************************************/
double error_check(long n, long m, REAL alpha, REAL dx, REAL dy, REAL * u_p, REAL * f_p) {
    int i;
    int j;
    REAL xx;
    REAL yy;
    REAL temp;
    double error;
    error = 0.0;
    REAL (*u)[m] = (REAL(*)[m])u_p;
    REAL (*f)[m] = (REAL(*)[m])f_p;
    #pragma omp parallel for private(xx,yy,temp,j,i) reduction(+:error)
    for (i = 0; i < n; i++)
        for (j = 0; j < m; j++) {
            xx = (-1.0 + (dx * (i - 1)));
            yy = (-1.0 + (dy * (j - 1)));
            temp = (u[i][j] - ((1.0 - (xx * xx)) * (1.0 - (yy * yy))));
            error = (error + (temp * temp));
        }
    error = (sqrt(error) / (n * m));
    return error;
}
void jacobi_seq(long n, long m, REAL dx, REAL dy, REAL alpha, REAL relax, REAL * u_p, REAL * f_p, REAL tol, int mits);
void jacobi_mpi(long n, long m, REAL dx, REAL dy, REAL alpha, REAL relax, REAL * u_p, REAL * f_p, REAL tol, int mits, REAL rtune_threshold);
int numprocs;
int myrank;

int main(int argc, char * argv[]) {
    long n = DEFAULT_DIMSIZE;
    long m = DEFAULT_DIMSIZE;
    REAL alpha = 0.0543;
    REAL tol = 0.000000001;
    REAL relax = 1.0;
    int mits = 50000;
    REAL rtune_threshold = 0.0002;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    if (myrank == 0) {
    fprintf(stderr,"Usage: jacobi [<n> <m> <rtune> <tol> <relax> <mits>]\n");
    fprintf(stderr, "\tn - grid dimension in x direction, default: %ld\n", n);
    fprintf(stderr, "\tm - grid dimension in y direction, default: n if provided or %ld\n", m);
    fprintf(stderr, "\trtune - threshold for RTune to terminate the computing (always greater than 0.0), default: %g\n", rtune_threshold);
    fprintf(stderr, "\ttol   - error tolerance for iterative solver, default: %g\n", tol);
    fprintf(stderr, "\trelax - Successice over relaxation parameter, default: %g\n", relax);
    fprintf(stderr, "\tmits  - Maximum iterations for iterative solver, default: %d\n", mits);
    }
    if (argc == 2)      { sscanf(argv[1], "%ld", &n); m = n; }
    else if (argc == 3) { sscanf(argv[1], "%ld", &n); sscanf(argv[2], "%ld", &m); }
    else if (argc == 4) { sscanf(argv[1], "%ld", &n); sscanf(argv[2], "%ld", &m); rtune_threshold = atof(argv[3]); /*sscanf(argv[3], "%g", &rtune_threshold);*/ }
    else if (argc == 5) { sscanf(argv[1], "%ld", &n); sscanf(argv[2], "%ld", &m); sscanf(argv[3], "%g", &rtune_threshold); sscanf(argv[4], "%g", &tol); }
    else if (argc == 6) { sscanf(argv[1], "%ld", &n); sscanf(argv[2], "%ld", &m); sscanf(argv[3], "%g", &rtune_threshold); sscanf(argv[4], "%g", &tol); sscanf(argv[5], "%g", &relax); }
    else if (argc == 7) { sscanf(argv[1], "%ld", &n); sscanf(argv[2], "%ld", &m); sscanf(argv[3], "%g", &rtune_threshold); sscanf(argv[4], "%g", &tol); sscanf(argv[5], "%g", &relax); sscanf(argv[6], "%d", &mits); }
    else {
        /* the rest of arg ignored */
    }
    REAL dx; /* grid spacing in x direction */
    REAL dy; /* grid spacing in y direction */
    dx = (2.0 / (n - 1));
    dy = (2.0 / (m - 1));
    
    REAL * u;
    REAL * f;
    REAL * umpi;
    REAL * fmpi;
    double elapsed_seq, elapsed_mpi;
    
    if (myrank == 0) {
        printf("jacobi %ld %ld %g %g %g %d\n", n, m, rtune_threshold, tol, relax, mits);
        printf("------------------------------------------------------------------------------------------------------\n");
        /** init the array */
        
        u = malloc(sizeof(REAL)*n*m);
        f = malloc(sizeof(REAL)*n*m);
        
        umpi = malloc(sizeof(REAL)*n*m);
        fmpi = malloc(sizeof(REAL)*n*m);
        
        initialize(n, m, alpha, dx, dy, u, f);
        
        memcpy(umpi, u, n*m*sizeof(REAL));
        memcpy(fmpi, f, n*m*sizeof(REAL));
   
        printf("================================= Sequential Execution ======================================\n");
        elapsed_seq = read_timer_ms();
        //jacobi_seq(n, m, dx, dy, alpha, relax, u, f, tol, mits);
        elapsed_seq = read_timer_ms() - elapsed_seq;
        printf("\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    
    /* TODO #1: process 0 performs data decomposition and distribution of the umpi and fmpi arrays to
     * other processes using MPI_Send/Recv. MPI_Scatter may work, but the recommendation is to implement using
     * MPI_Send/Recv first and then to see whether you can convert to MPI_Scatter.
     *
     * Row-wise decomposition should be used. Memory needs to be allocated for holding the umpi and
     * fmpi data distributed to each process.
     *
     * Assuming N is dividable by numprocs. Considering the boundary row(s) that
     * need to be exchanged between processes, you will need to properly set the address of MPI_Send/Recv
     * of of each process for sending data to its neighbour(s).
     * Some processes (0, numprocs-1) have only one neighbour, and others have two neighbours.
     *
     *
     */

    int temp = n / numprocs;
    int other, NumRows, NumRowsToSend, NumRowsToRecv;

    if (myrank == 0 || myrank == (numprocs - 1)) NumRows = temp + 1; 
        else NumRows = temp + 2;

    if(myrank != 0) {
        umpi = malloc(NumRows * m * sizeof(REAL));
        fmpi = malloc(NumRows * m * sizeof(REAL)); 
    }

    if (myrank == 0) {
        for (other = 1; other < numprocs; other++) {
            if (other == (numprocs - 1)) {
                NumRowsToSend = temp + 1;
            }  else {
                NumRowsToSend = temp + 2;
            }
            
            MPI_Send(umpi + (other * temp - 1) * m, NumRowsToSend * m, MPI_DOUBLE, other, other, MPI_COMM_WORLD);
            MPI_Send(fmpi + (other * temp - 1) * m, NumRowsToSend * m, MPI_DOUBLE, other, other, MPI_COMM_WORLD);
        }
    }  else {
        MPI_Recv(umpi, NumRows * m, MPI_DOUBLE, 0, myrank, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(fmpi, NumRows * m, MPI_DOUBLE, 0, myrank, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }


    MPI_Barrier(MPI_COMM_WORLD);

    
    if (myrank == 0) {
        printf("========================== Parallel MPI Execution (%d processes) =============================\n", numprocs);
        elapsed_mpi = read_timer_ms();
    }

    /* TODO #2: perform jacobi iterative method for computation, check more details in jacobi_mpi function comments
     *
     * You do not need to use exactly the same function and its arguments,
     * feel free to adjust as you see fit
     */
    REAL error = 0;
     
    jacobi_mpi(n, m, dx, dy, alpha, relax, umpi, fmpi, tol, mits, rtune_threshold);
    //if (myrank == 0) {
      //  jacobi_seq(n, m, dx, dy, alpha, relax, u, f, tol, mits);
    //} 
    MPI_Barrier(MPI_COMM_WORLD); /* this is necessnary */

    if (myrank == 0) {
        elapsed_mpi = read_timer_ms() - elapsed_mpi;
        printf("\n");
    }
    
    /* TODO #3: Each process sends data (umpi) to process 0 using MPI_Send/Recv, MPI_Gather may work as well. */
        
    if (myrank != 0) {
        MPI_Send(umpi, NumRows * m, MPI_DOUBLE, 0, myrank, MPI_COMM_WORLD);
    } else {
        for (other = 1; other < numprocs; other++) {
            if (other == (numprocs - 1)) {
                NumRowsToRecv = temp + 1;
            } else {
                NumRowsToRecv = temp + 2;
            }
            
            MPI_Recv(umpi + (other * temp - 1) * m, NumRowsToRecv * m, MPI_DOUBLE, other, other, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }


    if (myrank == 0) {
#if CORRECTNESS_CHECK
        print_array("Sequential Run", "u",    (REAL*)u, n, m);
        print_array("MPI Parallel ", "umpi", (REAL*)umpi, n, m);
#endif
  
        double flops = mits*(n-2)*(m-2)*13;
        printf("------------------------------------------------------------------------------------------------------\n");
        printf("Performance:\tRuntime(ms)\tMFLOPS\t\tError\n");
        printf("------------------------------------------------------------------------------------------------------\n");
        printf("base:\t\t%.2f\t\t%.2f\t\t%g\n", elapsed_seq, flops / (1.0e3 * elapsed_seq), error_check(n, m, alpha, dx, dy, u, f));
        printf("mpi(%d processes):\t%.2f\t\t%.2f\t\t%g\n", numprocs, elapsed_mpi, flops / (1.0e3 * elapsed_mpi), error_check(n, m, alpha, dx, dy, umpi, fmpi));
        
        free(u); free(f);
        free(umpi); free(fmpi);
    }
    MPI_Finalize();
    return 0;
}

/*      subroutine jacobi (n,m,dx,dy,alpha,omega,u,f,tol,mits)
 ******************************************************************
 * Subroutine HelmholtzJ
 * Solves poisson equation on rectangular grid assuming :
 * (1) Uniform discretization in each direction, and
 * (2) Dirichlect boundary conditions
 *
 * Jacobi method is used in this routine
 *
 * Input : n,m   Number of grid points in the X/Y directions
 *         dx,dy Grid spacing in the X/Y directions
 *         alpha Helmholtz eqn. coefficient
 *         omega Relaxation factor
 *         f(n,m) Right hand side function
 *         u(n,m) Dependent variable/Solution
 *         tol    Tolerance for iterative solver
 *         mits  Maximum number of iterations
 *
 * Output : u(n,m) - Solution
 *****************************************************************/
void jacobi_seq(long n, long m, REAL dx, REAL dy, REAL alpha, REAL omega, REAL * u_p, REAL * f_p, REAL tol, int mits) {
    long i, j, k;
    REAL error;
    REAL ax;
    REAL ay;
    REAL b;
    REAL resid;
    REAL uold[n][m];
    REAL (*u)[m] = (REAL(*)[m])u_p;
    REAL (*f)[m] = (REAL(*)[m])f_p;
    /*
     * Initialize coefficients */
    /* X-direction coef */
    ax = (1.0 / (dx * dx));
    /* Y-direction coef */
    ay = (1.0 / (dy * dy));
    /* Central coeff */
    b = (((-2.0 / (dx * dx)) - (2.0 / (dy * dy))) - alpha);
    error = (10.0 * tol);
    k = 1;
    while ((k <= mits) && (error > tol)) {
        error = 0.0;
        
        /* Copy new solution into old */
        for (i = 0; i < n; i++)
            for (j = 0; j < m; j++)
                uold[i][j] = u[i][j];
        
        for (i = 1; i < (n - 1); i++)
            for (j = 1; j < (m - 1); j++) {
                resid = (ax * (uold[i - 1][j] + uold[i + 1][j]) + ay * (uold[i][j - 1] + uold[i][j + 1]) + b * uold[i][j] - f[i][j]) / b;
                //printf("i: %d, j: %d, resid: %f\n", i, j, resid);
                
                u[i][j] = uold[i][j] - omega * resid;
                error = error + resid * resid;
            }
        /* Error check */
        if (k % 500 == 0)
            printf("Finished %ld iteration with error: %g\n", k, error);
        error = sqrt(error) / (n * m);
        
        k = k + 1;
    } /*  End iteration loop */
    printf("Total Number of Iterations: %ld\n", k);
    printf("Residual: %.15g\n", error);
}

/**
 * TODO #2: MPI implementation of Jacobi methods using row-wise data distribution. For each iteration of the while loop,
 * each process needs to exchange boundary data with its neighbor(s) using MPI_Send and MPI_Recv, and then perform computation.
 * You can also use MPI_Isend and MPI_Irecv for boundary exchange.
 *
 * In each iteration, error computed by each process needs to be reduced using add ops by process 0 and
 * then broadcasted to all processes, each of which will then check whether they need terminate the while loop. You can use
 * MPI_Allreduce or MPI_Reduce+MPI_Bcast to do that.
 *
 * printf calls should only be performed by process 0
 *
 * You do not need to use the same arguments, feel free to change as you think reasonable.
 *
 * Since this function is called by each process who computes u subarray(the portion of u array distributed from process 0),
 * u_p and f_p is actually the pointer of the subarray, and n and m are for the size of the subarray.
 */
void jacobi_mpi(long n, long m, REAL dx, REAL dy, REAL alpha, REAL omega, REAL * u_p, REAL * f_p, REAL tol, int mits, REAL rtune_threshold) {
    long i, j, k;
    REAL error, local_error;
    REAL ax;
    REAL ay;
    REAL b;
    REAL resid;
    REAL (*u)[m] = (REAL(*)[m])u_p;
    REAL (*f)[m] = (REAL(*)[m])f_p;

    int temp = n / numprocs;
    int NumRows;

    if (myrank == 0 || myrank == (numprocs - 1)) NumRows = temp + 1;
        else NumRows = temp + 2;

    //REAL unew[NumRows][m];
    REAL (*unew)[m] = (REAL(*)[m])u;

    /*
     * Initialize coefficients */
    /* X-direction coef */
    ax = (1.0 / (dx * dx));
    /* Y-direction coef */
    ay = (1.0 / (dy * dy));
    /* Central coeff */
    b = (((-2.0 / (dx * dx)) - (2.0 / (dy * dy))) - alpha);
    error = (10.0 * tol);
    k = 0;
    REAL val[2];
    // the last two values
    val[0] = 0.0;
    val[1] = 0.0;
    // the last two first derivatives
    REAL k1[2];
    // the second derivative
    REAL k2[2];
    k2[1] = 0.0;
    int side_walk = 0;
    REAL k2_limit = rtune_threshold;
    REAL errors[100];
    REAL fit_result[2];
    int start_index = 0;
    int amount = 0;

    REAL computing_time_start, computing_time = 0.0;
    REAL tuning_time_start, tuning_time = 0.0;
    
    while ((k <= mits) && (error > tol)) {
        error = 0.0; 
        local_error = 0.0;
        
        /* Copy new solution into old */
        /* TODO #2.a: since u and f are pointers to the subarray that contains boundary data, you will need to adjust the
         * start and end iteration of i and j depending on whether the process has one neighbour or two neighbours.
         * You need to do similar for the next for loops.
         */
       /* 
        long lb, ub;
        lb = myrank * temp - 1;
        if (lb < 0) {
            lb = 0;
        }
        ub = lb + NumRows - 1;
        if (ub > (n - 1)) {
            ub = n - 1;
        }

        if (myrank == 0) {
            for (j = 0; j < m; j++) {
                uold[0][j] = u[0][j]; 
            }
        }

        if (myrank == (numprocs - 1)) {
            for (j = 0; j < m; j++) {
                uold[n - 1][j] = u[n - 1][j];
            }
        }
        */
        /*
        for (i = 0; i < NumRows; i++) 
            for (j = 0; j < m; j++) 
                uold[i][j] = u[i][j];
         */

        /* TODO #2.b: boundary exchange with neighbour process(es) using MPI_Send/Recv.
         * The memory address of the boundary data should be correctly specified.
         */
        u_p = (REAL*)u; 
        if (myrank == 0) {
            MPI_Send(u_p + (NumRows - 2) * m, m, MPI_DOUBLE, myrank + 1, 0,MPI_COMM_WORLD);
            MPI_Recv(u_p + (NumRows - 1) * m, m, MPI_DOUBLE, myrank + 1, 0,MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        } else if (myrank == (numprocs - 1)) {
            MPI_Recv(u_p, m, MPI_DOUBLE, myrank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Send(u_p + m, m, MPI_DOUBLE, myrank - 1, 0, MPI_COMM_WORLD);
        
        } else {
            MPI_Recv(u_p, m, MPI_DOUBLE, myrank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Send(u_p + m, m, MPI_DOUBLE, myrank - 1, 0, MPI_COMM_WORLD);

            MPI_Send(u_p + (NumRows - 2) * m, m, MPI_DOUBLE, myrank + 1, 0,MPI_COMM_WORLD);
            MPI_Recv(u_p + (NumRows - 1) * m, m, MPI_DOUBLE, myrank + 1, 0,MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        }
 
        if (myrank == 0) {
            computing_time_start = read_timer_ms();
        };
        for (i = 1; i < (NumRows - 1); i++) {
            for (j = 1; j < (m - 1); j++) {
                resid = (ax * (u[i - 1][j] + u[i + 1][j]) + ay * (u[i][j - 1] + u[i][j + 1]) + b * u[i][j] - f[i][j]) / b;
                //printf("i: %d, j: %d, resid: %f\n", i, j, resid);
                
                unew[i][j] = u[i][j] - omega * resid;
                local_error = local_error + resid * resid;
            }
        }
        /* TODO #2.c: compute the global error using MPI_Allreduce or MPI_Reduce+MPI_Bcast
         */

        MPI_Allreduce(&local_error, &error, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        error = sqrt(error) / (n * m);
        if (myrank == 0) {
            computing_time += read_timer_ms() - computing_time_start;
        };

        /* Error Check */
        if (myrank == 0) {
            tuning_time_start = read_timer_ms();
        };
        //if (k % 500 == 0) printf("Finished %ld iteration with error: %g\n", k, error);
        if (k % 5 == 0) {
            k1[0] = val[1] - val[0];
            k1[1] = error - val[1];
            k2[0] = k2[1];
            k2[1] = k1[1] - k1[0];
            //if (myrank == 0) {
            //    printf("%d, %.15lg, %.15lg, %.15lg\n", k, error, k1[1], k2[1]);
            //};
            val[0] = val[1];
            val[1] = error;
            start_index = amount%100;
            errors[start_index] = error;
            amount += 1;
            if (k2[1] > 0 && fabs((k1[0] - k1[1])/k1[0]) < k2_limit) {
                if (side_walk < 15) {
                    side_walk += 1;
                }
                else {
                    if (myrank == 0) {
                        tuning_time += read_timer_ms() - tuning_time_start;
                        fit(errors, start_index, k, amount, fit_result);
                        REAL a = fit_result[0];
                        REAL b = fit_result[1];
                        REAL predict_finishing_iteration = (log10(tol) - b)/a;
                        printf("Saved iterations based on mits: %ld out of %d\n", mits-k, mits);
                        printf("Fitting curve is: y = %lg * x + %lg, targeted y = %lg\n", a, b, log10(tol));
                        printf("Predicted finishing iteration is: %ld\n", (long int)predict_finishing_iteration);
                        printf("Saved iterations based on error tolerance: %ld out of %ld\n", (long int)predict_finishing_iteration-k, (long int)predict_finishing_iteration);

                    }
                    break;
                }
            };
            // Reset the side_walk counter every 200 iterations
            if (k%200 == 0) {
                side_walk = 0;
            };
        };
        if (myrank == 0) {
            tuning_time += read_timer_ms() - tuning_time_start;
        };
        REAL (*swapper)[m] = unew;
        unew = u;
        u = swapper;
        k = k + 1;
    } /*  End iteration loop */
    if (myrank == 0) {
        printf("Computing Time: %.5lg, Tuning Time: %.5lg, Total Time: %.5lg\n", computing_time, tuning_time, computing_time + tuning_time);
    };
    
    if (k%2==1) {/* for odd-number of iterations, the final results are in unew, pointed by u, we need to manually copy to u which is pointed by unew */
        memcpy(unew, u, sizeof(REAL)*NumRows*m);
    }
    
    if (myrank == 0) {
        printf("Total Number of Iterations: %ld\n", k);
        printf("Residual: %.15g\n", error);
    }
}
