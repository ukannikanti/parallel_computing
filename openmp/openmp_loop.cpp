#include <omp.h>
#include <stdio.h>
#include <chrono>
#include <iostream>
#include <thread>

void cpu_timer_start(struct timespec *tstart_cpu) {
    clock_gettime(CLOCK_MONOTONIC, tstart_cpu);
}

double cpu_timer_stop(struct timespec tstart_cpu) {
    struct timespec tstop_cpu, tresult;
    clock_gettime(CLOCK_MONOTONIC, &tstop_cpu);
    tresult.tv_sec = tstop_cpu.tv_sec - tstart_cpu.tv_sec;
    tresult.tv_nsec = tstop_cpu.tv_nsec - tstart_cpu.tv_nsec;
    double result = (double)tresult.tv_sec + (double)tresult.tv_nsec * 1.0e-9;
    return (result);
}
/*
 Worksharing constructs:
    The declaration of a parallel region establishes a team of threads
    This offers the possibility of parallelism but to actually get the meaning
 ful parallelism, You need to split the work across a number of threads or
 processes.

    Openmp provides a worksharing constructs to divid the parallelizable work
 over a team of threads. Below are the useful constructs.

    . For
    . Sections
    . Single
*/

#define ARRAY_SIZE 80000000
static double a[ARRAY_SIZE], b[ARRAY_SIZE], c[ARRAY_SIZE];

void vector_add(double *c, double *a, double *b, int n) {
    for (int i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}

void omp_vector_add(double *c, double *a, double *b, int n) {
#pragma omp parallel for
    for (int i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}

void omp_for_vector_add(double *c, double *a, double *b, int n) {
#pragma omp for
    for (int i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}

void parallel_execution_with_no_loop_optimizations() {
#pragma omp parallel
    {
        if (omp_get_thread_num() == 0)
            printf("Running with %d thread(s)\n", omp_get_num_threads());
    }
    struct timespec tstart;
    double time_sum = 0.0;
    for (int i = 0; i < ARRAY_SIZE; i++) {
        a[i] = 1.0;
        b[i] = 2.0;
    }

    cpu_timer_start(&tstart);
    omp_vector_add(c, a, b, ARRAY_SIZE);
    time_sum += cpu_timer_stop(tstart);

    printf(
        "Runtime for parallel_execution_with_no_loop_optimizations is %lf "
        "msecs\n",
        time_sum);
}

void parallel_execution_with_loop_optimizations_1() {
#pragma omp parallel
    {
        if (omp_get_thread_num() == 0)
            printf("Running with %d thread(s)\n", omp_get_num_threads());
    }
    struct timespec tstart;
    double time_sum = 0.0;
#pragma omp parallel for
    {
        for (int i = 0; i < ARRAY_SIZE; i++) {
            a[i] = 1.0;
            b[i] = 2.0;
        }
    }
    cpu_timer_start(&tstart);
    omp_vector_add(c, a, b, ARRAY_SIZE);
    time_sum += cpu_timer_stop(tstart);

    printf(
        "Runtime for parallel_execution_with_loop_optimizations_1 is %lf "
        "msecs\n",
        time_sum);
}

void parallel_execution_with_loop_optimizations_2() {
#pragma omp parallel
    if (omp_get_thread_num() == 0)
        printf("Running with %d thread(s)\n", omp_get_num_threads());

    struct timespec tstart;
    double time_sum = 0.0;
#pragma omp parallel
    {
#pragma omp for
        for (int i = 0; i < ARRAY_SIZE; i++) {
            a[i] = 1.0;
            b[i] = 2.0;
        }

#pragma omp master
        cpu_timer_start(&tstart);
        omp_for_vector_add(c, a, b, ARRAY_SIZE);
#pragma omp master
        time_sum += cpu_timer_stop(tstart);
    }  // end of omp parallel

    printf("Runtime for parallel_execution_with_loop_optimizations_2 is %lf msecs\n", time_sum);
}

void serial_execution() {
    struct timespec tstart;
    double time_sum = 0.0;
    for (int i = 0; i < ARRAY_SIZE; i++) {
        a[i] = 1.0;
        b[i] = 2.0;
    }
    cpu_timer_start(&tstart);
    vector_add(c, a, b, ARRAY_SIZE);
    time_sum += cpu_timer_stop(tstart);
    printf("Runtime for serial execution is %lf msecs\n", time_sum);
}

int main(int argc, char const *argv[]) {
    serial_execution();
    parallel_execution_with_no_loop_optimizations();
    parallel_execution_with_loop_optimizations_1();
    parallel_execution_with_loop_optimizations_2();
    return 0;
}
