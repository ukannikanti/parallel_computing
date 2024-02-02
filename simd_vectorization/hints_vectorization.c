#include <stdio.h>
#include <math.h>

#define NCELLS 10000000
static double H[NCELLS], U[NCELLS], V[NCELLS], dx[NCELLS], dy[NCELLS];
static int celltype[NCELLS];
#define REAL_CELL 1

/*
  -fopt-info-vec-missed compiler flag to get a report on the missed loop vectorizations.

  gcc -g -O3 -fstrict-aliasing -ftree-vectorize -fopenmp-simd \
  -march=native -mtune=native -mprefer-vector-width=512 \
  -fopt-info-vec-optimized -fopt-info-vec-missed hints_vectorization.c

  Output:
    a.c:12:25: missed: couldn't vectorize loop
    a.c:12:25: missed: not vectorized: control flow in loop.

  This vectorization report tells us that the timestep loop was not vectorized due to the conditional in the loop.
*/

double timestep(int ncells, double g, double sigma, int* restrict celltype,
                double* H, double* U, double* V, double* dx, double* dy) {
    double wavespeed, xspeed, yspeed, dt;
    double mymindt = 1.0e20;
    for (int ic = 0; ic < ncells; ic++) {
        if (celltype[ic] == REAL_CELL) {
            wavespeed = sqrt(g * H[ic]);
            xspeed = (fabs(U[ic]) + wavespeed) / dx[ic];
            yspeed = (fabs(V[ic]) + wavespeed) / dy[ic];
            dt = sigma / (xspeed + yspeed);
            if (dt < mymindt) mymindt = dt;
        }
    }
    return (mymindt);
}

// Let’s see if we can get the loop to optimize by adding a pragma. 
// Also need these compiler flags for gcc to vectorize this code.
// -fno-trapping-math, -fno-math-errno
double timestep_with_hint(int ncells, double g, double sigma, int* restrict celltype,
                double* restrict H, double* restrict U, double*  restrict V, double* restrict dx, double* restrict dy) {
    double mymindt = 1.0e20;

    #pragma omp simd reduction(min:mymindt)
    for (int ic = 0; ic < ncells; ic++) {
        if (celltype[ic] == REAL_CELL) {
            double wavespeed = sqrt(g * H[ic]);
            double xspeed = (fabs(U[ic]) + wavespeed) / dx[ic];
            double yspeed = (fabs(V[ic]) + wavespeed) / dy[ic];
            double dt = sigma / (xspeed + yspeed);
            if (dt < mymindt) mymindt = dt;
        }
    }

    return (mymindt);
}

int main(int argc, char* argv[]) {
    double g = 9.80, sigma = 0.95;
    for (int ic = 0; ic < NCELLS; ic++) {
        H[ic] = 10.0;
        U[ic] = 0.0;
        V[ic] = 0.0;
        dx[ic] = 0.5;
        dy[ic] = 0.5;
        celltype[ic] = REAL_CELL;
    }
    H[NCELLS / 2] = 20.0;

    double mymindt = timestep_with_hint(NCELLS, g, sigma, celltype, H, U, V, dx, dy);

    printf("Minimum dt is %lf\n", mymindt);

    return 0;
}
