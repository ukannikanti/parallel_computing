#include <stdio.h>

// large enough to force into main memory
#define STREAM_ARRAY_SIZE 80000000
static double a[STREAM_ARRAY_SIZE], b[STREAM_ARRAY_SIZE], c[STREAM_ARRAY_SIZE];

/**
 * Compiling with gcc compiler results the following output.
   gcc -g -O3 -fstrict-aliasing -ftree-vectorize -march=native -mtune=native \
      -fopt-info-vec-optimized stream_triad.c

   stream_triad.c:8:19: optimized: loop vectorized using 32 byte vectors
   stream_triad.c:8:19: optimized:  loop versioned for vectorization because of
 possible aliasing stream_triad.c:8:19: optimized: loop vectorized using 32 byte
 vectors stream_triad.c:16:23: optimized: loop vectorized using 32 byte vectors

   The compiler cannot tell if the arguments(pointers) to the function point to
 the same or to overlapping data.

   Aliasing is where pointers point to overlapping regions of memory.
   In this situation, the compiler cannot tell if it is the same memory,
   and it would be unsafe to generate vectorized code or other optimizations.
   For solution see kernel_op_2
*/
void kernel_op_1(double* a, double* b, double* c, double scalar) {
    for (int i = 0; i < STREAM_ARRAY_SIZE; i++) {
        a[i] = b[i] + scalar * c[i];
    }
}

// By using the restrict attribute, you make a promise to the compiler that
// there is no aliasing. Also use the -fstrict-aliasing compiler flag
void kernel_op_2(double* __restrict__ a, double* __restrict__ b,
                 double* __restrict__ c, double scalar) {
    for (int i = 0; i < STREAM_ARRAY_SIZE; i++) {
        a[i] = b[i] + scalar * c[i];
    }
}

int main(int argc, char* argv[]) {
    // initializing data and arrays
    double scalar = 3.0;

    for (int i = 0; i < STREAM_ARRAY_SIZE; i++) {
        a[i] = 1.0;
        b[i] = 2.0;
    }

    kernel_op_2(a, b, c, scalar);
    c[1] = a[1];  // To avoid compiler optimizations.
}