#include <stdio.h>
#include <timer.h>

#include <iostream>
#include <kernels.cuh>

#define ROWS 9600
#define COLS 9600
#define TILE_DIM 32
#define NUM_STREAMS 32

// A simple matric mul kernel without any optimizations.
__global__ void matrix_mul_kernel(float* A, float* B, float* C, int N) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < N && col < N) {
        // compute the value for C[row][col] per thread
        float sum = 0.0f;
        for (int i = 0; i < N; i++) {
            sum += A[row * N + i] * B[i * N + col];
        }
        C[row * N + col] = sum;
    }
}

__global__ void matrix_mul_kernel_tile(float* A, float* B, float* C, int N) {
    // allocate the shared memory to hold the tile of A & B matrices.
    __shared__ float A_s[TILE_DIM][TILE_DIM];
    __shared__ float B_s[TILE_DIM][TILE_DIM];

    // each thread is responsible for computing a result for one cell.
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int row = by * TILE_DIM + ty;
    int col = bx * TILE_DIM + tx;

    // loop over the tiles and load the data to shared memory
    float result = 0.0f;
    for (unsigned int tile = 0; tile < ceil(N / float(TILE_DIM)); tile++) {
        if (row < N && tile * TILE_DIM + tx < N) {
            A_s[ty][tx] = A[row * N + tile * TILE_DIM + tx];
        }
        if (col < N && tile * TILE_DIM + ty < N) {
            B_s[ty][tx] = B[(tile * TILE_DIM + ty) * N + col];
        }
        __syncthreads();

        for (int k = 0; k < TILE_DIM; k++) {
            result += A_s[ty][k] * B_s[k][tx];
        }
        __syncthreads();
    }
    if (row < N && col < N) {
        C[row * N + col] = result;
    }
}

void launch_matrix_mul_kerenel() {
    struct timespec tstart;
    double time_sum = 0.0;
    float* A = (float*)malloc(ROWS * COLS * sizeof(float));
    float* B = (float*)malloc(ROWS * COLS * sizeof(float));
    float* C = (float*)malloc(ROWS * COLS * sizeof(float));
    cpu_timer_start(&tstart);
    for (int i = 0; i < ROWS * COLS; i++) {
        A[i] = 1.2f;
        B[i] = 2.1f;
    }
    time_sum = cpu_timer_stop(tstart);
    std::cout << "matrices creation time: " << time_sum << std::endl;

    // allocate the memory on device
    cpu_timer_start(&tstart);
    float* A_d;
    float* B_d;
    float* C_d;
    cudaMalloc((void**)&A_d, ROWS * COLS * sizeof(float));
    cudaMalloc((void**)&B_d, ROWS * COLS * sizeof(float));
    cudaMalloc((void**)&C_d, ROWS * COLS * sizeof(float));
    time_sum = cpu_timer_stop(tstart);
    std::cout << "device mem allocation time: " << time_sum << std::endl;

    // copy the data from host to device
    cpu_timer_start(&tstart);
    cudaMemcpy(A_d, A, ROWS * COLS * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B, ROWS * COLS * sizeof(float), cudaMemcpyHostToDevice);
    time_sum = cpu_timer_stop(tstart);
    std::cout << "host to device copy time: " << time_sum << std::endl;

    // launch the kernel for execution on device
    dim3 numThreadsPerBlock(TILE_DIM, TILE_DIM, 1);
    dim3 numBlocks(ceil(ROWS / TILE_DIM), ceil(COLS / TILE_DIM), 1);
    cpu_timer_start(&tstart);
    matrix_mul_kernel<<<numBlocks, numThreadsPerBlock>>>(A_d, B_d, C_d, COLS);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) printf("Error: %s\n", cudaGetErrorString(err));
    time_sum = cpu_timer_stop(tstart);
    std::cout << "compute device time: " << time_sum << std::endl;

    // copy the results back from device to host
    cpu_timer_start(&tstart);
    cudaMemcpy(C, C_d, ROWS * COLS * sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    time_sum = cpu_timer_stop(tstart);
    std::cout << "device to host copy time: " << time_sum << std::endl;

    std::cout << "C[10] value is: " << C[10] << std::endl;
    std::cout << "C[92159999] value is: " << C[92159999] << std::endl;
    std::cout << "C[92160000] value is: " << C[92160000] << std::endl;
}
