#include <timer.h>

#include <iostream>
#include <kernels.cuh>
#define SIZE_OF_ARRAYS 100000000

// Divide the input into segments
// Each segment size is 2 * blockDim

__global__ void reduction_kernel(float* input, float* partialSums, unsigned int N) {
    unsigned int segment = blockIdx.x * blockDim.x * 2;
    int tid = segment + threadIdx.x;
    // load the first iteration to shared memory
    __shared__ float input_sm[blockDim.x];
    input_sm[tid] = input[tid] + input[tid + blockDim.x];
    __syncthreads();

    // compute reductions
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride / 2) {
        if (tid < stride) {
            input_sm[tid] += input_sm[tid + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        partialSums[blockIdx.x] = input_sm[tid];
    }
}

void launch_reduction_kerenel() {
    int N = SIZE_OF_ARRAYS;
    float* A = (float*)malloc(N * sizeof(float));
    for (int i = 0; i < N; i++) {
        A[i] = 1.2f;
    }
    // input segments caluclations
    unsigned int numThreadsPerBlock = 256;
    unsigned int segmentSize = 2 * numThreadsPerBlock;
    unsigend int numBlocks = ceil(N / (1.0f * segmentSize));

    // Allocate memory on cpu to store the partial sums.
    float* partialSums = (float*)malloc(numBlocks * sizeof(float));

    // Allocate memory on device
    float* A_d;
    float* partialSums_d;
    cudaMalloc((void**)&A_d, N * sizeof(float));
    cudaMalloc((void**)&partialSums_d, numBlocks * sizeof(float));

    // copy the data from host to device
    cudaMemcpy(A_d, A, N * sizeof(float), cudaMemcpyHostToDevice);

    // launch the kernel for execution on device
    reduction_kernel<<<numBlocks, numThreadsPerBlock>>>(A_d, partialSums_d, N);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) 
        printf("Error: %s\n", cudaGetErrorString(err));

    // copy the results from device to host
    cudaMemcpy(partialSums, partialSums_d, N * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(A_d);
    cudaFree(partialSums_d);
    free(A);
    free(partialSums);
}