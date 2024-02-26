#include <stdio.h>
#include <timer.h>

#include <iostream>
#include <kernels.cuh>

#define SIZE_OF_ARRAYS 100000000
#define NUM_STREAMS 32


__global__ void add_kernel(float* A, float* B, float* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N ) {
        for (int i = 0; i < N; i++) {
            C[i] = A[i] + B[i];
        }
    }
}

void launch_vector_add_kerenel_with_streams() {
    int N = SIZE_OF_ARRAYS;
    float* A = (float*)malloc(N * sizeof(float));
    float* B = (float*)malloc(N * sizeof(float));
    float* C = (float*)malloc(N * sizeof(float));
    for (int i = 0; i < N; i++) {
        A[i] = 1.2f;
        B[i] = 2.1f;
    }
    // Allocate memory on device
    float* A_d;
    float* B_d;
    float* C_d;
    cudaMalloc((void**)&A_d, N * sizeof(float));
    cudaMalloc((void**)&B_d, N * sizeof(float));
    cudaMalloc((void**)&C_d, N * sizeof(float));

    // create streams
    cudaStream_t streams[NUM_STREAMS];
    for (unsigned int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamCreate(&streams[i]);
    }

    // segment the input data and assign to different streams
    unsigned int numSegments = NUM_STREAMS;
    unsigned int segmentSize = ceil( N / ((float) numSegments));
    for (int s = 0; s < numSegments; s++) {
        int start = s * segmentSize;
        int end = (start + segmentSize < N) ? start + segmentSize - 1 : N - 1;
        int segLength = end - start;
        // copy data from host to device in segments 
        cudaMemcpyAsync(&A_d[start], &A[start], segLength * sizeof(float), cudaMemcpyHostToDevice, streams[s]);
        cudaMemcpyAsync(&B_d[start], &B[start], segLength * sizeof(float), cudaMemcpyHostToDevice, streams[s]);

        // launch the kernel
        unsigned int numThreadsPerBlock = 512;
        unsigned int numBlocks = ceil(N / (float) numThreadsPerBlock);
        add_kernel<<<numBlocks, numThreadsPerBlock, 0, streams[s]>>>(&A_d[start], &B_d[start], &C_d[start], segmentSize);

        // copy data from device to host in segments
        cudaMemcpyAsync(&C[start], &C_d[start], segLength * sizeof(float), cudaMemcpyDeviceToHost, streams[s]);
    }

    // wait for all streams to be completed.
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) 
        printf("Error: %s\n", cudaGetErrorString(err));

    std::cout << "C[10] value is: " << C[10] << std::endl;
    std::cout << "C[92159999] value is: " << C[92159999] << std::endl;
    std::cout << "C[92160000] value is: " << C[92160000] << std::endl;

    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
}