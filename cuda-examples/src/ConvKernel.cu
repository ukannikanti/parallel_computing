#include <stdio.h>

#include <iostream>
#define ROWS 9600
#define COLS 9600

#define OUTPUT_TILE_DIM 30

// filter with same weights.. for testing only!!
#define CONV_FILTER_DIM 3
__constant__ float conv_filter[CONV_FILTER_DIM][CONV_FILTER_DIM];


// shared memory approach. there are 2 tiling strategies to go with.
// 1. Use the block size covers the output tile. [Use multiple steps to load the input tile]
// 2. Use the block size covers the input tile. [Load the input tile in one step, but turnoff some threads when calculating output.]
// Here we follow strategy 2. 
__global__ void conv_kernel_tile(float* input, float* output, unsigned int width, unsigned int height) {
    unsigned int tx = threadIdx.x;
    unsigned int ty = threadIdx.y;
    unsigned int outputRow = blockIdx.y * OUTPUT_TILE_DIM + ty;
    unsigned int outputCol = blockIdx.x * OUTPUT_TILE_DIM + tx;
    __shared__ float INPUT_SM[OUTPUT_TILE_DIM  + CONV_FILTER_DIM - 1][OUTPUT_TILE_DIM  + CONV_FILTER_DIM - 1];

    // load the input tile to shared memory
    int radius = CONV_FILTER_DIM / 2;
    int inputRow = outputRow - radius;
    int inputCol = outputCol - radius;
    if (inputRow >= 0 && inputCol >= 0 && inputRow < height && inputCol < width) {
        INPUT_SM[ty][tx] = input[inputRow * width + inputCol];
    } else {
        INPUT_SM[ty][tx] = 0.0f;
    }
    __syncthreads();

    // comput the convolution for this thread
    float sum = 0.0f;
    // enable only threads that need to compute the output. 
    // for example, input tile dim 4 * 4 & output tile dim 2 * 2, then we have to disable threads that are not in the range of (2,2).
    if (ty < OUTPUT_TILE_DIM && tx < OUTPUT_TILE_DIM) {
        for (unsigned int i = 0; i < CONV_FILTER_DIM; i++) {
            for (unsigned int j = 0; j < CONV_FILTER_DIM; j++) {
                sum += conv_filter[i][j] * INPUT_SM[i + ty][j + tx];
            }
        }
    }

    output[outputRow * width + outputCol] = sum;
}


__global__ void conv_kernel(float* input, float* output, unsigned int width, unsigned int height) {
    // for every pixel in output, apply the convolution and store the result
    // output[row][col] = conv(row, col);
    unsigned int outRow = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int outCol = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int RADIUS = CONV_FILTER_DIM / 2;

    // check for boundary condition.
    float result = 0.0f;
    if (outRow < height && outCol < width) {
        for (int filterRow = 0; filterRow < CONV_FILTER_DIM; filterRow++) {
            for (int filterCol = 0; filterCol < CONV_FILTER_DIM; filterCol++) {
                int currRow = outRow - RADIUS + filterRow;
                int currCol = outCol - RADIUS + filterCol;
                if (currRow < height && currCol < width && currRow >= 0 && currCol >= 0) {
                    result += input[currRow * width + currCol] * conv_filter[filterRow][filterCol];
                }
            }
        }
    }

    output[outRow * width + outCol] = result;
}

void launch_conv_kerenel() {
    float* A = (float*)malloc(ROWS * COLS * sizeof(float));
    float* C = (float*)malloc(ROWS * COLS * sizeof(float));
    float convFilter[3][3] = {{0.1, 0.1, 0.1}, {0.1,0.1,0.1}, {0.1,0.1,0.1}}; 
    for (int i = 0; i < ROWS * COLS; i++) {
        A[i] = 0.2f;
    }
 
    // allocate the memory on device
    float* A_d;
    float* C_d;
    cudaMalloc((void**)&A_d, ROWS * COLS * sizeof(float));
    cudaMalloc((void**)&C_d, ROWS * COLS * sizeof(float));

    // copy the data from host to device
    cudaMemcpy(A_d, A, ROWS * COLS * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(C_d, C, ROWS * COLS * sizeof(float), cudaMemcpyHostToDevice);
    // initialize the const memory from host
    cudaMemcpyToSymbol(conv_filter, convFilter, 9 * sizeof(float));

    // launch the kernel for execution on device
    // dim3 numThreadsPerBlock(32, 32, 1);
    // dim3 numBlocks(ceil(ROWS / 32), ceil(COLS / 32), 1);
    // conv_kernel<<<numBlocks, numThreadsPerBlock>>>(A_d, C_d, COLS, ROWS);
    // cudaDeviceSynchronize();
    // cudaError_t err = cudaGetLastError();
    // if (err != cudaSuccess) 
    //     printf("Error: %s\n", cudaGetErrorString(err));
    
    // For our tiling strategy, our block size should match with the input tile need to compute the output tile. 
    dim3 numThreadsPerBlock(OUTPUT_TILE_DIM + CONV_FILTER_DIM / 2, OUTPUT_TILE_DIM + CONV_FILTER_DIM / 2 , 1);
    dim3 numBlocks(ceil(ROWS / (1.0f * OUTPUT_TILE_DIM)), ceil(COLS / (1.0f * OUTPUT_TILE_DIM)), 1);
    conv_kernel_tile<<<numBlocks, numThreadsPerBlock>>>(A_d, C_d, COLS, ROWS);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) 
        printf("Error: %s\n", cudaGetErrorString(err));
    
    // copy the results back from device to host
    cudaMemcpy(C, C_d, ROWS * COLS * sizeof(float), cudaMemcpyDeviceToHost);
    std::cout << "C[1068] value is: " << C[1068] << std::endl;

    // free the memory. 
    cudaFree(A_d);
    cudaFree(C_d);
    free(A);
    free(C);
}
