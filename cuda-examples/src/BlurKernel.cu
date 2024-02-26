#include <stdio.h>
#include <iostream>

#define BLUR_SIZE 1
// “__global__” keyword indicates that the function being declared is a CUDA C
// kernel function.

// A kernel code to blur the given input image.
// parallelization approach: assign one thread to each output pixel, and have it
// read multiple input pixels.
__global__ void blur_kernel(unsigned char* image_in, unsigned char* image_out, int height, int width) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < height && col < width) {
        int pixVal = 0;
        int pixels = 0;
        // Get the average of the surrounding BLUR_SIZE * BLUR_SIZE box
        for(int blurRow = -BLUR_SIZE; blurRow < BLUR_SIZE + 1; ++blurRow) {
            for(int blurCol = -BLUR_SIZE; blurCol < BLUR_SIZE + 1; ++blurCol) {
                int currRow = row + blurRow;
                int currCol = col + blurCol;
                if (currRow >= 0 && currRow < height && currCol >= 0 && currCol < width) {
                    pixVal += image_in[currRow * width + currCol];
                    pixels++;
                }
            }
        }
        image_out[row * width + col]  = (unsigned char)(pixVal / pixels);
    }
}


void launch_gpu_blur_kernel(unsigned char* image_in, unsigned char* image_out, int height, int width) {
    // allocate the memory on device
    unsigned char* image_in_d;
    unsigned char* image_out_d;
    cudaMalloc((void**)&image_in_d, width * height * sizeof(unsigned char));
    cudaMalloc((void**)&image_out_d, width * height * sizeof(unsigned char));
    cudaDeviceSynchronize();

    // copy the data from host to device
    std::cout << "Copying the data to device memory" << std::endl;
    cudaMemcpy(image_in_d, image_in, width * height * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    
    // launch the kernel for execution on device
    dim3 numThreadsPerBlock(32, 32, 1);
    dim3 numBlocks(ceil(height / 32), ceil(width / 32), 1);
    std::cout << "Launching the kernel!!" << std::endl;
    blur_kernel<<<numBlocks, numThreadsPerBlock>>>(image_in_d, image_out_d, height, width);
    cudaDeviceSynchronize();
    std::cout << "Execution of the kernel is completed!!" << std::endl;
    // copy the results from device to host
    cudaMemcpy(image_out, image_out_d, width * height * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    std::cout << "Copying data back to host is completed!!!" << std::endl;
    
    // free the device memory
    cudaFree(image_in_d);
    cudaFree(image_out_d);
}