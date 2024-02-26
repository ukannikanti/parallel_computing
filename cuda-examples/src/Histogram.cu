__global__ void histogram_kernel(unsigned char* image, unsigned int* bins,
                                 int width, int height) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < width * height) {
        unsigned char bin = image[i];
        atomicAdd(&bins[bin], 1);
    }
}

// Privitization is an optimization where multiple private
// copies of an output are maintained, then the global copy is updated on
// completion.
__global__ void histogram_kernel_with_privitization(unsigned char* image,
                                                    unsigned int* bins,
                                                    int width, int height,
                                                    unsigned int numBins) {
    // create private copies of the histo[] array for each thread block
    __shared__ unsigned int histo_s[256];
    if (threadIdx.x < 256) histo_s[threadIdx.x] = 0u;
    __syncThreads();

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    atomicAdd(&(histo_s[image[tid]]), 1);
    // wait for all other threads in the block to finish updating the local copy
    __syncthreads();

    if (threadIdx.x < 256) {
        atomicAdd(&(histo[threadIdx.x]), histo_private[threadIdx.x]);
    }
}