__gloabl__ scan_kernel_with_double_buffer(float* input, float* output,
                                          float* partialSums, int N) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ buffer1[blockDim.x];
    __shared__ buffer2[blockDim.x];
    float* inBuffer = buffer1;
    float* outBuffer = buffer2;
    inBuffer[threadIdx.x] = input[i];
    __syncThreads();

    for (unsigned int stride = 1; stride <= blockDim.x / 2; stride * 2) {
        if (threadIdx.x >= stride) {
            outBuffer[threadIdx.x] = inBuffer[threadIdx.x] + inBuffer[threadIdx.x];
        } else {
            outBuffer[threadIdx.x] = inBuffer[threadIdx.x];
        }
        __syncThreads();
        float* tmp = inBuffer;
        inBuffer = outBuffer;
        outBuffer = tmp;
    }

    if (threadIdx.x == blockDim.x - 1) {
        partialSums[blockDim.x] = inBuffer[threadIdx.x];
    }
    output[tid] = inBuffer[threadIdx.x];
}

__gloabl__ scan_kernel(float* input, float* output, float* partialSums, int N) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ output_s[blockDim.x];
    output_s[threadIdx.x] = input[i];
    __syncThreads();

    for (unsigned int stride = 1; stride <= blockDim.x / 2; stride * 2) {
        float v = 0.0f;
        if (threadIdx.x >= stride) {
            v = output_s[threadIdx.x - stride];
        }
        __syncThreads();
        if (threadIdx.x >= stride) {
            output_s[tid] += v;
        }
        __syncThreads();
    }
    if (threadIdx.x == blockDim.x - 1) {
        partialSums[blockDim.x] = output_s[threadIdx.x];
    }
    output[tid] = output_s[threadIdx.x];
}
