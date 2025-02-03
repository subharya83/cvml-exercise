#include "cuda_kernels.cuh"

// Define the CUDA kernels first
__global__ void convertToFloat(const unsigned char* input, float* output, 
                             int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        output[y * width + x] = static_cast<float>(input[y * width + x]);
    }
}

__global__ void horizontalScan(const float* input, float* output, 
                             int width, int height) {
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (y < height) {
        // Use shared memory for efficient scanning
        extern __shared__ float temp[];
        
        // Load data into shared memory
        for (int x = threadIdx.x; x < width; x += blockDim.x) {
            temp[x] = input[y * width + x];
        }
        __syncthreads();
        
        // Perform exclusive scan in shared memory
        for (int offset = 1; offset < width; offset *= 2) {
            float t;
            if (threadIdx.x >= offset) {
                t = temp[threadIdx.x - offset];
            }
            __syncthreads();
            if (threadIdx.x >= offset) {
                temp[threadIdx.x] += t;
            }
            __syncthreads();
        }
        
        // Write results to global memory
        for (int x = threadIdx.x; x < width; x += blockDim.x) {
            output[y * width + x] = temp[x];
        }
    }
}

__global__ void verticalScan(const float* input, float* output, 
                            int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (x < width) {
        extern __shared__ float temp[];
        
        // Load data into shared memory
        for (int y = threadIdx.y; y < height; y += blockDim.y) {
            temp[y] = input[y * width + x];
        }
        __syncthreads();
        
        // Perform exclusive scan in shared memory
        for (int offset = 1; offset < height; offset *= 2) {
            float t;
            if (threadIdx.y >= offset) {
                t = temp[threadIdx.y - offset];
            }
            __syncthreads();
            if (threadIdx.y >= offset) {
                temp[threadIdx.y] += t;
            }
            __syncthreads();
        }
        
        // Write results to global memory
        for (int y = threadIdx.y; y < height; y += blockDim.y) {
            output[y * width + x] = temp[y];
        }
    }
}

// Now define the wrapper functions
extern "C" {
    cudaError_t launchConvertToFloat(
        const unsigned char* input, 
        float* output, 
        int width, 
        int height, 
        dim3 gridSize, 
        dim3 blockSize, 
        cudaStream_t stream
    ) {
        convertToFloat<<<gridSize, blockSize, 0, stream>>>(input, output, width, height);
        return cudaGetLastError();
    }

    cudaError_t launchHorizontalScan(
        const float* input, 
        float* output, 
        int width, 
        int height, 
        dim3 gridSize, 
        dim3 blockSize, 
        size_t sharedMemSize,
        cudaStream_t stream
    ) {
        horizontalScan<<<gridSize, blockSize, sharedMemSize, stream>>>(
            input, output, width, height
        );
        return cudaGetLastError();
    }

    cudaError_t launchVerticalScan(
        const float* input, 
        float* output, 
        int width, 
        int height, 
        dim3 gridSize, 
        dim3 blockSize, 
        size_t sharedMemSize,
        cudaStream_t stream
    ) {
        verticalScan<<<gridSize, blockSize, sharedMemSize, stream>>>(
            input, output, width, height
        );
        return cudaGetLastError();
    }
}