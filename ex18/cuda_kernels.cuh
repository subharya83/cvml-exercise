#ifndef CUDA_KERNELS_CUH
#define CUDA_KERNELS_CUH

#include <cuda_runtime.h>
#include <stdio.h>

// CUDA error checking macro
#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                   cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Structure for CUDA resources
struct CUDAResources {
    unsigned char* d_input;
    float* d_integral;
    float* d_temp;
    cudaStream_t stream;
    
    CUDAResources(size_t width, size_t height) {
        CHECK_CUDA(cudaMalloc(&d_input, width * height * sizeof(unsigned char)));
        CHECK_CUDA(cudaMalloc(&d_integral, width * height * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_temp, width * height * sizeof(float)));
        CHECK_CUDA(cudaStreamCreate(&stream));
    }
    
    ~CUDAResources() {
        CHECK_CUDA(cudaFree(d_input));
        CHECK_CUDA(cudaFree(d_integral));
        CHECK_CUDA(cudaFree(d_temp));
        CHECK_CUDA(cudaStreamDestroy(stream));
    }
};

// Kernel declarations
__global__ void convertToFloat(const unsigned char* input, float* output, 
                             int width, int height);
__global__ void horizontalScan(const float* input, float* output, 
                             int width, int height);
__global__ void verticalScan(const float* input, float* output, 
                            int width, int height);

#endif