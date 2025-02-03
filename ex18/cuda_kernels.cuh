// cuda_kernels.cuh
#ifndef CUDA_KERNELS_CUH
#define CUDA_KERNELS_CUH

#include <cuda_runtime.h>
#include <cstdlib>
#include <stdexcept>
#include <iostream>

// Extern C wrapper function declarations
extern "C" {
    cudaError_t launchConvertToFloat(
        const unsigned char* input, 
        float* output, 
        int width, 
        int height, 
        dim3 gridSize, 
        dim3 blockSize, 
        cudaStream_t stream
    );

    cudaError_t launchHorizontalScan(
        const float* input, 
        float* output, 
        int width, 
        int height, 
        dim3 gridSize, 
        dim3 blockSize, 
        size_t sharedMemSize,
        cudaStream_t stream
    );

    cudaError_t launchVerticalScan(
        const float* input, 
        float* output, 
        int width, 
        int height, 
        dim3 gridSize, 
        dim3 blockSize, 
        size_t sharedMemSize,
        cudaStream_t stream
    );
}

// CUDA error checking macro with exception handling
#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": " \
                      << cudaGetErrorString(err) << std::endl; \
            throw std::runtime_error("CUDA error: " + std::string(cudaGetErrorString(err))); \
        } \
    } while(0)

// Alternative macro for function-level error checking
inline void checkCudaError(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error at " << file << ":" << line << ": " 
                  << cudaGetErrorString(err) << std::endl;
        throw std::runtime_error("CUDA error");
    }
}

// Structure for CUDA resources with improved error handling
struct CUDAResources {
    unsigned char* d_input;
    float* d_integral;
    float* d_temp;
    cudaStream_t stream;
    
    CUDAResources(size_t width, size_t height) {
        try {
            CHECK_CUDA(cudaMalloc(&d_input, width * height * sizeof(unsigned char)));
            CHECK_CUDA(cudaMalloc(&d_integral, width * height * sizeof(float)));
            CHECK_CUDA(cudaMalloc(&d_temp, width * height * sizeof(float)));
            CHECK_CUDA(cudaStreamCreate(&stream));
        } catch (const std::exception& e) {
            cleanup();  // Ensure cleanup on error
            throw;
        }
    }
    
    void cleanup() {
        if (d_input) cudaFree(d_input);
        if (d_integral) cudaFree(d_integral);
        if (d_temp) cudaFree(d_temp);
        if (stream) cudaStreamDestroy(stream);
        
        d_input = nullptr;
        d_integral = nullptr;
        d_temp = nullptr;
        stream = nullptr;
    }
    
    ~CUDAResources() {
        cleanup();
    }
};

#endif // CUDA_KERNELS_CUH